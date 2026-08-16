"""Evidence-first risk analysis for the legal contract review pipeline.

The module deliberately does NOT calculate a numerical risk score.
It combines:
    - LegalBERT classification + confidence
    - GraphRAG relevance
    - original vs English text comparison
    - configured, source-backed risk factors
    - exact clause evidence
    - retrieved legal evidence

Ollama is used only as a restricted explanation layer. It is not allowed to
invent risk factors, legal rules, citations, or facts.
"""
from __future__ import annotations

import json
import re
from typing import Any

import requests

from interfaces import Clause, Classification, RetrievalResult, Issue


OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen2.5:3b"


# These are intentionally evidence-backed, qualitative factors rather than
# weighted scores. The sources justify why the factor is worth reviewing;
# they do NOT define a universal risk level for every contract.
RISK_FACTOR_SOURCES: dict[str, list[dict[str, str]]] = {
    "broad_indemnification": [
        {
            "title": "Indemnification, Monitoring, and Competition: Evidence from R&D Contracts (2022)",
            "url": "https://doi.org/10.1093/aler/ahac004",
            "basis": "Indemnification allocates liability and can affect incentives and liability exposure.",
        },
        {
            "title": "Risk Reduction Through Indemnification Contract Clauses (1992)",
            "url": "https://doi.org/10.1061/(ASCE)9742-597X(1992)8:3(267)",
            "basis": "The paper discusses careful drafting of indemnification language as a risk-allocation mechanism.",
        },
    ],
    "limitation_of_liability": [
        {
            "title": "Contractual Limitations of Liability and their Impact on Tort Claims (2024)",
            "url": "https://doi.org/10.1515/jetl-2024-0004",
            "basis": "Liability limitations are mechanisms for allocating contractual risk and limiting possible claims.",
        },
        {
            "title": "ACORD: An Expert-Annotated Dataset for Legal Contract Clause Retrieval (ACL 2025)",
            "url": "https://aclanthology.org/2025.acl-long.1206/",
            "basis": "The dataset explicitly distinguishes liability caps, damages waivers, warranty disclaimers, and related carve-outs as contract clause categories.",
        },
    ],
    "termination": [
        {
            "title": "10 Steps to Avoid Key Risk Allocation Pitfalls in Commercial Contracts (ACC)",
            "url": "https://www.acc.com/resource-library/10-steps-avoid-key-risk-allocation-pitfalls-commercial-contracts",
            "basis": "The guidance identifies termination conditions, cure periods, notice requirements, and effects of termination as items requiring careful drafting.",
        },
        {
            "title": "Contracting Excellence and Contract Risk Allocation (Journal of Law & Commerce)",
            "url": "https://jlc.law.pitt.edu/ojs/jlc/article/download/174/159",
            "basis": "The article identifies termination and other risk-allocation provisions as important contractual risk terms.",
        },
    ],
    "confidentiality": [
        {
            "title": "Secret recipe: key ingredients of agreements to protect confidential information (2011)",
            "url": "https://doi.org/10.1093/jiplp/jpr008",
            "basis": "Effective confidentiality drafting should define protected information, disclosure limits, duration, parties, and remedies.",
        },
        {
            "title": "The anatomy of a data transfer agreement for health research (2024)",
            "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11383768/",
            "basis": "The review discusses confidentiality duration, exclusions, and legally required disclosures in commercial-style agreements.",
        },
    ],
    "payment_terms": [
        {
            "title": "Governments’ Late Payments and Firms’ Survival: Evidence from the European Union (2021)",
            "url": "https://doi.org/10.1086/713502",
            "basis": "Delayed commercial payments can expose firms to liquidity risks and affect firm outcomes.",
        },
        {
            "title": "Optimal Payment Contracts in Trade Relationships (2023)",
            "url": "https://doi.org/10.1111/iere.12636",
            "basis": "Payment terms allocate residual noncompliance and financing risks between trading parties.",
        },
    ],
}


# Phrase-level triggers are only evidence extractors. They do not carry
# weights and do not by themselves determine a universal risk score.
RISK_FACTORS: dict[str, dict[str, Any]] = {
    "indemnification": {
        "family": "broad_indemnification",
        "triggers": [
            "any and all", "all claims", "all losses", "without limitation",
            "unlimited", "solely responsible", "hold harmless", "defend",
        ],
        "description": "Broad or potentially open-ended indemnification language.",
    },
    "limitation_of_liability": {
        "family": "limitation_of_liability",
        "triggers": [
            "no limitation", "unlimited liability", "without limitation",
            "liable for all", "all damages", "consequential damages",
        ],
        "description": "Liability scope, exclusions, or damages language that merits review.",
    },
    "termination": {
        "family": "termination",
        "triggers": [
            "immediately", "without notice", "sole discretion",
            "automatic renewal", "automatically renew",
        ],
        "description": "Termination or renewal language requiring attention to notice, cure, or discretion.",
    },
    "confidentiality": {
        "family": "confidentiality",
        "triggers": [
            "in perpetuity", "without limitation", "all information",
        ],
        "description": "Confidentiality wording that may warrant review of scope or duration.",
    },
    "payment_terms": {
        "family": "payment_terms",
        "triggers": [
            "immediately due", "non-refundable", "late fee", "interest", "penalty",
        ],
        "description": "Payment wording that may create financial or timing exposure.",
    },
}

LABEL_TO_FAMILY = {
    "Indemnifications": "indemnification",
    "Indemnity": "indemnification",
    "Terminations": "termination",
    "Governing Laws": None,
    "Jurisdictions": None,
    "Costs": "payment_terms",
    "Expenses": "payment_terms",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _text_comparison(original: str, english: str) -> dict[str, Any]:
    original = original or ""
    english = english or ""
    if not original or not english:
        return {
            "available": False,
            "same_text": original == english,
            "original_length": len(original),
            "english_length": len(english),
            "note": "Original/English comparison unavailable for one or both fields.",
        }
    same = _norm(original) == _norm(english)
    return {
        "available": True,
        "same_text": same,
        "original_length": len(original),
        "english_length": len(english),
        "length_difference": abs(len(original) - len(english)),
        "note": (
            "The normalized texts are equivalent."
            if same
            else "The original and English texts differ; review the translation alongside the original before relying on translated wording."
        ),
    }


def _best_relevance(retr: RetrievalResult) -> float:
    return max((hit.relevance for hit in retr.results), default=0.0)


def _retrieval_evidence(retr: RetrievalResult, limit: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "source_id": hit.source_id,
            "title": hit.title,
            "snippet": hit.snippet,
            "relevance": hit.relevance,
            "graph_path": hit.graph_path or [],
        }
        for hit in retr.results[:limit]
    ]


def _source_items(family: str) -> list[dict[str, str]]:
    return RISK_FACTOR_SOURCES.get(family, [])


def _restricted_ollama_explanation(payload: dict[str, Any]) -> str | None:
    prompt = """You are a restricted legal-analysis explanation module.

You are NOT a legal researcher and NOT a legal decision-maker.
You must ONLY explain the evidence supplied in the JSON below.

STRICT RULES:
1. Do not invent facts, risks, legal rules, case law, citations, or clause text.
2. Do not introduce a risk factor that is absent from risk_factors.
3. Do not change the classification or confidence.
4. Do not infer facts about the parties, jurisdiction, business, money, intent, or enforceability.
5. Do not call anything illegal, invalid, unenforceable, or definitely risky unless that exact conclusion is present in the supplied evidence.
6. If evidence is insufficient, explicitly say that the evidence is insufficient.
7. Cite evidence only by referring to the supplied evidence labels/source titles; never create citations.
8. Return JSON only with: {"explanation": "...", "evidence_basis": ["..."]}.
9. Keep the explanation under 120 words.

SUPPLIED EVIDENCE:
"""
    prompt += json.dumps(payload, ensure_ascii=False, indent=2)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=20,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        data = json.loads(raw)
        explanation = data.get("explanation")
        if isinstance(explanation, str) and explanation.strip():
            return explanation.strip()
    except Exception:
        return None
    return None


def analyze_clause(
    clause: Clause,
    cls: Classification,
    retr: RetrievalResult,
) -> dict[str, Any]:
    """Produce evidence-first qualitative analysis with no numeric risk score."""
    clause_type = cls.predicted_type.strip()
    family = LABEL_TO_FAMILY.get(clause_type, clause_type.lower())
    factor_config = RISK_FACTORS.get(family)

    # Some labels do not have a configured source-backed risk family yet.
    detected: list[dict[str, Any]] = []
    text = _norm(clause.text_en)

    if factor_config:
        for trigger in factor_config["triggers"]:
            if trigger in text:
                detected.append({
                    "factor": factor_config["description"],
                    "trigger": trigger,
                    "matched_text": trigger,
                    "source_family": factor_config["family"],
                    "sources": _source_items(factor_config["family"]),
                })

    # Additional safeguard observations are evidence, not scores.
    observations: list[str] = []
    if family == "termination" and ("terminate" in text or "termination" in text) and "notice" not in text:
        observations.append("A termination provision is present, but the analyzed text does not contain the word 'notice'.")
    if family == "confidentiality" and "confidential" in text:
        exceptions = ["required by law", "publicly available", "public information", "independently developed"]
        if not any(x in text for x in exceptions):
            observations.append("No common confidentiality exception phrase was detected in the analyzed text.")
    if family == "indemnification" and any(x in text for x in ["indemnify", "indemnity", "hold harmless"]):
        limits = ["limited to", "liability cap", "maximum liability", "to the extent"]
        if not any(x in text for x in limits):
            observations.append("No common scope-limit phrase was detected in the analyzed indemnification text.")
    if family == "limitation_of_liability" and "liability" in text:
        limits = ["liability cap", "limited to", "maximum liability", "fees paid"]
        if not any(x in text for x in limits):
            observations.append("No common liability-cap phrase was detected in the analyzed text.")

    best_relevance = _best_relevance(retr)
    comparison = _text_comparison(clause.text_original, clause.text_en)

    # Evidence-backed status. No arbitrary numeric risk range is used.
    if detected:
        assessment = "attention_required"
        assessment_basis = "One or more configured, source-backed risk factors were directly matched in the clause text."
    elif not retr.results or best_relevance < 0.45:
        assessment = "insufficient_evidence"
        assessment_basis = "No configured source-backed factor was matched and the retrieved legal evidence is weak or absent."
    else:
        assessment = "no_configured_risk_found"
        assessment_basis = "No configured source-backed risk factor was matched in the clause text. This is not a statement that the clause is legally risk-free."

    issues: list[Issue] = []
    for item in detected:
        issues.append(Issue(
            category="risk",
            severity="high" if len(detected) >= 2 else "medium",
            description=item["factor"],
            evidence=[item["matched_text"]] + [s["title"] for s in item["sources"]],
        ))
    for observation in observations:
        issues.append(Issue(
            category="compliance",
            severity="medium",
            description=observation,
            evidence=[clause.text_en[:500]],
        ))

    recommendations = []
    if detected:
        recommendations.append("Review the matched provision against the cited source-backed risk factor and the surrounding contract context.")
    if observations:
        recommendations.append("Verify whether the surrounding agreement contains the safeguard or limitation that is not visible in this clause.")
    if comparison["available"] and not comparison["same_text"]:
        recommendations.append("Review the original-language text alongside the English text because the two versions differ.")
    if not recommendations:
        recommendations.append("No configured evidence-backed risk factor was found; review the clause in the context of the complete agreement.")

    evidence_payload = {
        "clause_text": clause.text_en,
        "original_text": clause.text_original,
        "classification": {
            "predicted_type": cls.predicted_type,
            "confidence": cls.confidence,
            "top_k": [x.model_dump() for x in cls.top_k],
        },
        "retrieval": {
            "best_relevance": best_relevance,
            "results": _retrieval_evidence(retr),
        },
        "text_comparison": comparison,
        "risk_factors": detected,
        "observations": observations,
        "assessment": assessment,
        "assessment_basis": assessment_basis,
    }

    llm_explanation = _restricted_ollama_explanation(evidence_payload)
    if not llm_explanation:
        llm_explanation = (
            f"{assessment_basis} Classification confidence is {cls.confidence:.2f}; "
            f"best retrieval relevance is {best_relevance:.2f}. "
            f"The analysis is limited to the supplied clause, retrieval evidence, and configured source-backed factors."
        )

    return {
        "clause_id": clause.clause_id,
        "clause_type": cls.predicted_type,
        "risk_assessment": assessment,
        "risk_factors": detected,
        "issues": issues,
        "recommendation": " ".join(recommendations),
        "confidence": round(float(cls.confidence), 4),
        "best_relevance": round(best_relevance, 4),
        "text_comparison": comparison,
        "evidence": evidence_payload,
        "explanation": llm_explanation,
    }
