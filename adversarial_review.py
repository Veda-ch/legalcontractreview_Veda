"""Adversarial Review Agent.

Runs only on clauses risk_analysis.py has already marked "attention_required".
Acts as an opposing reviewer: given ONLY the evidence risk_analysis already
assembled, it may point out a counter-reading that IS present in the clause
text (e.g. a limitation phrase the trigger-matcher missed) or an evidence
gap — but it may never introduce a new risk factor, fact, or citation that
wasn't already in the supplied evidence. Same restricted-LLM philosophy as
risk_analysis.py's explanation layer, reframed as a challenge.

Fails CLOSED, not open: if Ollama is unreachable or returns invalid output,
the original "attention_required" finding is preserved unchanged
(still_concerning=True) rather than silently cleared. An LLM outage must
never be able to silently downgrade a flagged risk.
"""
from __future__ import annotations

import json
from typing import Any

import requests

from interfaces import Clause, Classification, RetrievalResult

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen2.5:3b"

_FALLBACK_NOTE = (
    "Adversarial review unavailable (Ollama unreachable or returned invalid output); "
    "original risk assessment preserved unchanged."
)


def _raw_llm_challenge(payload: dict[str, Any], ollama_url: str) -> dict[str, Any] | None:
    prompt = """You are a restricted adversarial-review module.

You are NOT a legal researcher and NOT a legal decision-maker. Given ONLY the
supplied evidence below, challenge whether the "attention_required" conclusion
still holds up.

STRICT RULES:
1. Do not invent facts, risks, legal rules, case law, citations, or clause text.
2. Do not introduce a risk factor that is absent from the supplied risk_factors.
3. You MAY point out a counter-reading using a limitation/exception/scope phrase
   that IS present verbatim in the supplied clause text.
4. You MAY point out that the supplied evidence is too thin to support the
   conclusion, if that is genuinely the case.
5. Do not change the classification or confidence.
6. Do not infer facts about the parties, jurisdiction, business, money, intent, or enforceability.
7. Cite evidence only by referring to the supplied evidence labels/source titles; never create citations.
8. Return JSON only with: {"still_concerning": true|false, "note": "...", "missed_angle": "..." or null}.
9. Keep the note under 80 words. If the finding holds up, still_concerning must be true.

SUPPLIED EVIDENCE:
"""
    prompt += json.dumps(payload, ensure_ascii=False, indent=2)

    try:
        response = requests.post(
            ollama_url,
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
            },
            timeout=90,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        data = json.loads(raw)
        if "still_concerning" not in data:
            return None
        return {
            "still_concerning": bool(data["still_concerning"]),
            "note": str(data.get("note", "")).strip(),
            "missed_angle": data.get("missed_angle") or None,
        }
    except Exception:
        return None


def review_finding(
    clause: Clause,
    cls: Classification,
    retr: RetrievalResult,
    partial: dict[str, Any],
    ollama_url: str = OLLAMA_URL,
) -> dict[str, Any]:
    """Challenge an 'attention_required' finding. Always returns a dict, never raises."""
    payload = {
        "clause_text": clause.text_en,
        "predicted_type": cls.predicted_type,
        "confidence": cls.confidence,
        "risk_factors": partial.get("risk_factors", []),
        "issues": [i.model_dump() if hasattr(i, "model_dump") else i for i in partial.get("issues", [])],
        "evidence_snippets": [
            {"title": hit.title, "snippet": hit.snippet, "relevance": hit.relevance}
            for hit in retr.results[:5]
        ],
    }

    result = _raw_llm_challenge(payload, ollama_url)
    if result is not None:
        result["source"] = "llm"
        return result

    return {
        "still_concerning": True,
        "note": _FALLBACK_NOTE,
        "missed_angle": None,
        "source": "fallback",
    }


if __name__ == "__main__":
    # Standalone demo / verification (Tier 1).
    from interfaces import TypeScore, Explanation, RetrievalHit

    demo_clause = Clause(
        clause_id="demo_1",
        index=0,
        heading="Indemnification",
        text_original="",
        text_en=(
            "The Vendor shall indemnify, defend, and hold harmless the Client from "
            "any and all claims, losses, damages, and expenses, without limitation."
        ),
        page=1,
        char_span=(0, 0),
    )
    demo_cls = Classification(
        clause_id="demo_1",
        predicted_type="Indemnifications",
        confidence=0.91,
        top_k=[TypeScore(type="Indemnifications", score=0.91)],
        explanation=Explanation(method="attention", salient_tokens=["indemnify", "hold harmless"], rationale="demo"),
        low_conf_baseline=False,
    )
    demo_retr = RetrievalResult(
        query=demo_clause.text_en,
        results=[
            RetrievalHit(
                source_id="src_demo_001",
                title="Risk Reduction Through Indemnification Contract Clauses (1992)",
                snippet="Discusses careful drafting of indemnification language as a risk-allocation mechanism.",
                relevance=0.81,
                graph_path=["indemnification"],
            )
        ],
    )
    demo_partial = {
        "risk_assessment": "attention_required",
        "risk_factors": [
            {
                "factor": "Broad or potentially open-ended indemnification language.",
                "trigger": "any and all",
                "matched_text": "any and all",
                "source_family": "broad_indemnification",
            }
        ],
        "issues": [],
    }

    print("--- Live Ollama call ---")
    live = review_finding(demo_clause, demo_cls, demo_retr, demo_partial)
    print(json.dumps(live, indent=2))
    assert "still_concerning" in live and isinstance(live["still_concerning"], bool)
    print("OK — live call returned a well-formed structured result.\n")

    print("--- Fallback path (dead Ollama port) ---")
    dead = review_finding(demo_clause, demo_cls, demo_retr, demo_partial, ollama_url="http://localhost:1/api/generate")
    print(json.dumps(dead, indent=2))
    assert dead["still_concerning"] is True and dead["source"] == "fallback"
    print("OK — fallback fails closed (preserves the flag) and never raises.\n")

    print("ALL CHECKS PASSED.")
