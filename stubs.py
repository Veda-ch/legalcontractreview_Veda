
"""Connected implementations for the legal contract review pipeline.

The file keeps the original stage interfaces used by pipeline.py while
replacing the old arbitrary 0-100 risk scoring with evidence-first analysis.
"""
from __future__ import annotations

from datetime import datetime, timezone

from interfaces import (
    Document,
    Clause,
    Classification,
    RetrievalResult,
    UncertaintyDecision,
    ClauseFinding,
    ReviewReport,
)
from parsing import parse_document
from classifier import classify_clause
from legal_kb import retrieve
from risk_analysis import analyze_clause as _analyze_clause
from uncertainty_bandit import uncertainty_agent as _uncertainty_agent


def uncertainty_agent(features: dict) -> UncertaintyDecision:
    """Stage 5 — RL-based Uncertainty Verification Agent (contextual bandit).
    See uncertainty_bandit.py for the policy itself; this delegates so any
    remaining direct `stubs.uncertainty_agent` callers keep working."""
    return _uncertainty_agent(features)


def analyze_clause(clause: Clause, cls: Classification, retr: RetrievalResult) -> dict:
    return _analyze_clause(clause, cls, retr)


def build_report(doc: Document, findings: list[ClauseFinding]) -> ReviewReport:
    attention = sum(f.risk_assessment == "attention_required" for f in findings)
    insufficient = sum(f.risk_assessment == "insufficient_evidence" for f in findings)
    flagged = [f.clause_id for f in findings if f.review_decision in {"flag", "reanalyze"}]

    if attention:
        overall = "attention_required"
        summary = (
            f"The contract contains {len(findings)} analyzed clause(s). "
            f"{attention} clause(s) contain configured, source-backed risk factors. "
            f"{insufficient} clause(s) have insufficient supporting evidence, and "
            f"{len(flagged)} clause(s) require human review or reanalysis."
        )
    elif insufficient:
        overall = "insufficient_evidence"
        summary = (
            f"The contract contains {len(findings)} analyzed clause(s), but the available "
            f"evidence is insufficient for a stronger contract-level qualitative assessment. "
            f"{len(flagged)} clause(s) require human review or reanalysis."
        )
    else:
        overall = "no_configured_risk_found"
        summary = (
            f"The contract contains {len(findings)} analyzed clause(s). No configured, "
            f"source-backed risk factor was detected in the analyzed clauses. This does "
            f"not mean the contract is legally risk-free."
        )

    return ReviewReport(
        doc_id=doc.doc_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        overall_assessment=overall,
        summary=summary,
        clause_findings=findings,
        flagged_for_review=flagged,
        metadata={
            "analysis_method": "evidence-first qualitative analysis",
            "numeric_risk_score": False,
            "llm_role": "restricted explanation only; not a risk-factor generator",
            "retrieval_method": "dataset-backed GraphRAG",
            "datasets_in_knowledge_base": ["LEDGAR", "CUAD", "MAUD"],
        },
    )



# """Stub implementations for every stage.

# Each function returns a VALID instance of its contract so the whole pipeline
# runs end-to-end today. Replace the body of your stub with real code when you
# reach your stage. Keep the signature and return type identical.

#   Stage 1  parse_document      -> Drashti
#   Stage 2  classify_clause     -> Veda
#   Stage 3  retrieve            -> Veda
#   Stage 4  analyze_clause      -> Kashish
#   Stage 5  uncertainty_agent   -> Kashish
#   Stage 6  build_report        -> Kashish
# """
# from __future__ import annotations
# from datetime import datetime, timezone
# from interfaces import (
#     Document, Clause, Classification, TypeScore, Explanation,
#     RetrievalResult, RetrievalHit, UncertaintyDecision,
#     Issue, ClauseFinding, ReviewReport,
# )
# from parsing import parse_document

# # Types the classifier treats as inherently higher risk (stub heuristic).
# HIGH_RISK_TYPES = {"indemnification", "limitation_of_liability"}


# # ---------- Stage 1 (Drashti): parsing ----------
# # parse_document is now the real implementation, imported from parsing.py.


# # ---------- Stage 2 (Veda): classification ----------
# def classify_clause(clause: Clause) -> Classification:
#     """STUB: keyword guess. Replace with fine-tuned LegalBERT + real XAI."""
#     text = clause.text_en.lower()
#     if "indemnif" in text:
#         ptype, conf = "indemnification", 0.74
#     elif "terminat" in text or "renew" in text:
#         ptype, conf = "termination", 0.88
#     else:
#         ptype, conf = "other", 0.55
#     return Classification(
#         clause_id=clause.clause_id,
#         predicted_type=ptype,
#         confidence=conf,
#         top_k=[TypeScore(type=ptype, score=conf)],
#         explanation=Explanation(
#             method="attention",
#             salient_tokens=[w for w in ("indemnify", "terminate", "renew") if w[:6] in text],
#             rationale="Stub keyword match; replace with model attributions.",
#         ),
#         low_conf_baseline=conf < 0.60,
#     )


# # ---------- Stage 3 (Veda): GraphRAG retrieval ----------
# def retrieve(query: str, clause_type: str | None = None, top_k: int = 5) -> RetrievalResult:
#     """STUB: canned hit. Replace with GraphRAG over the legal knowledge graph."""
#     return RetrievalResult(
#         query=query,
#         results=[
#             RetrievalHit(
#                 source_id="src_stub_001",
#                 title="Stub legal source",
#                 snippet="Placeholder passage relevant to the clause.",
#                 relevance=0.70,
#                 graph_path=[clause_type or "unknown", "stub_node"],
#             )
#         ],
#     )


# # ---------- Stage 4 (Kashish): analysis agents ----------
# def analyze_clause(clause: Clause, cls: Classification, retr: RetrievalResult) -> dict:
#     """STUB: returns finding fields EXCEPT the review decision (that comes from
#     the RL uncertainty agent in stage 5). Replace with risk/compliance/loophole
#     agents orchestrated by the supervisor."""
#     high_risk = cls.predicted_type in HIGH_RISK_TYPES
#     issues = []
#     if high_risk:
#         issues.append(Issue(
#             category="risk",
#             severity="high",
#             description="Potentially uncapped or one-sided obligation.",
#             evidence=[h.source_id for h in retr.results],
#         ))
#     return {
#         "clause_id": clause.clause_id,
#         "clause_type": cls.predicted_type,
#         "risk_level": "high" if high_risk else "low",
#         "issues": issues,
#         "recommendation": "Add a liability cap." if high_risk else "No action needed.",
#         "confidence": cls.confidence,
#         "explanation": "Stub analysis based on clause type.",
#     }


# # ---------- Stage 5 (Kashish): RL uncertainty agent ----------
# def uncertainty_agent(features: dict) -> UncertaintyDecision:
#     """STUB: heuristic policy. Replace with the trained RL policy.
#     features = {clause_id, confidence, relevance, agreement, risk_level}"""
#     risky = features["risk_level"] == "high"
#     unsure = features["confidence"] < 0.80
#     if risky and unsure:
#         action, reason = "flag", "High-risk clause with only moderate model confidence."
#     elif unsure:
#         action, reason = "reanalyze", "Low confidence; rerun analysis."
#     else:
#         action, reason = "accept", "Confident, low-risk finding."
#     return UncertaintyDecision(
#         clause_id=features["clause_id"],
#         action=action,
#         policy_confidence=0.80,
#         reason=reason,
#     )


# # ---------- Stage 6 (Kashish): report ----------
# def build_report(doc: Document, findings: list[ClauseFinding]) -> ReviewReport:
#     """STUB: assembles the report. Replace with real scoring + summary generation."""
#     order = {"low": 0, "medium": 1, "high": 2}
#     overall = max((f.risk_level for f in findings), key=lambda r: order[r], default="low")
#     score = round(100 * sum(order[f.risk_level] for f in findings) / (2 * max(len(findings), 1)), 1)
#     flagged = [f.clause_id for f in findings if f.review_decision == "flag"]
#     return ReviewReport(
#         doc_id=doc.doc_id,
#         generated_at=datetime.now(timezone.utc).isoformat(),
#         overall_risk=overall,
#         overall_score=score,
#         summary=f"{len(flagged)} clause(s) flagged for human review out of {len(findings)}.",
#         clause_findings=findings,
#         flagged_for_review=flagged,
#         metadata={
#             "agents_run": ["risk", "compliance", "loophole", "recommendation", "uncertainty_rl"],
#             "language": doc.source_language,
#             "processing_time_sec": 0.0,
#         },
#     )
