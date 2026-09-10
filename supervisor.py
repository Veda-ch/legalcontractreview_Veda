"""Supervisor Agent — LangGraph orchestration layer.

Replaces the old fixed sequence (classify -> retrieve -> analyze ->
uncertainty) with conditional routing:

  classify -> retrieve -> analyze
  analyze  -> [attention_required?]      -> adversarial review -> uncertainty
  analyze  -> [weak evidence, 1 retry]   -> retrieve (broadened) -> analyze
  analyze  -> [otherwise]                -> uncertainty (always runs; it must
                                             produce review_decision, a
                                             required ClauseFinding field)
  uncertainty -> finalize -> ClauseFinding
"""
from __future__ import annotations

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from interfaces import Clause, Classification, RetrievalResult, UncertaintyDecision, ClauseFinding
from classifier import classify_clause
from legal_kb import retrieve
from risk_analysis import analyze_clause
from adversarial_review import review_finding
from uncertainty_bandit import uncertainty_agent

RELEVANCE_RETRY_THRESHOLD = 0.45
MAX_RETRIEVAL_ATTEMPTS = 1  # bounded retry: 1 retry beyond the initial attempt


class ClauseState(TypedDict, total=False):
    clause: Clause
    cls: Classification
    retr: RetrievalResult
    retrieval_attempts: int
    partial: dict
    adversarial: dict | None
    decision: UncertaintyDecision
    finding: ClauseFinding


def _classify_node(state: ClauseState) -> dict:
    return {"cls": classify_clause(state["clause"])}


def _retrieve_node(state: ClauseState) -> dict:
    attempts = state.get("retrieval_attempts", 0)
    clause = state["clause"]
    cls = state["cls"]
    if attempts == 0:
        retr = retrieve(clause.text_en, cls.predicted_type, top_k=5)
    else:
        # Broaden the search: drop graph-neighborhood narrowing, widen top_k.
        # A literal retry with identical args would be a no-op since
        # retrieval is deterministic.
        retr = retrieve(clause.text_en, None, top_k=10)
    return {"retr": retr, "retrieval_attempts": attempts + 1}


def _analyze_node(state: ClauseState) -> dict:
    partial = analyze_clause(state["clause"], state["cls"], state["retr"])
    return {"partial": partial}


def _adversarial_node(state: ClauseState) -> dict:
    adversarial = review_finding(state["clause"], state["cls"], state["retr"], state["partial"])
    return {"adversarial": adversarial}


def _uncertainty_node(state: ClauseState) -> dict:
    partial = state["partial"]
    adversarial = state.get("adversarial")
    agreement = 1.0
    if adversarial is not None:
        agreement = 0.0 if adversarial.get("still_concerning") else 1.0

    features = {
        "clause_id": state["clause"].clause_id,
        "confidence": state["cls"].confidence,
        "relevance": partial["best_relevance"],
        "risk_assessment": partial["risk_assessment"],
        "agreement": agreement,
    }
    return {"decision": uncertainty_agent(features)}


def _finalize_node(state: ClauseState) -> dict:
    partial = dict(state["partial"])
    decision = state["decision"]
    finding = ClauseFinding(
        **partial,
        review_decision=decision.action,
        review_reason=decision.reason,
        adversarial_review=state.get("adversarial"),
    )
    return {"finding": finding}


def _route_after_retrieve(state: ClauseState) -> str:
    return "analyze"


def _route_after_analyze(state: ClauseState) -> str:
    partial = state["partial"]
    attempts = state.get("retrieval_attempts", 0)

    if partial["best_relevance"] < RELEVANCE_RETRY_THRESHOLD and attempts <= MAX_RETRIEVAL_ATTEMPTS:
        return "retry"
    if partial["risk_assessment"] == "attention_required":
        return "adversarial"
    return "uncertainty"


def _build_graph() -> StateGraph:
    g = StateGraph(ClauseState)
    g.add_node("classify", _classify_node)
    g.add_node("retrieve", _retrieve_node)
    g.add_node("analyze", _analyze_node)
    g.add_node("adversarial", _adversarial_node)
    g.add_node("uncertainty", _uncertainty_node)
    g.add_node("finalize", _finalize_node)

    g.add_edge(START, "classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "analyze")
    g.add_conditional_edges(
        "analyze",
        _route_after_analyze,
        {"retry": "retrieve", "adversarial": "adversarial", "uncertainty": "uncertainty"},
    )
    g.add_edge("adversarial", "uncertainty")
    g.add_edge("uncertainty", "finalize")
    g.add_edge("finalize", END)
    return g


_compiled = _build_graph().compile()


def run_clause(clause: Clause) -> ClauseFinding:
    result = _compiled.invoke({"clause": clause, "retrieval_attempts": 0})
    return result["finding"]


if __name__ == "__main__":
    # Self-test with classify/retrieve monkeypatched to fixed fake values —
    # no legalbert_finetuned/ or FAISS index required (Tier 1 verification).
    from unittest.mock import patch
    from interfaces import TypeScore, Explanation, RetrievalHit

    def fake_classify(clause: Clause, *, confidence: float, predicted_type: str = "Indemnifications") -> Classification:
        return Classification(
            clause_id=clause.clause_id,
            predicted_type=predicted_type,
            confidence=confidence,
            top_k=[TypeScore(type=predicted_type, score=confidence)],
            explanation=Explanation(method="attention", salient_tokens=[], rationale="fake"),
            low_conf_baseline=confidence < 0.60,
        )

    def fake_retrieve_factory(relevance: float):
        def _fake_retrieve(query, clause_type, top_k=5):
            return RetrievalResult(
                query=query,
                results=[
                    RetrievalHit(
                        source_id="src_fake",
                        title="Fake source",
                        snippet="fake snippet",
                        relevance=relevance,
                        graph_path=[clause_type or "unknown"],
                    )
                ],
            )
        return _fake_retrieve

    # No risk-trigger phrases in this text, so risk_analysis.analyze_clause
    # (run for real — only classify/retrieve are mocked) will never mark it
    # attention_required, regardless of the injected confidence/relevance.
    no_risk_clause = Clause(
        clause_id="c_000", index=0, heading="Governing Law",
        text_original="", text_en="This Agreement shall be governed by the laws of the State of Delaware.",
        page=1, char_span=(0, 0),
    )

    demo_clause = Clause(
        clause_id="c_001", index=0, heading=None,
        text_original="", text_en="The Vendor shall indemnify and hold harmless the Client from any and all claims.",
        page=1, char_span=(0, 0),
    )

    print("--- Case 1: low confidence, no risk trigger -> adversarial skipped, decision=reanalyze ---")
    # Patch via __name__ (resolves to "__main__" when run directly with
    # `python supervisor.py`, or "supervisor" when imported) so the patch
    # lands on the module namespace the compiled graph's nodes actually read
    # from at call time — hardcoding "supervisor" here would silently patch
    # a second, unrelated copy of the module when run as a script.
    with patch(f"{__name__}.classify_clause", lambda c: fake_classify(c, confidence=0.40, predicted_type="Governing Laws")), \
         patch(f"{__name__}.retrieve", fake_retrieve_factory(relevance=0.9)):
        finding1 = run_clause(no_risk_clause)
    print(f"  review_decision={finding1.review_decision}  adversarial_review={finding1.adversarial_review}")
    assert finding1.review_decision == "reanalyze"
    assert finding1.adversarial_review is None
    print("  OK\n")

    print("--- Case 2: attention_required + high confidence/relevance -> adversarial runs ---")
    with patch(f"{__name__}.classify_clause", lambda c: fake_classify(c, confidence=0.92)), \
         patch(f"{__name__}.retrieve", fake_retrieve_factory(relevance=0.85)):
        finding2 = run_clause(demo_clause)
    print(f"  risk_assessment={finding2.risk_assessment}  review_decision={finding2.review_decision}")
    print(f"  adversarial_review={finding2.adversarial_review}")
    assert finding2.risk_assessment == "attention_required"
    assert finding2.adversarial_review is not None
    print("  OK\n")

    print("--- Case 3: low relevance -> exactly one bounded retry, broadened params ---")
    call_log = []
    original_retrieve = fake_retrieve_factory(relevance=0.2)

    def logging_retrieve(query, clause_type, top_k=5):
        call_log.append({"clause_type": clause_type, "top_k": top_k})
        return original_retrieve(query, clause_type, top_k)

    with patch(f"{__name__}.classify_clause", lambda c: fake_classify(c, confidence=0.92)), \
         patch(f"{__name__}.retrieve", logging_retrieve):
        finding3 = run_clause(demo_clause)
    print(f"  retrieve() call log: {call_log}")
    assert len(call_log) == 2, f"expected exactly 2 retrieve calls (initial + 1 retry), got {len(call_log)}"
    assert call_log[0]["clause_type"] == "Indemnifications" and call_log[0]["top_k"] == 5
    assert call_log[1]["clause_type"] is None and call_log[1]["top_k"] == 10
    print("  OK — retried exactly once with broadened parameters, did not loop.\n")

    print("ALL CHECKS PASSED.")
