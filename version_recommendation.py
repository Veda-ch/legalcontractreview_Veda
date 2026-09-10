"""Version Recommendation Agent: recommends the safer contract version.

The DECISION (old / new / equivalent) is deterministic, computed purely from
the RiskComparisonReport tally — it is never gated on LLM availability, since
recommending the wrong version because a local LLM was slow to respond would
be a much worse failure mode than a slightly less articulate explanation.

Only the REASONING TEXT explaining the decision goes through the same
restricted-Ollama pattern as risk_analysis.py / adversarial_review.py
(evidence-only prompt built from the already-computed tally, never allowed
to change the decision or invent new risk factors), falling back to a
deterministic templated sentence if Ollama is unreachable.
"""
from __future__ import annotations

import json
from typing import Any

import requests

from interfaces import RiskComparisonReport, VersionRecommendation

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen2.5:3b"


def _decide(comparison: RiskComparisonReport) -> str:
    if comparison.increased_count > comparison.reduced_count:
        return "old"
    if comparison.reduced_count > comparison.increased_count:
        return "new"
    return "equivalent"


def _fallback_reasoning(comparison: RiskComparisonReport, safer_version: str) -> str:
    verdict = {
        "old": "the previous version is recommended, since more clauses show increased risk than reduced risk",
        "new": "the new version is recommended, since more clauses show reduced risk than increased risk",
        "equivalent": "the two versions are treated as roughly equivalent in risk, since increased and reduced changes balance out",
    }[safer_version]
    return (
        f"{comparison.increased_count} clause(s) show increased risk, "
        f"{comparison.reduced_count} show reduced risk, and "
        f"{comparison.unchanged_count} are unchanged between versions; "
        f"based on this tally, {verdict}."
    )


def _raw_llm_reasoning(comparison: RiskComparisonReport, safer_version: str, ollama_url: str) -> str | None:
    payload = {
        "safer_version": safer_version,
        "increased_count": comparison.increased_count,
        "reduced_count": comparison.reduced_count,
        "unchanged_count": comparison.unchanged_count,
        "clause_changes": [c.model_dump() for c in comparison.clause_comparisons],
    }
    prompt = """You are a restricted version-recommendation explanation module.

The recommendation decision below has ALREADY been made deterministically
from the tally of risk changes; you may NOT change it. Your only job is to
explain WHY, in plain language, using ONLY the supplied tally and clause
changes.

STRICT RULES:
1. Do not invent facts, risks, legal rules, case law, citations, or clause text.
2. Do not introduce a risk factor or clause change absent from the supplied data.
3. Do not change safer_version; just explain it.
4. Do not infer facts about the parties, jurisdiction, business, money, intent, or enforceability.
5. Keep the explanation under 100 words.
6. Return JSON only with: {"reasoning": "..."}.

SUPPLIED DATA:
"""
    prompt += json.dumps(payload, ensure_ascii=False, indent=2)

    try:
        response = requests.post(
            ollama_url,
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False, "format": "json"},
            timeout=90,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        data = json.loads(raw)
        reasoning = data.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()
    except Exception:
        return None
    return None


def recommend_version(comparison: RiskComparisonReport, ollama_url: str = OLLAMA_URL) -> VersionRecommendation:
    safer_version = _decide(comparison)

    llm_reasoning = _raw_llm_reasoning(comparison, safer_version, ollama_url)
    if llm_reasoning:
        reasoning, source = llm_reasoning, "llm"
    else:
        reasoning, source = _fallback_reasoning(comparison, safer_version), "fallback"

    return VersionRecommendation(
        old_doc_id=comparison.old_doc_id,
        new_doc_id=comparison.new_doc_id,
        safer_version=safer_version,
        reasoning=reasoning,
        reasoning_source=source,
    )


if __name__ == "__main__":
    from interfaces import ClauseRiskComparison

    def make_comparison(increased: int, reduced: int, unchanged: int) -> RiskComparisonReport:
        comparisons = (
            [ClauseRiskComparison(change_type="modified", risk_change="increased", explanation="x") for _ in range(increased)]
            + [ClauseRiskComparison(change_type="modified", risk_change="reduced", explanation="x") for _ in range(reduced)]
            + [ClauseRiskComparison(change_type="unchanged", risk_change="unchanged", explanation="x") for _ in range(unchanged)]
        )
        return RiskComparisonReport(
            old_doc_id="d_old", new_doc_id="d_new", clause_comparisons=comparisons,
            increased_count=increased, reduced_count=reduced, unchanged_count=unchanged,
        )

    print("--- Deterministic decision (fallback path, dead Ollama port) ---")
    dead_url = "http://localhost:1/api/generate"

    rec_old = recommend_version(make_comparison(3, 1, 0), ollama_url=dead_url)
    print(f"  3 increased, 1 reduced -> safer_version={rec_old.safer_version} source={rec_old.reasoning_source}")
    assert rec_old.safer_version == "old" and rec_old.reasoning_source == "fallback" and rec_old.reasoning

    rec_new = recommend_version(make_comparison(1, 3, 0), ollama_url=dead_url)
    print(f"  1 increased, 3 reduced -> safer_version={rec_new.safer_version} source={rec_new.reasoning_source}")
    assert rec_new.safer_version == "new"

    rec_equiv = recommend_version(make_comparison(2, 2, 1), ollama_url=dead_url)
    print(f"  2 increased, 2 reduced -> safer_version={rec_equiv.safer_version} source={rec_equiv.reasoning_source}")
    assert rec_equiv.safer_version == "equivalent"
    print("  OK — decision is deterministic and never raises when Ollama is unreachable.\n")

    print("--- Live Ollama call (best-effort) ---")
    rec_live = recommend_version(make_comparison(3, 1, 0))
    print(f"  safer_version={rec_live.safer_version} source={rec_live.reasoning_source}")
    print(f"  reasoning: {rec_live.reasoning}")
    assert rec_live.safer_version == "old", "decision must stay deterministic regardless of LLM availability"
    assert rec_live.reasoning
    print("  OK — decision unaffected by LLM path; reasoning always non-empty.\n")

    print("ALL CHECKS PASSED.")
