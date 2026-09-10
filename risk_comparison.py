"""Risk Comparison Agent: given a VersionDiff plus both versions' findings,
classify each matched/added/removed clause's risk change as increased,
reduced, or unchanged.

Reuses the severity order already implicit in stubs.build_report
(attention_required > insufficient_evidence > no_configured_risk_found) for
consistency, with review_decision as a secondary tiebreak.

Known, deliberate limitation: an added/removed clause is compared against an
implicit "no risk" baseline on the missing side, since there's no finding to
compare against. This means the comparison can't judge clause FAVORABILITY —
e.g. removing a protective liability cap (bad) and removing a risky
obligation (good) both just register as "clause removed, risk_change
determined by whether that removed clause was itself risky." This matches
the rest of the codebase's explicit no-numeric/no-polarity-scoring
philosophy (see risk_analysis.py's docstring) rather than inventing a new
judgment the system isn't equipped to make.
"""
from __future__ import annotations

from interfaces import (
    ClauseFinding,
    ClauseRiskComparison,
    ReviewReport,
    RiskComparisonReport,
    VersionDiff,
)

RISK_RANK = {"no_configured_risk_found": 0, "insufficient_evidence": 1, "attention_required": 2}
REVIEW_RANK = {"accept": 0, "reanalyze": 1, "flag": 2}

_BASELINE_RISK = "no_configured_risk_found"
_BASELINE_REVIEW = "accept"


def _rank_pair(old_assessment: str, new_assessment: str, old_decision: str, new_decision: str) -> str:
    old_rank = RISK_RANK[old_assessment]
    new_rank = RISK_RANK[new_assessment]
    if new_rank > old_rank:
        return "increased"
    if new_rank < old_rank:
        return "reduced"
    # Same risk_assessment rank: break the tie with review_decision.
    old_review_rank = REVIEW_RANK.get(old_decision, 0)
    new_review_rank = REVIEW_RANK.get(new_decision, 0)
    if new_review_rank > old_review_rank:
        return "increased"
    if new_review_rank < old_review_rank:
        return "reduced"
    return "unchanged"


def compare_versions(diff: VersionDiff, old_report: ReviewReport, new_report: ReviewReport) -> RiskComparisonReport:
    old_findings = {f.clause_id: f for f in old_report.clause_findings}
    new_findings = {f.clause_id: f for f in new_report.clause_findings}

    comparisons: list[ClauseRiskComparison] = []

    for entry in diff.entries:
        if entry.change_type in ("modified", "unchanged"):
            old_finding = old_findings.get(entry.old_clause_id)
            new_finding = new_findings.get(entry.new_clause_id)
            if old_finding is None or new_finding is None:
                continue  # no finding to compare (shouldn't normally happen)
            risk_change = _rank_pair(
                old_finding.risk_assessment, new_finding.risk_assessment,
                old_finding.review_decision, new_finding.review_decision,
            )
            explanation = (
                f"{entry.change_type.capitalize()} clause: risk assessment "
                f"{old_finding.risk_assessment} -> {new_finding.risk_assessment}, "
                f"review decision {old_finding.review_decision} -> {new_finding.review_decision}."
            )
            comparisons.append(ClauseRiskComparison(
                change_type=entry.change_type, risk_change=risk_change,
                old_clause_id=entry.old_clause_id, new_clause_id=entry.new_clause_id,
                old_risk_assessment=old_finding.risk_assessment, new_risk_assessment=new_finding.risk_assessment,
                old_review_decision=old_finding.review_decision, new_review_decision=new_finding.review_decision,
                explanation=explanation,
            ))

        elif entry.change_type == "added":
            new_finding = new_findings.get(entry.new_clause_id)
            if new_finding is None:
                continue
            risk_change = _rank_pair(_BASELINE_RISK, new_finding.risk_assessment, _BASELINE_REVIEW, new_finding.review_decision)
            comparisons.append(ClauseRiskComparison(
                change_type="added", risk_change=risk_change,
                new_clause_id=entry.new_clause_id,
                new_risk_assessment=new_finding.risk_assessment, new_review_decision=new_finding.review_decision,
                explanation=f"New clause added with risk assessment {new_finding.risk_assessment}; no prior clause existed to compare against.",
            ))

        elif entry.change_type == "removed":
            old_finding = old_findings.get(entry.old_clause_id)
            if old_finding is None:
                continue
            risk_change = _rank_pair(old_finding.risk_assessment, _BASELINE_RISK, old_finding.review_decision, _BASELINE_REVIEW)
            comparisons.append(ClauseRiskComparison(
                change_type="removed", risk_change=risk_change,
                old_clause_id=entry.old_clause_id,
                old_risk_assessment=old_finding.risk_assessment, old_review_decision=old_finding.review_decision,
                explanation=f"Clause removed (previously had risk assessment {old_finding.risk_assessment}); "
                            "this does not judge whether removing it was itself favorable.",
            ))

    return RiskComparisonReport(
        old_doc_id=diff.old_doc_id,
        new_doc_id=diff.new_doc_id,
        clause_comparisons=comparisons,
        increased_count=sum(c.risk_change == "increased" for c in comparisons),
        reduced_count=sum(c.risk_change == "reduced" for c in comparisons),
        unchanged_count=sum(c.risk_change == "unchanged" for c in comparisons),
    )


if __name__ == "__main__":
    from interfaces import ClauseDiffEntry

    def finding(clause_id: str, risk_assessment: str, review_decision: str) -> ClauseFinding:
        return ClauseFinding(
            clause_id=clause_id, clause_type="Indemnifications", risk_assessment=risk_assessment,
            recommendation="", review_decision=review_decision, review_reason="", confidence=0.9,
            best_relevance=0.8, explanation="test",
        )

    def report(doc_id: str, findings: list[ClauseFinding]) -> ReviewReport:
        return ReviewReport(
            doc_id=doc_id, generated_at="2026-01-01T00:00:00Z", overall_assessment="no_configured_risk_found",
            summary="test", clause_findings=findings, flagged_for_review=[],
        )

    diff = VersionDiff(
        old_doc_id="d_old", new_doc_id="d_new", old_clause_count=3, new_clause_count=3,
        entries=[
            ClauseDiffEntry(change_type="modified", old_clause_id="o1", new_clause_id="n1", similarity=0.7),
            ClauseDiffEntry(change_type="unchanged", old_clause_id="o2", new_clause_id="n2", similarity=0.99),
            ClauseDiffEntry(change_type="added", new_clause_id="n3", similarity=0.0),
            ClauseDiffEntry(change_type="removed", old_clause_id="o3", similarity=0.0),
        ],
    )
    old_report = report("d_old", [
        finding("o1", "no_configured_risk_found", "accept"),
        finding("o2", "attention_required", "flag"),
        finding("o3", "attention_required", "flag"),
    ])
    new_report = report("d_new", [
        finding("n1", "attention_required", "flag"),   # modified: got worse -> increased
        finding("n2", "attention_required", "flag"),   # unchanged: same -> unchanged
        finding("n3", "no_configured_risk_found", "accept"),  # added, low risk -> unchanged
    ])

    result = compare_versions(diff, old_report, new_report)
    by_id = {c.new_clause_id or c.old_clause_id: c for c in result.clause_comparisons}

    print(f"increased={result.increased_count} reduced={result.reduced_count} unchanged={result.unchanged_count}")
    assert by_id["n1"].risk_change == "increased", "modified clause got riskier -> increased"
    assert by_id["n2"].risk_change == "unchanged", "unchanged clause, same risk -> unchanged"
    assert by_id["n3"].risk_change == "unchanged", "added low-risk clause -> unchanged"
    assert by_id["o3"].risk_change == "reduced", "removed a previously-flagged clause -> reduced"
    assert result.increased_count == 1 and result.reduced_count == 1 and result.unchanged_count == 2

    print("ALL CHECKS PASSED.")
