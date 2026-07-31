"""The walking skeleton: runs all stages in order using whatever implementation
(stub or real) is currently wired in stubs.py.

Run:  python pipeline.py
"""
from __future__ import annotations
import json
import sys
from interfaces import ClauseFinding, ReviewReport, Document
import stubs


def run(file_path: str = "data/sample_document.json") -> tuple[Document, ReviewReport]:
    # Stage 1 — Drashti: parse into clauses.
    # A pre-made .json (e.g. the sample doc) is already a serialized Document
    # and is loaded directly; anything else goes through the real parser.
    if file_path.lower().endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            doc: Document = Document.model_validate(json.load(f))
    else:
        doc = stubs.parse_document(file_path)

    findings: list[ClauseFinding] = []
    for clause in doc.clauses:
        # Stage 2 — Veda: classify
        cls = stubs.classify_clause(clause)

        # Stage 3 — Veda: retrieve legal knowledge
        retr = stubs.retrieve(clause.text_en, cls.predicted_type)

        # Stage 4 — Kashish: analysis agents
        partial = stubs.analyze_clause(clause, cls, retr)

        # Stage 5 — Kashish: RL uncertainty agent decides the review action
        decision = stubs.uncertainty_agent({
            "clause_id": clause.clause_id,
            "confidence": cls.confidence,
            "relevance": max((h.relevance for h in retr.results), default=0.0),
            "agreement": 1.0,  # stub: agreement across analysis agents
            "risk_level": partial["risk_level"],
        })

        findings.append(ClauseFinding(
            **partial,
            review_decision=decision.action,
            review_reason=decision.reason,
        ))

    # Stage 6 — Kashish: assemble report
    return doc, stubs.build_report(doc, findings)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_document.json"
    _doc, report = run(path)
    print(report.model_dump_json(indent=2))
    print("\nOK — walking skeleton ran end-to-end and produced a valid ReviewReport.")
