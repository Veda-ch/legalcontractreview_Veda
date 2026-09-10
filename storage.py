"""Database Storage: persists every completed analysis and finds whether a
freshly-parsed Document is a later version of one already stored.

SQLite, one short-lived connection per call (no module-level cached
connection like legal_kb.py's FAISS singleton — that pattern exists because
loading FAISS/graph data is expensive; sqlite3.connect() is cheap and
file-backed, so there's nothing to gain by caching, and caching a connection
across Streamlit reruns would be the riskier choice).

Version matching is automatic and content-based (word-level text similarity
over clause text + heading, with filename as a minor secondary signal) —
never a guess: an ambiguous or below-threshold candidate returns no match at
all rather than a wrong auto-link. See text_similarity.py for the primitive.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

from interfaces import Document, ReviewReport
from text_similarity import word_ratio, char_ratio, best_match

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "contract_review.db")

# A best-matching stored document must clear this similarity score...
DOC_MATCH_THRESHOLD = 0.35
# ...and beat the runner-up candidate by at least this margin, or it's
# treated as ambiguous and no match is returned.
DOC_MATCH_MARGIN = 0.08

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    source_language TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    clause_count INTEGER NOT NULL,
    document_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reports (
    doc_id TEXT PRIMARY KEY REFERENCES documents(doc_id),
    overall_assessment TEXT NOT NULL,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def _document_text(doc: Document) -> str:
    parts = []
    for clause in doc.clauses:
        if clause.heading:
            parts.append(clause.heading)
        parts.append(clause.text_en)
    return " ".join(parts)


def save_analysis(doc: Document, report: ReviewReport, db_path: str = DB_PATH) -> None:
    """Idempotent: re-saving the same doc_id overwrites the prior row."""
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        with conn:
            conn.execute(
                """INSERT OR REPLACE INTO documents
                   (doc_id, filename, source_language, page_count, clause_count, document_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (doc.doc_id, doc.filename, doc.source_language, doc.page_count,
                 len(doc.clauses), doc.model_dump_json(), now),
            )
            conn.execute(
                """INSERT OR REPLACE INTO reports
                   (doc_id, overall_assessment, report_json, created_at)
                   VALUES (?, ?, ?, ?)""",
                (doc.doc_id, report.overall_assessment, report.model_dump_json(), now),
            )
    finally:
        conn.close()


def list_analyses(db_path: str = DB_PATH) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """SELECT d.doc_id, d.filename, d.page_count, d.clause_count, d.created_at,
                      r.overall_assessment
               FROM documents d LEFT JOIN reports r ON r.doc_id = d.doc_id
               ORDER BY d.created_at DESC"""
        ).fetchall()
    finally:
        conn.close()
    columns = ["doc_id", "filename", "page_count", "clause_count", "created_at", "overall_assessment"]
    return [dict(zip(columns, row)) for row in rows]


def load_analysis(doc_id: str, db_path: str = DB_PATH) -> tuple[Document, ReviewReport] | None:
    conn = _connect(db_path)
    try:
        doc_row = conn.execute(
            "SELECT document_json FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        report_row = conn.execute(
            "SELECT report_json FROM reports WHERE doc_id = ?", (doc_id,)
        ).fetchone()
    finally:
        conn.close()
    if doc_row is None or report_row is None:
        return None
    return Document.model_validate_json(doc_row[0]), ReviewReport.model_validate_json(report_row[0])


def find_best_matching_document(new_doc: Document, db_path: str = DB_PATH) -> tuple[str, float] | None:
    """Compare `new_doc` against every stored document by content similarity
    (with filename as a minor secondary signal) and return the best match's
    doc_id + score, or None if there's no confident match. Call this BEFORE
    save_analysis(new_doc, ...), so the new document can't match itself."""
    conn = _connect(db_path)
    try:
        rows = conn.execute("SELECT doc_id, filename, document_json FROM documents").fetchall()
    finally:
        conn.close()
    if not rows:
        return None

    new_text = _document_text(new_doc)
    scores: dict[str, float] = {}
    for doc_id, filename, document_json in rows:
        stored_doc = Document.model_validate_json(document_json)
        content_score = word_ratio(new_text, _document_text(stored_doc))
        filename_score = char_ratio(new_doc.filename, filename)
        scores[doc_id] = 0.85 * content_score + 0.15 * filename_score

    return best_match(scores, DOC_MATCH_THRESHOLD, DOC_MATCH_MARGIN)


if __name__ == "__main__":
    import tempfile
    from interfaces import Clause

    def make_doc(doc_id: str, filename: str, clause_texts: list[tuple[str, str]]) -> Document:
        return Document(
            doc_id=doc_id, filename=filename, source_language="en", page_count=1,
            clauses=[
                Clause(clause_id=f"{doc_id}_c{i}", index=i, heading=heading, text_original="",
                       text_en=text, page=1, char_span=(0, 0))
                for i, (heading, text) in enumerate(clause_texts)
            ],
        )

    def make_report(doc: Document, overall: str = "no_configured_risk_found") -> ReviewReport:
        return ReviewReport(
            doc_id=doc.doc_id, generated_at="2026-01-01T00:00:00Z",
            overall_assessment=overall, summary="test", clause_findings=[], flagged_for_review=[],
        )

    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(db_path)  # let _connect create it fresh
    try:
        original = make_doc("d_original", "vendor_agreement_v1.pdf", [
            ("Indemnification", "The Vendor shall indemnify, defend, and hold harmless the Client from any and all claims."),
            ("Termination", "Either party may terminate this Agreement upon thirty days written notice."),
        ])
        save_analysis(original, make_report(original), db_path=db_path)

        print("--- list_analyses ---")
        rows = list_analyses(db_path=db_path)
        print(rows)
        assert len(rows) == 1 and rows[0]["doc_id"] == "d_original"

        print("--- load_analysis round-trip ---")
        loaded = load_analysis("d_original", db_path=db_path)
        assert loaded is not None
        loaded_doc, loaded_report = loaded
        assert loaded_doc.model_dump() == original.model_dump()
        assert loaded_report.model_dump() == make_report(original).model_dump()
        print("OK — round-trip is byte-for-byte identical.\n")

        print("--- find_best_matching_document: light edit should match ---")
        light_edit = make_doc("d_v2", "vendor_agreement_v2.pdf", [
            ("Indemnification", "The Vendor shall indemnify, defend, and hold harmless the Client from any and all claims and losses."),
            ("Termination", "Either party may terminate this Agreement upon sixty days written notice."),
        ])
        match = find_best_matching_document(light_edit, db_path=db_path)
        print(f"  match = {match}")
        assert match is not None and match[0] == "d_original"
        print("  OK\n")

        print("--- find_best_matching_document: unrelated document should NOT match ---")
        unrelated = make_doc("d_unrelated", "employment_offer.pdf", [
            ("Compensation", "Employee shall receive an annual base salary payable in accordance with standard payroll practices."),
            ("Benefits", "Employee shall be eligible to participate in the Company's standard benefits programs."),
        ])
        match2 = find_best_matching_document(unrelated, db_path=db_path)
        print(f"  match = {match2}")
        assert match2 is None
        print("  OK\n")

        print("ALL CHECKS PASSED.")
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
        for suffix in ("-wal", "-shm"):
            if os.path.exists(db_path + suffix):
                os.remove(db_path + suffix)
