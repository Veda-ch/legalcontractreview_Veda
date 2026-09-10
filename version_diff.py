"""Version Comparison Agent: diffs two Documents at the clause level.

Because the old and new Document were parsed independently, their clause_ids
are unrelated (see parsing.py — doc_id is a fresh uuid4 on every parse) — so
clauses must be matched by CONTENT, not by id.

Matching algorithm: global greedy over the full old x new similarity matrix
(score every pair, consume highest-scoring pairs first, provided both sides
are still unmatched). This is deliberately not naive per-row greedy ("for
each new clause, take the best remaining old clause in list order"), which
is order-dependent — an early clause can steal the old clause that would
have been the better match for a later one, producing avoidable mispairs.
"""
from __future__ import annotations

from interfaces import Clause, Document, ClauseDiffEntry, VersionDiff
from text_similarity import word_ratio

# Minimum combined score to treat two clauses as the SAME clause across
# versions (below this: removed + added, not modified).
CLAUSE_MATCH_THRESHOLD = 0.30
# At/above this combined score, a matched pair counts as "unchanged" rather
# than "modified".
CLAUSE_IDENTICAL_THRESHOLD = 0.92

_SNIPPET_LEN = 160


def _snippet(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _SNIPPET_LEN else text[:_SNIPPET_LEN].rsplit(" ", 1)[0] + "..."


def _scores(old: Clause, new: Clause) -> tuple[float, float]:
    """Returns (match_score, body_score). match_score (body+heading blended)
    decides WHETHER two clauses are the same clause across versions.
    body_score alone then decides HOW identical they are (unchanged vs
    modified) — heading similarity must not be able to mask a real body
    edit as "unchanged" just because the section title didn't change,
    which is the common case for a real edit."""
    body_score = word_ratio(old.text_en, new.text_en)
    if old.heading and new.heading:
        heading_score = word_ratio(old.heading, new.heading)
        return 0.5 * body_score + 0.5 * heading_score, body_score
    # No heading on one or both sides: fall back to body-only for matching
    # too. Known limitation — two heavily-rewritten, unheaded clauses can
    # score below CLAUSE_MATCH_THRESHOLD and be misclassified as
    # removed+added instead of modified (documented, not silently hidden).
    return body_score, body_score


def _match_clauses(old_clauses: list[Clause], new_clauses: list[Clause]) -> list[tuple[Clause, Clause, float, float]]:
    candidates: list[tuple[float, float, Clause, Clause]] = []
    for old_clause in old_clauses:
        for new_clause in new_clauses:
            match_score, body_score = _scores(old_clause, new_clause)
            if match_score >= CLAUSE_MATCH_THRESHOLD:
                candidates.append((match_score, body_score, old_clause, new_clause))
    candidates.sort(key=lambda c: c[0], reverse=True)

    matched_old: set[str] = set()
    matched_new: set[str] = set()
    matches: list[tuple[Clause, Clause, float, float]] = []
    for match_score, body_score, old_clause, new_clause in candidates:
        if old_clause.clause_id in matched_old or new_clause.clause_id in matched_new:
            continue
        matched_old.add(old_clause.clause_id)
        matched_new.add(new_clause.clause_id)
        matches.append((old_clause, new_clause, match_score, body_score))
    return matches


def diff_documents(old_doc: Document, new_doc: Document) -> VersionDiff:
    matches = _match_clauses(old_doc.clauses, new_doc.clauses)
    matched_old_ids = {m[0].clause_id for m in matches}
    matched_new_ids = {m[1].clause_id for m in matches}

    entries: list[ClauseDiffEntry] = []

    for old_clause, new_clause, match_score, body_score in matches:
        change_type = "unchanged" if body_score >= CLAUSE_IDENTICAL_THRESHOLD else "modified"
        entries.append(ClauseDiffEntry(
            change_type=change_type,
            old_clause_id=old_clause.clause_id,
            new_clause_id=new_clause.clause_id,
            similarity=round(match_score, 4),
            old_heading=old_clause.heading,
            new_heading=new_clause.heading,
            old_text_snippet=_snippet(old_clause.text_en),
            new_text_snippet=_snippet(new_clause.text_en),
        ))

    for old_clause in old_doc.clauses:
        if old_clause.clause_id not in matched_old_ids:
            entries.append(ClauseDiffEntry(
                change_type="removed",
                old_clause_id=old_clause.clause_id,
                similarity=0.0,
                old_heading=old_clause.heading,
                old_text_snippet=_snippet(old_clause.text_en),
            ))

    for new_clause in new_doc.clauses:
        if new_clause.clause_id not in matched_new_ids:
            entries.append(ClauseDiffEntry(
                change_type="added",
                new_clause_id=new_clause.clause_id,
                similarity=0.0,
                new_heading=new_clause.heading,
                new_text_snippet=_snippet(new_clause.text_en),
            ))

    return VersionDiff(
        old_doc_id=old_doc.doc_id,
        new_doc_id=new_doc.doc_id,
        entries=entries,
        old_clause_count=len(old_doc.clauses),
        new_clause_count=len(new_doc.clauses),
    )


if __name__ == "__main__":
    def clause(cid: str, heading: str | None, text: str) -> Clause:
        return Clause(clause_id=cid, index=0, heading=heading, text_original="", text_en=text, page=1, char_span=(0, 0))

    old_doc = Document(doc_id="d_old", filename="a.pdf", source_language="en", page_count=1, clauses=[
        clause("o1", "Indemnification", "The Vendor shall indemnify, defend, and hold harmless the Client from any and all claims."),
        clause("o2", "Termination", "Either party may terminate this Agreement upon thirty days written notice."),
        clause("o3", "Confidentiality", "Each party shall keep confidential information secret and not disclose it to third parties."),
    ])

    new_doc = Document(doc_id="d_new", filename="b.pdf", source_language="en", page_count=1, clauses=[
        # o1 lightly edited -> should be "modified"
        clause("n1", "Indemnification", "The Vendor shall indemnify, defend, and hold harmless the Client from any and all claims and losses."),
        # o2 dropped entirely -> "removed"
        # o3 unchanged verbatim -> "unchanged"
        clause("n2", "Confidentiality", "Each party shall keep confidential information secret and not disclose it to third parties."),
        # brand new clause -> "added"
        clause("n3", "Governing Law", "This Agreement shall be governed by the laws of the State of Delaware."),
    ])

    diff = diff_documents(old_doc, new_doc)
    by_type: dict[str, list] = {"added": [], "removed": [], "modified": [], "unchanged": []}
    for entry in diff.entries:
        by_type[entry.change_type].append(entry)

    print(f"added={len(by_type['added'])} removed={len(by_type['removed'])} "
          f"modified={len(by_type['modified'])} unchanged={len(by_type['unchanged'])}")

    assert len(by_type["added"]) == 1 and by_type["added"][0].new_clause_id == "n3"
    assert len(by_type["removed"]) == 1 and by_type["removed"][0].old_clause_id == "o2"
    assert len(by_type["modified"]) == 1 and by_type["modified"][0].old_clause_id == "o1" and by_type["modified"][0].new_clause_id == "n1"
    assert len(by_type["unchanged"]) == 1 and by_type["unchanged"][0].old_clause_id == "o3" and by_type["unchanged"][0].new_clause_id == "n2"

    print("OK — added/removed/modified/unchanged all correctly classified.")

    # Negative case: a same-length but genuinely different clause must NOT
    # be matched to an unrelated one just because they're similar length.
    d1 = Document(doc_id="d1", filename="x.pdf", source_language="en", page_count=1, clauses=[
        clause("x1", "Payment Terms", "Payment is due within thirty days of the invoice date."),
    ])
    d2 = Document(doc_id="d2", filename="y.pdf", source_language="en", page_count=1, clauses=[
        clause("y1", "Governing Law", "This Agreement is governed by the laws of Delaware."),
    ])
    diff2 = diff_documents(d1, d2)
    types2 = {e.change_type for e in diff2.entries}
    assert types2 == {"removed", "added"}, f"expected no false match, got {[e.change_type for e in diff2.entries]}"
    print("OK — genuinely different clauses correctly rejected as a match (removed+added, not modified).")

    print("\nALL CHECKS PASSED.")
