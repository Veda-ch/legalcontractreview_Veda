"""Shared content-similarity primitive used by storage.py (document matching)
and version_diff.py (clause matching).

Word-level SequenceMatcher, not character-level: character-level ratio is an
unreliable discriminator once a clause/document is heavily reworded (it can
rank a genuinely different pair as more similar than a heavily-edited true
match, because character n-grams overlap by chance more than whole words do).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import TypeVar

_WHITESPACE_RE = re.compile(r"\s+")

K = TypeVar("K")


def normalize(text: str | None) -> str:
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def word_ratio(a: str | None, b: str | None) -> float:
    """Similarity over word-token sequences (order-sensitive, like a diff)."""
    tokens_a = normalize(a).split()
    tokens_b = normalize(b).split()
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    return SequenceMatcher(None, tokens_a, tokens_b).ratio()


def char_ratio(a: str | None, b: str | None) -> float:
    """Similarity over raw characters. Only used for short strings (e.g.
    filenames) where word-level tokenization is too coarse to be useful."""
    na, nb = normalize(a), normalize(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def best_match(
    scores: dict[K, float],
    threshold: float,
    margin: float,
) -> tuple[K, float] | None:
    """Return the best-scoring key if it clears `threshold` AND beats the
    runner-up by at least `margin`. Otherwise None (ambiguous or no
    confident match) — the safe default is "no match", never a guess."""
    if not scores:
        return None
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_key, best_score = ranked[0]
    if best_score < threshold:
        return None
    if len(ranked) > 1:
        _, second_score = ranked[1]
        if best_score - second_score < margin:
            return None
    return best_key, best_score


if __name__ == "__main__":
    # Standalone sanity checks (Tier 1 — pure functions, no I/O).
    assert word_ratio("Hello world", "hello   WORLD") == 1.0
    assert word_ratio("", "") == 1.0
    assert word_ratio("something", "") == 0.0
    assert 0.0 < word_ratio("The cat sat on the mat", "The dog sat on the mat") < 1.0

    # Heavily-reworded same clause should still beat an unrelated clause,
    # once word-level ratio is used instead of character-level.
    original = "The Vendor shall indemnify, defend, and hold harmless the Client from any and all claims, losses, damages, and expenses."
    reworded = "Vendor agrees to fully protect, cover, and reimburse Client against every claim, loss, damage, or cost that may arise."
    unrelated = "Payment is due within thirty days of the invoice date, subject to a late fee of one percent per month."

    sim_reworded = word_ratio(original, reworded)
    sim_unrelated = word_ratio(original, unrelated)
    print(f"word_ratio(original, reworded)  = {sim_reworded:.3f}")
    print(f"word_ratio(original, unrelated) = {sim_unrelated:.3f}")
    assert sim_reworded > sim_unrelated, "reworded same-topic clause should score higher than an unrelated one"

    scores = {"doc_a": 0.62, "doc_b": 0.30, "doc_c": 0.10}
    assert best_match(scores, threshold=0.35, margin=0.08) == ("doc_a", 0.62)
    assert best_match({"doc_a": 0.40, "doc_b": 0.38}, threshold=0.35, margin=0.08) is None  # too close
    assert best_match({"doc_a": 0.20}, threshold=0.35, margin=0.08) is None  # below threshold
    assert best_match({}, threshold=0.35, margin=0.08) is None

    print("ALL CHECKS PASSED.")
