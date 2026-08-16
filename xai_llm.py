"""Optional SLM layer for Stage 2's XAI: turns the classifier's
attention-derived salient tokens into a natural-language rationale sentence,
using the same local Ollama LLM Drashti's parser (parsing.py) already runs —
same backend, same model, no new infrastructure.

Important: this LLM only *phrases* what the real attention weights already
found. It's given the predicted type, confidence, and the actual salient
tokens, and explicitly told not to invent legal reasoning beyond them. It's
a writing layer on top of real XAI output, not a second, independent
explanation method — the underlying `Explanation.method` stays "attention".

Same fallback philosophy as the rest of Veda's layer: if Ollama isn't
running, is slow, or errors, classify_clause() falls back to its own
template rationale — nothing breaks, you just get plainer wording.
"""
from __future__ import annotations
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen2.5:3b"   # matches parsing.py — swap here if you use a different model

_RATIONALE_PROMPT = """You are explaining a legal-clause classifier's decision to a \
law student. Write ONE short, plain-English sentence (max 30 words) explaining why the \
clause below was classified as "%s" (confidence %.2f). Base your explanation ONLY on \
these model-attended words: %s. Do not introduce new legal claims or words not implied \
by that list. Return ONLY JSON: {"rationale": "..."}

CLAUSE:
\"\"\"
%s
\"\"\""""


def generate_rationale(
    clause_text: str, predicted_type: str, confidence: float,
    salient_tokens: list[str], timeout: float = 15.0,
) -> str | None:
    """Returns an LLM-phrased rationale, or None if Ollama is unreachable, slow,
    or returns something unusable — caller should fall back to its template
    rationale in that case."""
    if not salient_tokens:
        return None
    prompt = _RATIONALE_PROMPT % (
        predicted_type, confidence, ", ".join(salient_tokens), clause_text,
    )
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False, "format": "json"},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = json.loads(resp.json()["response"])
        rationale = data.get("rationale")
        if isinstance(rationale, str) and rationale.strip():
            return rationale.strip()
    except Exception:
        pass
    return None
