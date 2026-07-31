# Legal Contract Review — project scaffold

A **walking skeleton**: a fake contract flows through all 7 stages using stubs
and prints a valid report. Everything already runs. Each person then replaces
her stub with real code, in stage order, without breaking the pipeline.

## Run it now

```bash
pip install -r requirements.txt        # or: pip install "pydantic>=2"
python pipeline.py                     # prints a ReviewReport, ends with OK
streamlit run app.py                   # optional: see it in a browser
```

## What's here

| File                       | Purpose                                          |
| -------------------------- | ------------------------------------------------ |
| `interfaces.py`            | Frozen Pydantic contracts. Do not edit alone.    |
| `stubs.py`                 | One stub per stage — replace the body of yours.  |
| `pipeline.py`              | Wires the 7 stages in order (the skeleton).      |
| `app.py`                   | Placeholder viewer (Drashti's real UI later).    |
| `data/sample_document.json`| Test contract until the real parser exists.      |
| `docs/INTERFACES.md`       | Full spec for the contracts.                     |

## Who replaces which stub, and when

Build sequentially. A stage is "done" when it returns real data in the same
shape and `python pipeline.py` still ends with OK.

1. `parse_document`   — **Drashti**: real PDF/DOCX parsing, translation, segmentation
2. `classify_clause`  — **Veda**: fine-tuned LegalBERT + explanation
3. `retrieve`         — **Veda**: GraphRAG over the legal knowledge graph
4. `analyze_clause`   — **Kashish**: risk / compliance / loophole / recommendation agents
5. `uncertainty_agent`— **Kashish**: trained RL policy (keep the heuristic as a baseline)
6. `build_report`     — **Kashish**: real scoring + summary
7. `app.py`           — **Drashti**: the real interface

## The one rule

Never change a field in `interfaces.py` without all three agreeing. The frozen
shapes are what let each stub-swap be a plug-in instead of a rewrite.
