# Interface Contracts — Legal Contract Review System

**Purpose:** These are the data contracts between the three layers. The team
builds the pipeline **sequentially** (stage by stage, owner leads), so each
contract is the agreed hand-off shape from one stage to the next. Freeze a
field only once its stage is built; don't change it afterward without all three
signing off.

## Freeze rules

- **Additive is free.** Adding a new *optional* field needs no discussion.
- **Breaking changes need all three.** Renaming, removing, or retyping an
  existing field requires agreement from Drashti, Veda, and Kashish.
- **`clause_id` and `doc_id` are sacred.** They are the join keys tying the
  layers together. Once assigned by Drashti's parser, they never change.

## Ownership at a glance

| Schema                     | Produced by | Consumed by      | Boundary it freezes     |
| -------------------------- | ----------- | ---------------- | ----------------------- |
| `Document` / `Clause`      | Drashti     | Veda, Kashish    | Ingestion -> everything |
| `Classification`           | Veda        | Kashish          | Model -> agents         |
| `RetrievalResult`          | Veda (API)  | Kashish          | Knowledge -> agents     |
| `UncertaintyDecision` (RL) | Kashish     | Kashish (report) | RL agent -> report      |
| `ReviewReport`             | Kashish     | Drashti (UI)     | Analysis -> interface   |

Data flows: **Drashti -> Veda -> Kashish -> Drashti (UI)**.

> Note on the guide's requirement: uncertainty is **not** a threshold in Veda's
> layer. It is a reinforcement-learning agent in Kashish's layer (section 4).
> Veda's classifier only emits a raw `confidence` signal that the RL agent uses
> as one of its inputs.

---

## 1. `Document` / `Clause` — Drashti's output (stage 1)

The parser turns an uploaded file into a document of ordered, segmented,
English-normalized clauses. Everyone downstream keys off `clause_id`.

### Fields

**Document**
| Field             | Type          | Notes                                   |
| ----------------- | ------------- | --------------------------------------- |
| `doc_id`          | string (uuid) | Assigned once, never changes.           |
| `filename`        | string        | Original upload name.                   |
| `source_language` | string        | ISO 639-1: `"en"`, `"hi"`, `"fr"`, ...  |
| `page_count`      | int           |                                         |
| `clauses`         | list[Clause]  | In document order.                      |

**Clause**
| Field           | Type           | Notes                                                    |
| --------------- | -------------- | -------------------------------------------------------- |
| `clause_id`     | string         | e.g. `"c_001"`. The universal join key.                  |
| `index`         | int            | Position in document, 0-based.                           |
| `heading`       | string \| null | Section heading if detected.                             |
| `text_original` | string         | Clause text in the source language.                      |
| `text_en`       | string         | English translation. Equals `text_original` if English. **Veda and Kashish always read `text_en`.** |
| `page`          | int            |                                                          |
| `char_span`     | [int, int]     | Start/end offsets in the original doc (UI highlighting). |

### Example

```json
{
  "doc_id": "d_9f3a...",
  "filename": "vendor_agreement.pdf",
  "source_language": "en",
  "page_count": 3,
  "clauses": [
    {
      "clause_id": "c_001",
      "index": 0,
      "heading": "Indemnification",
      "text_original": "The Vendor shall indemnify and hold harmless...",
      "text_en": "The Vendor shall indemnify and hold harmless...",
      "page": 1,
      "char_span": [1420, 1890]
    }
  ]
}
```

---

## 2. `Classification` — Veda's per-clause output (stage 2)

LegalBERT returns a type, a raw confidence, an explanation, and a *baseline*
low-confidence flag. The authoritative uncertainty decision is NOT here — it is
made by the RL agent in section 4. `confidence` is a feature the RL agent reads.

### Fields

| Field              | Type                | Notes                                                                 |
| ------------------ | ------------------- | --------------------------------------------------------------------- |
| `clause_id`        | string              | Must match a `Clause.clause_id`.                                      |
| `predicted_type`   | string              | e.g. `"indemnification"`, `"termination"`, `"confidentiality"`, `"governing_law"`, `"payment_terms"`, `"limitation_of_liability"`. Freeze the label set once. |
| `confidence`       | float               | 0.0-1.0. Feature consumed by the RL uncertainty agent.               |
| `top_k`            | list[{type, score}] | Alternative labels, for UI hover.                                     |
| `explanation`      | object              | `{ "method": "attention"\|"shap", "salient_tokens": [str], "rationale": str }`. |
| `low_conf_baseline`| bool                | `true` if `confidence` < agreed threshold. **Baseline only** — used to benchmark the RL agent against, not to drive routing. |

### Example

```json
{
  "clause_id": "c_001",
  "predicted_type": "indemnification",
  "confidence": 0.93,
  "top_k": [
    { "type": "indemnification", "score": 0.93 },
    { "type": "limitation_of_liability", "score": 0.04 }
  ],
  "explanation": {
    "method": "attention",
    "salient_tokens": ["indemnify", "hold harmless"],
    "rationale": "Strong indemnity language directed at the Vendor."
  },
  "low_conf_baseline": false
}
```

---

## 3. `RetrievalResult` — Veda's GraphRAG API (stage 3)

A retrieval function Kashish's agents call. A **function contract**, not just a
data shape. `relevance` is also a feature for the RL uncertainty agent.

### Signature

```
retrieve(query: str, clause_type: str | None = None, top_k: int = 5) -> RetrievalResult
```

### Fields

**RetrievalResult**
| Field     | Type               | Notes                    |
| --------- | ------------------ | ------------------------ |
| `query`   | string             | Echo of the input query. |
| `results` | list[RetrievalHit] | Ranked, best first.      |

**RetrievalHit**
| Field        | Type                 | Notes                                          |
| ------------ | -------------------- | ---------------------------------------------- |
| `source_id`  | string               | Stable ID; Kashish cites these in findings.    |
| `title`      | string               | Human-readable source name.                    |
| `snippet`    | string               | The relevant passage.                          |
| `relevance`  | float                | 0.0-1.0.                                        |
| `graph_path` | list[string] \| null | Optional GraphRAG nodes traversed, for XAI.    |

### Example

```json
{
  "query": "indemnification unlimited liability enforceability",
  "results": [
    {
      "source_id": "src_ucc_2_719",
      "title": "UCC 2-719 — Limitation of Remedy",
      "snippet": "Consequential damages may be limited or excluded unless...",
      "relevance": 0.88,
      "graph_path": ["indemnity", "liability_cap", "ucc_2_719"]
    }
  ]
}
```

---

## 4. `UncertaintyDecision` — Kashish's RL agent (stage 5)

**Per the guide's requirement, uncertainty is a reinforcement-learning agent.**
It decides, per clause, whether the automated finding can be trusted or must go
to a human. Keep Veda's `low_conf_baseline` as the comparison baseline so you
can show the RL policy outperforms a fixed threshold.

### RL formulation (contextual bandit is a defensible choice)

| Element  | Definition                                                                          |
| -------- | ----------------------------------------------------------------------------------- |
| State    | `[classification.confidence, max retrieval relevance, agent agreement, risk_level]` |
| Actions  | `accept` \| `flag` \| `reanalyze`                                                    |
| Reward   | large negative for auto-accepting a wrong high-risk finding (missed error); small negative for flagging a clause that was fine (wasted human effort); positive for a correct auto-accept. |

### Signature

```
uncertainty_agent(features: dict) -> UncertaintyDecision
```

### Fields

**UncertaintyDecision**
| Field              | Type   | Notes                                        |
| ------------------ | ------ | -------------------------------------------- |
| `clause_id`        | string | Join key.                                    |
| `action`           | string | `"accept"` \| `"flag"` \| `"reanalyze"`.     |
| `policy_confidence`| float  | 0.0-1.0. The policy's confidence in the action.|
| `reason`           | string | Short human-readable justification.          |

### Example

```json
{
  "clause_id": "c_001",
  "action": "flag",
  "policy_confidence": 0.81,
  "reason": "High-risk clause with only moderate retrieval support."
}
```

---

## 5. `ReviewReport` — Kashish's final output (stage 6)

The report Drashti's UI renders. `flagged_for_review` is populated from the RL
agent's `flag` actions, and each finding records the RL `review_decision`.

### Fields

**ReviewReport**
| Field                | Type                | Notes                                            |
| -------------------- | ------------------- | ------------------------------------------------ |
| `doc_id`             | string              | Matches `Document.doc_id`.                       |
| `generated_at`       | string (ISO 8601)   |                                                  |
| `overall_risk`       | string              | `"low"` \| `"medium"` \| `"high"`.               |
| `overall_score`      | float               | 0-100.                                           |
| `summary`            | string              | Executive summary.                               |
| `clause_findings`    | list[ClauseFinding] | One per analyzed clause.                         |
| `flagged_for_review` | list[string]        | `clause_id`s where the RL agent chose `flag`.    |
| `metadata`           | object              | `{ "agents_run": [str], "language": str, "processing_time_sec": float }` |

**ClauseFinding**
| Field             | Type        | Notes                                                     |
| ----------------- | ----------- | --------------------------------------------------------- |
| `clause_id`       | string      | Join key.                                                 |
| `clause_type`     | string      | From Veda's classification.                               |
| `risk_level`      | string      | `"low"` \| `"medium"` \| `"high"`.                        |
| `issues`          | list[Issue] | Findings from the analysis agents.                        |
| `recommendation`  | string      | Suggested revision or action.                             |
| `review_decision` | string      | `"accept"` \| `"flag"` \| `"reanalyze"` from the RL agent.|
| `review_reason`   | string      | Why the RL agent made that call.                          |
| `confidence`      | float       | 0.0-1.0. Agent confidence in the finding.                 |
| `explanation`     | string      | Why this finding was made.                                |

**Issue**
| Field         | Type         | Notes                                        |
| ------------- | ------------ | -------------------------------------------- |
| `category`    | string       | `"risk"` \| `"compliance"` \| `"loophole"`.  |
| `severity`    | string       | `"low"` \| `"medium"` \| `"high"`.           |
| `description` | string       |                                              |
| `evidence`    | list[string] | `source_id`s from `RetrievalResult`.         |

### Example

```json
{
  "doc_id": "d_9f3a...",
  "generated_at": "2026-09-14T10:22:00Z",
  "overall_risk": "high",
  "overall_score": 71.5,
  "summary": "Two high-severity issues: uncapped indemnity and missing governing-law clause.",
  "clause_findings": [
    {
      "clause_id": "c_001",
      "clause_type": "indemnification",
      "risk_level": "high",
      "issues": [
        {
          "category": "risk",
          "severity": "high",
          "description": "Indemnity is uncapped and one-sided against the Vendor.",
          "evidence": ["src_ucc_2_719"]
        }
      ],
      "recommendation": "Add a liability cap and make the indemnity mutual.",
      "review_decision": "flag",
      "review_reason": "High-risk clause with only moderate retrieval support.",
      "confidence": 0.9,
      "explanation": "Uncapped, one-directional indemnity language detected."
    }
  ],
  "flagged_for_review": ["c_001"],
  "metadata": {
    "agents_run": ["risk", "compliance", "loophole", "recommendation", "uncertainty_rl"],
    "language": "en",
    "processing_time_sec": 12.4
  }
}
```

---

## 6. Executable source of truth (`interfaces.py`)

Drop this in the repo so every stage validates against the same models.

```python
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

RiskLevel = Literal["low", "medium", "high"]
ReviewAction = Literal["accept", "flag", "reanalyze"]

# ---- Layer 1: Drashti ----
class Clause(BaseModel):
    clause_id: str
    index: int
    heading: Optional[str] = None
    text_original: str
    text_en: str
    page: int
    char_span: tuple[int, int]

class Document(BaseModel):
    doc_id: str
    filename: str
    source_language: str
    page_count: int
    clauses: list[Clause]

# ---- Layer 2: Veda ----
class TypeScore(BaseModel):
    type: str
    score: float

class Explanation(BaseModel):
    method: Literal["attention", "shap"]
    salient_tokens: list[str]
    rationale: str

class Classification(BaseModel):
    clause_id: str
    predicted_type: str
    confidence: float = Field(ge=0.0, le=1.0)   # feature for the RL agent
    top_k: list[TypeScore]
    explanation: Explanation
    low_conf_baseline: bool                       # threshold baseline, not the decision

class RetrievalHit(BaseModel):
    source_id: str
    title: str
    snippet: str
    relevance: float = Field(ge=0.0, le=1.0)
    graph_path: Optional[list[str]] = None

class RetrievalResult(BaseModel):
    query: str
    results: list[RetrievalHit]

# ---- Layer 3: Kashish ----
class UncertaintyDecision(BaseModel):
    clause_id: str
    action: ReviewAction
    policy_confidence: float = Field(ge=0.0, le=1.0)
    reason: str

class Issue(BaseModel):
    category: Literal["risk", "compliance", "loophole"]
    severity: RiskLevel
    description: str
    evidence: list[str] = []

class ClauseFinding(BaseModel):
    clause_id: str
    clause_type: str
    risk_level: RiskLevel
    issues: list[Issue]
    recommendation: str
    review_decision: ReviewAction                 # from the RL uncertainty agent
    review_reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str

class ReviewReport(BaseModel):
    doc_id: str
    generated_at: str
    overall_risk: RiskLevel
    overall_score: float = Field(ge=0.0, le=100.0)
    summary: str
    clause_findings: list[ClauseFinding]
    flagged_for_review: list[str]
    metadata: dict


# API contracts Veda implements and Kashish calls:
#   def classify_clause(clause: Clause) -> Classification: ...
#   def retrieve(query: str, clause_type: str | None = None, top_k: int = 5) -> RetrievalResult: ...
# RL agent Kashish implements:
#   def uncertainty_agent(features: dict) -> UncertaintyDecision: ...
```

## 7. Sequential build order

Build one stage at a time; the owner leads, the other two support. Each stage
is "done" when it produces a valid instance of its contract and the pipeline
runs end-to-end up to that point.

1. **Parse, translate, segment** (Drashti) -> emits `Document`.
2. **Classify + confidence + XAI** (Veda) -> emits `Classification` per clause.
3. **GraphRAG retrieval** (Veda) -> implements `retrieve()`.
4. **Supervisor + analysis agents** (Kashish) -> risk / compliance / loophole / recommendation.
5. **RL uncertainty agent** (Kashish) -> implements `uncertainty_agent()`. Built
   after stage 4, because it needs the pipeline running to generate the state
   signals (confidence, relevance, agreement) it learns from.
6. **Report generation** (Kashish) -> emits `ReviewReport`.
7. **UI integration** (Drashti) -> renders `ReviewReport`. Scaffold early, finish here.
8. **Evaluation + baseline comparison + paper** (all) -> including RL agent vs
   `low_conf_baseline`.

Because every stage writes to a frozen contract, moving to the next stage is a
plug-in, not a rewrite.
