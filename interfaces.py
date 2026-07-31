"""Frozen data contracts for the legal contract review pipeline.

Every stage validates its input/output against these models. Do not change a
field without all three owners agreeing (see docs/INTERFACES.md).
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

RiskLevel = Literal["low", "medium", "high"]
ReviewAction = Literal["accept", "flag", "reanalyze"]


# ---- Layer 1: Drashti (ingestion) ----
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


# ---- Layer 2: Veda (classification + knowledge) ----
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
    confidence: float = Field(ge=0.0, le=1.0)   # feature for the RL uncertainty agent
    top_k: list[TypeScore]
    explanation: Explanation
    low_conf_baseline: bool                       # threshold baseline, NOT the decision


class RetrievalHit(BaseModel):
    source_id: str
    title: str
    snippet: str
    relevance: float = Field(ge=0.0, le=1.0)
    graph_path: Optional[list[str]] = None


class RetrievalResult(BaseModel):
    query: str
    results: list[RetrievalHit]


# ---- Layer 3: Kashish (agents, RL uncertainty, reports) ----
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
