
"""Data contracts shared by every stage of the legal contract review pipeline."""
from __future__ import annotations
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field

RiskLevel = Literal["low", "medium", "high"]
ReviewAction = Literal["accept", "flag", "reanalyze"]
RiskAssessment = Literal[
    "attention_required",
    "no_configured_risk_found",
    "insufficient_evidence",
]


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
    confidence: float = Field(ge=0.0, le=1.0)
    top_k: list[TypeScore]
    explanation: Explanation
    low_conf_baseline: bool


class RetrievalHit(BaseModel):
    source_id: str
    title: str
    snippet: str
    relevance: float = Field(ge=0.0, le=1.0)
    graph_path: Optional[list[str]] = None


class RetrievalResult(BaseModel):
    query: str
    results: list[RetrievalHit]


class UncertaintyDecision(BaseModel):
    clause_id: str
    action: ReviewAction
    policy_confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class Issue(BaseModel):
    category: Literal["risk", "compliance", "loophole"]
    severity: RiskLevel
    description: str
    evidence: list[str] = Field(default_factory=list)


class ClauseFinding(BaseModel):
    clause_id: str
    clause_type: str
    risk_assessment: RiskAssessment
    risk_factors: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    recommendation: str
    review_decision: ReviewAction
    review_reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    best_relevance: float = Field(ge=0.0, le=1.0)
    text_comparison: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    explanation: str
    adversarial_review: Optional[dict[str, Any]] = None


class ReviewReport(BaseModel):
    doc_id: str
    generated_at: str
    overall_assessment: RiskAssessment
    summary: str
    clause_findings: list[ClauseFinding]
    flagged_for_review: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---- Version Workflow (Phase 2): database storage + version comparison ----

VersionChangeType = Literal["added", "removed", "modified", "unchanged"]
RiskChangeType = Literal["increased", "reduced", "unchanged"]


class ClauseDiffEntry(BaseModel):
    change_type: VersionChangeType
    old_clause_id: Optional[str] = None
    new_clause_id: Optional[str] = None
    similarity: float = Field(ge=0.0, le=1.0)
    old_heading: Optional[str] = None
    new_heading: Optional[str] = None
    old_text_snippet: Optional[str] = None
    new_text_snippet: Optional[str] = None


class VersionDiff(BaseModel):
    old_doc_id: str
    new_doc_id: str
    entries: list[ClauseDiffEntry]
    old_clause_count: int
    new_clause_count: int


class ClauseRiskComparison(BaseModel):
    change_type: VersionChangeType
    risk_change: RiskChangeType
    old_clause_id: Optional[str] = None
    new_clause_id: Optional[str] = None
    old_risk_assessment: Optional[RiskAssessment] = None
    new_risk_assessment: Optional[RiskAssessment] = None
    old_review_decision: Optional[ReviewAction] = None
    new_review_decision: Optional[ReviewAction] = None
    explanation: str


class RiskComparisonReport(BaseModel):
    old_doc_id: str
    new_doc_id: str
    clause_comparisons: list[ClauseRiskComparison]
    increased_count: int
    reduced_count: int
    unchanged_count: int


class VersionRecommendation(BaseModel):
    old_doc_id: str
    new_doc_id: str
    safer_version: Literal["old", "new", "equivalent"]
    reasoning: str
    reasoning_source: Literal["llm", "fallback"]


class VersionAnalysisResult(BaseModel):
    matched_doc_id: Optional[str] = None
    match_score: Optional[float] = None
    diff: Optional[VersionDiff] = None
    comparison: Optional[RiskComparisonReport] = None
    recommendation: Optional[VersionRecommendation] = None


# """Frozen data contracts for the legal contract review pipeline.

# Every stage validates its input/output against these models. Do not change a
# field without all three owners agreeing (see docs/INTERFACES.md).
# """
# from __future__ import annotations
# from typing import Literal, Optional
# from pydantic import BaseModel, Field

# RiskLevel = Literal["low", "medium", "high"]
# ReviewAction = Literal["accept", "flag", "reanalyze"]


# # ---- Layer 1: Drashti (ingestion) ----
# class Clause(BaseModel):
#     clause_id: str
#     index: int
#     heading: Optional[str] = None
#     text_original: str
#     text_en: str
#     page: int
#     char_span: tuple[int, int]


# class Document(BaseModel):
#     doc_id: str
#     filename: str
#     source_language: str
#     page_count: int
#     clauses: list[Clause]


# # ---- Layer 2: Veda (classification + knowledge) ----
# class TypeScore(BaseModel):
#     type: str
#     score: float


# class Explanation(BaseModel):
#     method: Literal["attention", "shap"]
#     salient_tokens: list[str]
#     rationale: str


# class Classification(BaseModel):
#     clause_id: str
#     predicted_type: str
#     confidence: float = Field(ge=0.0, le=1.0)   # feature for the RL uncertainty agent
#     top_k: list[TypeScore]
#     explanation: Explanation
#     low_conf_baseline: bool                       # threshold baseline, NOT the decision


# class RetrievalHit(BaseModel):
#     source_id: str
#     title: str
#     snippet: str
#     relevance: float = Field(ge=0.0, le=1.0)
#     graph_path: Optional[list[str]] = None


# class RetrievalResult(BaseModel):
#     query: str
#     results: list[RetrievalHit]


# # ---- Layer 3: Kashish (agents, RL uncertainty, reports) ----
# class UncertaintyDecision(BaseModel):
#     clause_id: str
#     action: ReviewAction
#     policy_confidence: float = Field(ge=0.0, le=1.0)
#     reason: str


# class Issue(BaseModel):
#     category: Literal["risk", "compliance", "loophole"]
#     severity: RiskLevel
#     description: str
#     evidence: list[str] = []


# class ClauseFinding(BaseModel):
#     clause_id: str
#     clause_type: str
#     risk_level: RiskLevel
#     issues: list[Issue]
#     recommendation: str
#     review_decision: ReviewAction                 # from the RL uncertainty agent
#     review_reason: str
#     confidence: float = Field(ge=0.0, le=1.0)
#     explanation: str


# class ReviewReport(BaseModel):
#     doc_id: str
#     generated_at: str
#     overall_risk: RiskLevel
#     overall_score: float = Field(ge=0.0, le=100.0)
#     summary: str
#     clause_findings: list[ClauseFinding]
#     flagged_for_review: list[str]
#     metadata: dict
