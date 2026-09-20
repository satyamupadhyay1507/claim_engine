from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, ConfigDict, Field


class DecisionStatus(str, Enum):
    ADMISSIBLE = "ADMISSIBLE"
    ADMISSIBLE_WITH_LIMITS = "ADMISSIBLE_WITH_LIMITS"
    PARTIALLY_ADMISSIBLE = "PARTIALLY_ADMISSIBLE"
    NOT_ADMISSIBLE = "NOT_ADMISSIBLE"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class Citation(BaseModel):
    model_config = ConfigDict(extra="allow")

    claim: str = Field(..., description="Policy-supported statement or finding")
    source: str = Field(default="policy.pdf", description="Authoritative policy source document")
    page: int = Field(..., description="Page number in policy document (1-indexed)")
    section: str = Field(..., description="Section or heading title in policy")
    chunk_id: str = Field(..., description="Deterministic chunk identifier")


class ValidationReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str = Field(..., description="Validation status: PASS or FAIL")
    unsupported_claims: List[str] = Field(default_factory=list, description="Claims flagged without sufficient citation support")


class TraceEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    agent: str = Field(..., description="Name of the agent executing the action")
    action: str = Field(..., description="Action or tool invocation description")
    timestamp_ms: float = Field(..., description="Relative timestamp or duration in milliseconds")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata such as retrieval counts or validation flags")


class AdjudicationResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str = Field(..., description="Case identifier")
    decision: DecisionStatus = Field(..., description="Adjudication outcome")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    key_findings: List[str] = Field(default_factory=list, description="List of primary factual and policy findings")
    applicable_limits: List[str] = Field(default_factory=list, description="List of limits, sub-limits, or deductions applied")
    missing_evidence: List[str] = Field(default_factory=list, description="List of missing documents or evidence gaps")
    citations: List[Citation] = Field(default_factory=list, description="Policy citations backing every material statement")
    validation: ValidationReport = Field(..., description="Validation guardrail outcome")
    trace: List[TraceEntry] = Field(default_factory=list, description="Auditable agent execution trace")
