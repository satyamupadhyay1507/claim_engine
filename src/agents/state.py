from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from ..models.claim import ClaimCase
from ..models.decision import DecisionStatus, Citation, TraceEntry, ValidationReport


@dataclass
class InvestigationPlan:
    dimensions: List[str] = field(default_factory=list)
    checklist: List[str] = field(default_factory=list)
    initial_missing_fields: List[str] = field(default_factory=list)


@dataclass
class EvidenceItem:
    chunk_id: str
    source: str
    page: int
    section: str
    clause: str
    text: str
    score: float = 0.0


@dataclass
class CoverageFinding:
    dimension: str
    status: str  # COVERED, EXCLUDED, WAITING_PERIOD_APPLIES, UNVERIFIED, SUB_LIMIT_APPLIES
    reasoning: str
    citation: Optional[Citation] = None


@dataclass
class FinancialAssessment:
    claimed_amount: float = 0.0
    admissible_amount: float = 0.0
    deductions: List[Dict[str, Any]] = field(default_factory=list)
    limits_applied: List[str] = field(default_factory=list)


@dataclass
class ClaimState:
    """
    Typed state container passed through the multi-agent workflow.
    Agents exchange structured data rather than free-form text.
    """
    case: ClaimCase
    investigation_plan: Optional[InvestigationPlan] = None
    retrieved_evidence: List[EvidenceItem] = field(default_factory=list)
    coverage_findings: List[CoverageFinding] = field(default_factory=list)
    financials: Optional[FinancialAssessment] = None
    decision: Optional[DecisionStatus] = None
    confidence: float = 0.0
    key_findings: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    validation_report: Optional[ValidationReport] = None
    trace: List[TraceEntry] = field(default_factory=list)
