from typing import Optional
from pathlib import Path
from .state import ClaimState
from .analyzer import CaseAnalysisAgent
from .retriever_agent import PolicyEvidenceAgent
from .coverage import CoverageExclusionAgent
from .adjudicator import DecisionAgent
from .validator import ValidationAgent
from ..models.claim import ClaimCase
from ..models.decision import AdjudicationResult
from ..rag.retriever import HybridRetriever
from ..config import settings


class ClaimAdjudicationPipeline:
    """
    Multi-Agent Orchestration Pipeline:
    Coordinates sequential specialized agents exchanging typed state:
    Case Analysis -> Policy Evidence -> Coverage & Exclusion -> Decision Adjudication -> Validation Guardrail.
    """

    def __init__(self, retriever: Optional[HybridRetriever] = None):
        if retriever is None:
            chunks_path = settings.policy_chunks_path
            self.retriever = HybridRetriever(chunks_path)
        else:
            self.retriever = retriever

        self.analyzer = CaseAnalysisAgent()
        self.evidence_agent = PolicyEvidenceAgent(self.retriever)
        self.coverage_agent = CoverageExclusionAgent()
        self.decision_agent = DecisionAgent()
        self.validation_agent = ValidationAgent()

    def run(self, case: ClaimCase) -> AdjudicationResult:
        # Initialize structured state
        state = ClaimState(case=case)

        # 1. Case Analysis
        state = self.analyzer.process(state)

        # 2. Policy Evidence Retrieval & Reranking
        state = self.evidence_agent.process(state)

        # 3. Coverage & Exclusion Evaluation
        state = self.coverage_agent.process(state)

        # 4. Adjudication & Financial Calculation
        state = self.decision_agent.process(state)

        # 5. Validation Guardrail
        state = self.validation_agent.process(state)

        # Construct final structured contract
        return AdjudicationResult(
            case_id=case.case_id,
            decision=state.decision,
            confidence=round(state.confidence, 2),
            key_findings=state.key_findings,
            applicable_limits=state.applicable_limits,
            missing_evidence=state.missing_evidence,
            citations=state.citations,
            validation=state.validation_report,
            trace=state.trace
        )
