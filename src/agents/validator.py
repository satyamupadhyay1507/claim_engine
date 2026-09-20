import re
import time
from typing import List, Set
from .state import ClaimState
from ..models.decision import ValidationReport, TraceEntry


class ValidationAgent:
    """
    Validation & Guardrail Agent:
    Verifies that every material decision statement and citation claim is strictly
    grounded in the retrieved policy evidence chunks. Detects hallucinations and unsupported claims.
    """

    def __init__(self):
        self.name = "Validation Agent"

    def _tokenize(self, text: str) -> Set[str]:
        tokens = re.findall(r"\b[A-Za-z0-9_-]{3,}\b", text.lower())
        stopwords = {
            "the", "and", "for", "with", "this", "that", "from", "shall", "under",
            "such", "will", "any", "are", "not", "have", "been", "which", "all",
            "were", "their", "there", "about", "into", "over"
        }
        return {t for t in tokens if t not in stopwords}

    def process(self, state: ClaimState) -> ClaimState:
        t0 = time.time()
        evidence_corpus = " ".join([e.text + " " + e.clause + " " + e.section for e in state.retrieved_evidence]).lower()
        evidence_tokens = self._tokenize(evidence_corpus)

        unsupported: List[str] = []

        # 1. Audit citations claims
        for citation in state.citations:
            claim_tokens = self._tokenize(citation.claim)
            if not claim_tokens:
                continue

            # Check overlap against the cited chunk or overall retrieved evidence
            overlap = claim_tokens.intersection(evidence_tokens)
            overlap_ratio = len(overlap) / len(claim_tokens) if claim_tokens else 0.0

            # If less than 25% overlap with policy evidence, flag as unsupported
            if overlap_ratio < 0.25:
                unsupported.append(f"Citation claim lacks policy backing: '{citation.claim}' (overlap: {overlap_ratio:.1%})")

        # 2. Audit material statements in key findings
        for finding in state.key_findings:
            # Skip pure arithmetic summary statements
            if "INR" in finding and ("claimed" in finding or "payable" in finding):
                continue

            f_tokens = self._tokenize(finding)
            if not f_tokens:
                continue

            overlap = f_tokens.intersection(evidence_tokens)
            overlap_ratio = len(overlap) / len(f_tokens) if f_tokens else 0.0

            if overlap_ratio < 0.20:
                unsupported.append(f"Material statement lacks sufficient policy evidence: '{finding[:80]}...'")

        if unsupported:
            status = "FAIL"
        else:
            status = "PASS"

        state.validation_report = ValidationReport(
            status=status,
            unsupported_claims=unsupported
        )

        elapsed_ms = (time.time() - t0) * 1000.0
        state.trace.append(TraceEntry(
            agent=self.name,
            action=f"Validated policy grounding: status {status} with {len(unsupported)} unsupported claims flagged",
            timestamp_ms=round(elapsed_ms, 2),
            details={
                "status": status,
                "unsupported_count": len(unsupported),
                "unsupported_claims": unsupported
            }
        ))
        return state
