import time
from typing import List, Dict, Any
from .state import ClaimState, FinancialAssessment
from ..models.decision import DecisionStatus, TraceEntry, Citation


class DecisionAgent:
    """
    Decision & Adjudication Agent:
    Combines specialist findings into the final adjudication decision,
    calculates policy limits, deductions, room rent caps, and compiles citations.
    """

    def __init__(self):
        self.name = "Decision Agent"

    def process(self, state: ClaimState) -> ClaimState:
        t0 = time.time()
        case = state.case
        findings = state.coverage_findings
        evidence = state.retrieved_evidence

        key_findings: List[str] = []
        applicable_limits: List[str] = []
        citations: List[Citation] = []
        missing_evidence: List[str] = list(state.missing_evidence)

        # Collect citations from coverage findings
        for f in findings:
            if f.citation and f.citation not in citations:
                citations.append(f.citation)

        # Check for unverified evidence requiring abstention (NEEDS_REVIEW)
        has_unverified = any(f.status == "UNVERIFIED" for f in findings)
        has_excluded = any(f.status == "EXCLUDED" for f in findings)
        has_waiting_period = any(f.status == "WAITING_PERIOD_APPLIES" for f in findings)
        has_sub_limits = any(f.status == "SUB_LIMIT_APPLIES" for f in findings)

        # Financial calculations
        total_claimed = (
            case.expenses_inr.room +
            case.expenses_inr.doctor_fees +
            case.expenses_inr.medicines_diagnostics +
            case.expenses_inr.pre_hospitalization +
            case.expenses_inr.post_hospitalization +
            case.expenses_inr.ambulance
        )

        deductions: List[Dict[str, Any]] = []
        limits_applied: List[str] = []
        admissible_amount = total_claimed

        # 1. Evaluate Decision Status
        if has_unverified or len(missing_evidence) > 0 and (
            case.evidence_context and (
                case.evidence_context.hospital_registered is None or
                case.evidence_context.hospital_minimum_criteria_documented is False or
                case.evidence_context.medical_necessity_confirmed is None
            )
        ):
            decision = DecisionStatus.NEEDS_REVIEW
            confidence = 0.85
            key_findings.append("The claim cannot be safely adjudicated because required evidence or compliance with policy definitions is missing.")
            for f in findings:
                if f.status == "UNVERIFIED":
                    key_findings.append(f.reasoning)
            admissible_amount = 0.0

        elif has_excluded:
            decision = DecisionStatus.NOT_ADMISSIBLE
            confidence = 0.95
            for f in findings:
                if f.status == "EXCLUDED":
                    key_findings.append(f.reasoning)
            admissible_amount = 0.0

        elif has_waiting_period:
            decision = DecisionStatus.NOT_ADMISSIBLE
            confidence = 0.95
            for f in findings:
                if f.status == "WAITING_PERIOD_APPLIES":
                    key_findings.append(f.reasoning)
            admissible_amount = 0.0

        else:
            # Calculate Admissibility and Limits
            # Room rent cap: 1% of Sum Insured per day
            admission_days = max(1, case.treatment.admission_hours // 24)
            daily_room_cap = 0.01 * case.sum_insured_inr
            max_room_allowed = daily_room_cap * admission_days

            actual_room = case.expenses_inr.room
            if actual_room > max_room_allowed and admission_days > 0:
                room_deduction = actual_room - max_room_allowed
                deductions.append({
                    "category": "Room Rent",
                    "claimed": actual_room,
                    "allowed": max_room_allowed,
                    "deduction": room_deduction,
                    "reason": f"Room rent exceeds policy sub-limit of 1% of Sum Insured (INR {daily_room_cap:,.0f}/day for {admission_days} days)."
                })
                applicable_limits.append(f"Room rent capped at INR {daily_room_cap:,.0f}/day (1% of Sum Insured). Excess of INR {room_deduction:,.0f} deducted.")
                admissible_amount -= room_deduction

                # Proportionate deduction on associate medical fees
                ratio = max_room_allowed / actual_room
                assoc_claimed = case.expenses_inr.doctor_fees + case.expenses_inr.medicines_diagnostics
                assoc_allowed = assoc_claimed * ratio
                assoc_deduction = assoc_claimed - assoc_allowed
                if assoc_deduction > 0:
                    deductions.append({
                        "category": "Proportionate Medical & Doctor Fees",
                        "claimed": assoc_claimed,
                        "allowed": assoc_allowed,
                        "deduction": assoc_deduction,
                        "reason": f"Proportionate deduction applied due to room rent category overshoot (ratio: {ratio:.2%})."
                    })
                    applicable_limits.append(f"Proportionate deduction of INR {assoc_deduction:,.0f} applied to doctor fees and diagnostics due to room rent over-limit.")
                    admissible_amount -= assoc_deduction

            # Ambulance cap: Rs. 1,000 per hospitalization
            if case.expenses_inr.ambulance > 1000.0:
                amb_deduction = case.expenses_inr.ambulance - 1000.0
                deductions.append({
                    "category": "Ambulance",
                    "claimed": case.expenses_inr.ambulance,
                    "allowed": 1000.0,
                    "deduction": amb_deduction,
                    "reason": "Ambulance charges capped at policy limit of INR 1,000 per hospitalization."
                })
                applicable_limits.append("Ambulance charges capped at INR 1,000 per policy schedule.")
                admissible_amount -= amb_deduction

            # Domiciliary sub-limit: 20% of Sum Insured
            if case.treatment.type.lower() == "domiciliary":
                dom_cap = 0.20 * case.sum_insured_inr
                if total_claimed > dom_cap:
                    dom_deduction = total_claimed - dom_cap
                    deductions.append({
                        "category": "Domiciliary Sub-Limit",
                        "claimed": total_claimed,
                        "allowed": dom_cap,
                        "deduction": dom_deduction,
                        "reason": f"Domiciliary treatment capped at 20% of Sum Insured (INR {dom_cap:,.0f})."
                    })
                    applicable_limits.append(f"Domiciliary treatment sub-limit of 20% Sum Insured applied (capped at INR {dom_cap:,.0f}).")
                    admissible_amount = dom_cap

            # Category-specific overall cap (Sum Insured limit)
            if admissible_amount > case.sum_insured_inr:
                si_deduction = admissible_amount - case.sum_insured_inr
                applicable_limits.append(f"Total payable capped at policy Sum Insured of INR {case.sum_insured_inr:,.0f}.")
                admissible_amount = case.sum_insured_inr

            # Determine whether ADMISSIBLE or ADMISSIBLE_WITH_LIMITS
            if len(applicable_limits) > 0 or len(deductions) > 0:
                decision = DecisionStatus.ADMISSIBLE_WITH_LIMITS
                confidence = 0.90
                key_findings.append(f"Claim is admissible under policy terms, but subject to applicable sub-limits and proportionate deductions. Payable amount: INR {admissible_amount:,.0f} out of INR {total_claimed:,.0f} claimed.")
            else:
                decision = DecisionStatus.ADMISSIBLE
                confidence = 0.95
                key_findings.append(f"Claim is fully admissible under policy terms with no material limit deductions. Full claimed amount of INR {total_claimed:,.0f} is payable.")

            for f in findings:
                if f.status == "COVERED":
                    key_findings.append(f.reasoning)

        state.decision = decision
        state.confidence = confidence
        state.key_findings = key_findings
        state.applicable_limits = applicable_limits
        state.citations = citations
        state.financials = FinancialAssessment(
            claimed_amount=total_claimed,
            admissible_amount=max(0.0, admissible_amount),
            deductions=deductions,
            limits_applied=applicable_limits
        )

        elapsed_ms = (time.time() - t0) * 1000.0
        state.trace.append(TraceEntry(
            agent=self.name,
            action=f"Adjudicated claim status as {decision.value} with {len(applicable_limits)} limits applied",
            timestamp_ms=round(elapsed_ms, 2),
            details={
                "decision": decision.value,
                "confidence": confidence,
                "claimed_amount": total_claimed,
                "admissible_amount": max(0.0, admissible_amount),
                "deductions_count": len(deductions)
            }
        ))
        return state
