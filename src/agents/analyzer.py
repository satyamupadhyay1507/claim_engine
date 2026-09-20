import time
from typing import List
from .state import ClaimState, InvestigationPlan
from ..models.decision import TraceEntry


class CaseAnalysisAgent:
    """
    Case Analysis Agent:
    Extracts facts, identifies decision dimensions, detects missing fields,
    and creates the investigation plan for policy retrieval.
    """

    def __init__(self):
        self.name = "Case Analysis Agent"

    def process(self, state: ClaimState) -> ClaimState:
        t0 = time.time()
        case = state.case

        dimensions: List[str] = []
        checklist: List[str] = []
        missing: List[str] = []

        # 1. Evaluate coverage duration & initial waiting period
        if case.continuous_coverage_months == 0:
            dimensions.append("30-day initial waiting period illness")
            checklist.append("Check if treatment is due to illness within first 30 days of policy commencement")

        # 2. Evaluate pre-existing disease (PED)
        if case.treatment.pre_existing:
            dimensions.append("pre-existing disease 48-month waiting period")
            checklist.append("Assess 48-month waiting period for pre-existing conditions and any continuous coverage credit")

        # 3. Specific diseases waiting periods
        diag_lower = case.treatment.diagnosis.lower()
        proc_lower = case.treatment.procedure.lower()

        specific_ailments = ["cataract", "hernia", "hydrocele", "fistula", "piles", "sinusitis", "knee replacement", "joint replacement", "calculus", "stones"]
        if any(w in diag_lower or w in proc_lower for w in specific_ailments):
            dimensions.append("specific disease waiting period 1 year 2 year")
            checklist.append("Verify specific waiting period schedule for named ailment")

        # 4. Treatment modality
        ttype = case.treatment.type.lower()
        if ttype == "domiciliary":
            dimensions.append("domiciliary hospitalization conditions sub-limit")
            checklist.append("Confirm condition exceeded 3 days and hospital bed unavailability or patient immobility documented")
        elif ttype == "day_care" or case.treatment.admission_hours < 24:
            dimensions.append("day care treatment less than 24 hours technological advancement")
            checklist.append("Confirm day-care procedure eligibility under technological advancements")
        else:
            dimensions.append("inpatient treatment room rent ICU limits")
            checklist.append("Verify inpatient admission >= 24 hours, room rent 1% cap, ICU 2% cap")

        # 5. General Exclusions check
        if "cosmetic" in diag_lower or "cosmetic" in proc_lower or "aesthetic" in diag_lower:
            dimensions.append("cosmetic aesthetic surgery exclusion")
            checklist.append("Check policy exclusion for cosmetic or aesthetic treatments")

        if case.treatment.experimental or "experimental" in diag_lower or "experimental" in proc_lower or "unproven" in proc_lower:
            dimensions.append("experimental unproven treatment exclusion")
            checklist.append("Check policy exclusion for experimental or unproven therapies")

        # 6. Pre and Post hospitalization expense windows
        if case.expenses_inr.pre_hospitalization > 0 or case.expenses_inr.post_hospitalization > 0:
            dimensions.append("pre-hospitalization 30 days post-hospitalization 60 days")
            checklist.append("Verify pre-hospitalization within 30 days and post-hospitalization within 60 days")

        # 7. Portability & Prior Policy
        if case.prior_insurer_continuous_years > 0 or case.prior_policy:
            dimensions.append("portability waiting period continuity credit previous insurer")
            checklist.append("Apply portability credit from continuous prior Indian health insurance")

        # 8. Hospital Definition & Evidence Completeness
        dimensions.append("hospital definition registration minimum beds nursing staff")
        checklist.append("Verify whether the treating facility meets the policy definition of Hospital")

        # Check for explicit missing context
        if case.evidence_context:
            if case.evidence_context.hospital_registered is None and not case.hospital.network_provider:
                missing.append("Treating facility registration certificate or proof of hospital status")
            if case.evidence_context.hospital_minimum_criteria_documented is False:
                missing.append("Evidence that facility maintains minimum required beds and qualified nursing staff")
            if case.evidence_context.medical_necessity_confirmed is None:
                missing.append("Treating doctor confirmation of medical necessity")

        # Check required documentation
        doc_list = [d.lower() for d in case.documents]
        if "discharge_summary" not in doc_list and ttype != "domiciliary":
            missing.append("Hospital discharge summary")
        if "itemized_bill" not in doc_list:
            missing.append("Itemized hospital bill and breakdown")

        state.investigation_plan = InvestigationPlan(
            dimensions=dimensions,
            checklist=checklist,
            initial_missing_fields=missing
        )
        state.missing_evidence.extend(missing)

        elapsed_ms = (time.time() - t0) * 1000.0
        state.trace.append(TraceEntry(
            agent=self.name,
            action="Analyzed claim facts, extracted 8 decision dimensions and generated investigation plan",
            timestamp_ms=round(elapsed_ms, 2),
            details={
                "dimensions_count": len(dimensions),
                "checklist_count": len(checklist),
                "initial_missing_evidence": missing
            }
        ))
        return state
