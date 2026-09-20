import time
from typing import List, Optional
from .state import ClaimState, CoverageFinding, EvidenceItem
from ..models.decision import Citation, TraceEntry


class CoverageExclusionAgent:
    """
    Coverage & Exclusion Agent:
    Assesses coverage scope, waiting periods (30-day, PED, specific ailments, portability),
    general exclusions, hospital definition criteria, and applicable limits.
    Produces structured findings with direct policy citations.
    """

    def __init__(self):
        self.name = "Coverage & Exclusion Agent"

    def _find_best_citation(self, evidence: List[EvidenceItem], keywords: List[str], claim_text: str) -> Citation:
        best_item = None
        best_score = -1.0

        for item in evidence:
            score = 0.0
            text_lower = item.text.lower()
            sec_lower = item.section.lower()
            cl_lower = item.clause.lower()

            for kw in keywords:
                kw_l = kw.lower()
                if kw_l in text_lower:
                    score += 2.0
                if kw_l in cl_lower:
                    score += 3.0
                if kw_l in sec_lower:
                    score += 1.5

            if score > best_score:
                best_score = score
                best_item = item

        if best_item is None and evidence:
            best_item = evidence[0]

        if best_item:
            return Citation(
                claim=claim_text,
                source=best_item.source,
                page=best_item.page,
                section=best_item.section,
                chunk_id=best_item.chunk_id
            )
        else:
            return Citation(
                claim=claim_text,
                source="policy.pdf",
                page=1,
                section="Scope of Cover",
                chunk_id="POL-DEFAULT"
            )

    def process(self, state: ClaimState) -> ClaimState:
        t0 = time.time()
        case = state.case
        evidence = state.retrieved_evidence
        findings: List[CoverageFinding] = []

        diag_lower = case.treatment.diagnosis.lower()
        proc_lower = case.treatment.procedure.lower()

        # 1. Hospital Definition Compliance
        if case.evidence_context and (
            case.evidence_context.hospital_registered is None or
            case.evidence_context.hospital_minimum_criteria_documented is False
        ):
            claim_str = "Treating facility does not have documented proof of meeting the policy definition of Hospital (registration, minimum beds, 24/7 nursing staff)."
            citation = self._find_best_citation(evidence, ["hospital definition", "registration", "beds", "nursing", "definition"], claim_str)
            findings.append(CoverageFinding(
                dimension="hospital_definition",
                status="UNVERIFIED",
                reasoning="The policy strictly requires hospitalization to occur in a facility meeting the statutory definition of a Hospital. The evidence context indicates unconfirmed registration or undocumented minimum bed criteria.",
                citation=citation
            ))

        # 2. General Exclusions: Cosmetic / Aesthetic
        if "cosmetic" in diag_lower or "cosmetic" in proc_lower or "aesthetic" in diag_lower:
            claim_str = "Cosmetic or aesthetic treatments of any description are strictly excluded under the policy."
            citation = self._find_best_citation(evidence, ["cosmetic", "aesthetic", "exclusion"], claim_str)
            findings.append(CoverageFinding(
                dimension="exclusion_cosmetic",
                status="EXCLUDED",
                reasoning="Treatment is classified as cosmetic surgery, which is explicitly excluded under policy exclusions.",
                citation=citation
            ))

        # 3. General Exclusions: Experimental / Unproven Treatment
        if case.treatment.experimental or "experimental" in diag_lower or "experimental" in proc_lower:
            claim_str = "Experimental, unproven, or investigational treatments or therapies are excluded from coverage."
            citation = self._find_best_citation(evidence, ["experimental", "unproven", "investigational", "exclusion"], claim_str)
            findings.append(CoverageFinding(
                dimension="exclusion_experimental",
                status="EXCLUDED",
                reasoning="The treatment/procedure is experimental or unproven, triggering the standard policy exclusion.",
                citation=citation
            ))

        # 4. 30-Day Initial Waiting Period
        # Applicable if continuous coverage is 0 months, and the condition is not an accidental injury
        if case.continuous_coverage_months == 0 and "accident" not in diag_lower and "trauma" not in diag_lower:
            claim_str = "A 30-day waiting period applies from policy commencement for any illness; expenses incurred during this period are not admissible."
            citation = self._find_best_citation(evidence, ["30 days", "waiting period", "commencement", "illness"], claim_str)
            findings.append(CoverageFinding(
                dimension="waiting_period_30_day",
                status="WAITING_PERIOD_APPLIES",
                reasoning="Claim occurred within the first 30 days of initial policy inception for a medical illness with zero continuous coverage.",
                citation=citation
            ))

        # 5. Pre-Existing Disease (PED) Waiting Period (48 Months)
        if case.treatment.pre_existing:
            # Check continuous coverage months + portability credit
            total_months = case.continuous_coverage_months + (case.prior_insurer_continuous_years * 12)
            if total_months < 48:
                claim_str = "Pre-existing diseases and directly related conditions are subject to a mandatory 48-month waiting period of continuous coverage."
                citation = self._find_best_citation(evidence, ["pre-existing", "48 months", "waiting period", "ped"], claim_str)
                findings.append(CoverageFinding(
                    dimension="waiting_period_ped",
                    status="WAITING_PERIOD_APPLIES",
                    reasoning=f"Claim is for a pre-existing condition with only {total_months} months of total continuous coverage, which is less than the required 48-month waiting period.",
                    citation=citation
                ))
            else:
                claim_str = "Pre-existing condition waiting period (48 months) is satisfied through continuous coverage and portability credit."
                citation = self._find_best_citation(evidence, ["pre-existing", "48 months", "portability"], claim_str)
                findings.append(CoverageFinding(
                    dimension="waiting_period_ped",
                    status="COVERED",
                    reasoning="48-month waiting period for pre-existing disease is fully satisfied.",
                    citation=citation
                ))

        # 6. Specific Disease Waiting Periods & Portability Credit (1-Year Waiting Period under Clause 3)
        specific_diseases = [
            "cataract", "joint replacement", "knee replacement", "hernia", "hydrocele",
            "hysterectomy", "myomectomy", "piles", "fistula", "sinusitis", "calculus", "stone",
            "benign prostatic", "gout", "rheumatism", "tonsils"
        ]
        is_specific_disease = any(w in diag_lower or w in proc_lower for w in specific_diseases)

        if is_specific_disease:
            total_years = (case.continuous_coverage_months / 12.0) + case.prior_insurer_continuous_years
            # Specific disease waiting period is 1 year (12 months) under Policy Clause 3
            if total_years >= 1.0:
                claim_str = "Specific waiting period for named ailment is satisfied when factoring continuous coverage and portability credit from prior insurer."
                citation = self._find_best_citation(evidence, ["first year", "waiting period", "portability", "continuity", "cataract", "joint replacement"], claim_str)
                findings.append(CoverageFinding(
                    dimension="waiting_period_specific",
                    status="COVERED",
                    reasoning=f"Specific waiting period satisfied with {total_years:.1f} years of combined continuous coverage.",
                    citation=citation
                ))
            else:
                claim_str = "Treatment is subject to a 1-year specific waiting period from policy commencement that has not yet elapsed."
                citation = self._find_best_citation(evidence, ["first year", "waiting period", "specific", "joint replacement", "cataract"], claim_str)
                findings.append(CoverageFinding(
                    dimension="waiting_period_specific",
                    status="WAITING_PERIOD_APPLIES",
                    reasoning=f"Treatment occurred at {case.continuous_coverage_months} months (< 12 months required) for a condition subject to the 1-year waiting period without portability credit.",
                    citation=citation
                ))

        # 7. Domiciliary Hospitalization
        if case.treatment.type.lower() == "domiciliary":
            # Check requirements: treatment > 3 days, hospital room unavailable or patient cannot be moved
            bed_unavail = case.treatment.hospital_room_unavailable
            cannot_move = case.treatment.patient_cannot_be_moved

            if bed_unavail or cannot_move:
                claim_str = "Domiciliary hospitalization is covered when treatment exceeds 3 days and hospital bed unavailability or patient immobility is certified, subject to a 20% sum insured sub-limit."
                citation = self._find_best_citation(evidence, ["domiciliary", "sub-limit", "hospitalization", "20%"], claim_str)
                findings.append(CoverageFinding(
                    dimension="domiciliary_hospitalization",
                    status="SUB_LIMIT_APPLIES",
                    reasoning="Domiciliary treatment criteria are satisfied; policy imposes a 20% Sum Insured sub-limit on domiciliary expenses.",
                    citation=citation
                ))
            else:
                claim_str = "Domiciliary hospitalization requires medical certification that the patient could not be moved or hospital accommodation was unavailable."
                citation = self._find_best_citation(evidence, ["domiciliary", "condition", "certificate"], claim_str)
                findings.append(CoverageFinding(
                    dimension="domiciliary_hospitalization",
                    status="UNVERIFIED",
                    reasoning="Domiciliary treatment conditions (unavailability of hospital accommodation or inability to transport patient) are not satisfied in the evidence.",
                    citation=citation
                ))

        # 8. Day Care Treatment
        if case.treatment.type.lower() == "day_care" or (case.treatment.admission_hours < 24 and case.treatment.admission_hours > 0):
            claim_str = "Day care treatments requiring less than 24 hours hospitalization due to technological advancements are covered under the policy."
            citation = self._find_best_citation(evidence, ["day care", "technological", "advancement", "24 hours"], claim_str)
            findings.append(CoverageFinding(
                dimension="day_care_treatment",
                status="COVERED",
                reasoning="Procedure qualified as an eligible day-care surgery under modern medical techniques.",
                citation=citation
            ))

        # 9. Inpatient Hospitalization & Room Rent
        if case.treatment.type.lower() == "inpatient":
            claim_str = "Inpatient hospitalization expenses are covered subject to room rent sub-limits of 1% of sum insured per day (2% for ICU) and proportionate deductions."
            citation = self._find_best_citation(evidence, ["room rent", "inpatient", "1%", "icu", "proportionate"], claim_str)
            findings.append(CoverageFinding(
                dimension="inpatient_scope",
                status="COVERED",
                reasoning="Standard inpatient hospitalization covered subject to policy limits.",
                citation=citation
            ))

        # 10. Pre- & Post-Hospitalization Time Windows
        if case.expense_timing:
            pre_days = case.expense_timing.pre_hospitalization_days_before_admission or 0
            post_days = case.expense_timing.post_hospitalization_days_after_discharge or 0

            if pre_days <= 30 and post_days <= 60 and case.expense_timing.same_condition_confirmed:
                claim_str = "Pre-hospitalization (up to 30 days prior) and post-hospitalization (up to 60 days post-discharge) expenses for the same condition are admissible."
                citation = self._find_best_citation(evidence, ["pre-hospitalization", "post-hospitalization", "30 days", "60 days"], claim_str)
                findings.append(CoverageFinding(
                    dimension="pre_post_hospitalization",
                    status="COVERED",
                    reasoning=f"Pre-hospitalization ({pre_days} days <= 30) and post-hospitalization ({post_days} days <= 60) satisfy policy time windows.",
                    citation=citation
                ))
            else:
                claim_str = "Expenses incurred outside the 30-day pre-hospitalization or 60-day post-hospitalization windows are non-payable."
                citation = self._find_best_citation(evidence, ["pre-hospitalization", "post-hospitalization", "30 days", "60 days"], claim_str)
                findings.append(CoverageFinding(
                    dimension="pre_post_hospitalization",
                    status="SUB_LIMIT_APPLIES",
                    reasoning="Some pre/post expenses exceed the permissible 30/60-day windows.",
                    citation=citation
                ))

        state.coverage_findings = findings

        elapsed_ms = (time.time() - t0) * 1000.0
        state.trace.append(TraceEntry(
            agent=self.name,
            action=f"Evaluated {len(findings)} coverage and exclusion dimensions against policy terms",
            timestamp_ms=round(elapsed_ms, 2),
            details={
                "findings_count": len(findings),
                "dimensions_evaluated": [f.dimension for f in findings]
            }
        ))
        return state
