import pytest
from src.models.claim import ClaimCase
from src.models.decision import DecisionStatus
from src.agents.workflow import ClaimAdjudicationPipeline
from src.rag.retriever import HybridRetriever
from src.config import settings


@pytest.fixture(scope="module")
def pipeline():
    retriever = HybridRetriever(settings.policy_chunks_path)
    return ClaimAdjudicationPipeline(retriever=retriever)


def test_waiting_period_30_days(pipeline):
    case = ClaimCase(
        case_id="TEST-001",
        policy_id="USGIC-CSC-2017-2018",
        policy_start_date="2026-01-01",
        claim_date="2026-01-15",
        sum_insured_inr=500000,
        continuous_coverage_months=0,
        patient={"age": 30},
        hospital={"name": "City Hospital", "network_provider": True},
        treatment={
            "type": "inpatient",
            "admission_hours": 48,
            "diagnosis": "Viral gastroenteritis",
            "procedure": "Medical management",
            "pre_existing": False,
            "experimental": False
        },
        expenses_inr={"room": 5000, "doctor_fees": 5000, "medicines_diagnostics": 10000},
        documents=["claim_form", "discharge_summary", "itemized_bill"],
        task="Test 30 day waiting period"
    )
    result = pipeline.run(case)
    assert result.decision == DecisionStatus.NOT_ADMISSIBLE
    assert result.validation.status == "PASS"


def test_room_rent_capping_and_proportionate_deduction(pipeline):
    case = ClaimCase(
        case_id="TEST-002",
        policy_id="USGIC-CSC-2017-2018",
        policy_start_date="2024-01-01",
        claim_date="2026-02-10",
        sum_insured_inr=500000,
        continuous_coverage_months=25,
        patient={"age": 40},
        hospital={"name": "Metro Hospital", "network_provider": True},
        treatment={
            "type": "inpatient",
            "admission_hours": 96,  # 4 days
            "diagnosis": "Acute appendicitis",
            "procedure": "Appendectomy",
            "pre_existing": False,
            "experimental": False
        },
        expenses_inr={
            "room": 40000,  # 10k/day vs 5k/day cap
            "doctor_fees": 30000,
            "medicines_diagnostics": 60000
        },
        documents=["claim_form", "discharge_summary", "itemized_bill"],
        task="Test room rent capping"
    )
    result = pipeline.run(case)
    assert result.decision == DecisionStatus.ADMISSIBLE_WITH_LIMITS
    assert len(result.applicable_limits) > 0
    assert any("Room rent capped" in lim for lim in result.applicable_limits)
    assert any("Proportionate deduction" in lim for lim in result.applicable_limits)


def test_abstention_on_unverified_hospital(pipeline):
    case = ClaimCase(
        case_id="TEST-003",
        policy_id="USGIC-CSC-2017-2018",
        policy_start_date="2024-01-01",
        claim_date="2026-03-01",
        sum_insured_inr=500000,
        continuous_coverage_months=26,
        patient={"age": 35},
        hospital={"name": "Unknown Clinic", "network_provider": False},
        treatment={
            "type": "inpatient",
            "admission_hours": 48,
            "diagnosis": "Acute infection",
            "procedure": "IV antibiotics",
            "pre_existing": False,
            "experimental": False
        },
        expenses_inr={"room": 10000, "doctor_fees": 10000, "medicines_diagnostics": 20000},
        documents=["claim_form"],
        evidence_context={
            "hospital_registered": None,
            "hospital_minimum_criteria_documented": False
        },
        task="Test abstention when hospital definition is unverified"
    )
    result = pipeline.run(case)
    assert result.decision == DecisionStatus.NEEDS_REVIEW
    assert len(result.missing_evidence) > 0
