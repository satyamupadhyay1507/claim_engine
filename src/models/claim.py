from typing import Dict, List, Optional, Any
from pydantic import BaseModel, ConfigDict, Field


class Patient(BaseModel):
    model_config = ConfigDict(extra="allow")
    age: int = Field(..., description="Age of the claimant in years")


class Hospital(BaseModel):
    model_config = ConfigDict(extra="allow")
    name: str = Field(..., description="Name of the hospital or clinic")
    network_provider: Optional[bool] = Field(default=None, description="Whether the hospital is in the insurer network")


class Treatment(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: str = Field(..., description="Treatment setting: inpatient, day_care, domiciliary, outpatient")
    admission_hours: int = Field(default=0, description="Total hours admitted in the medical facility")
    diagnosis: str = Field(..., description="Primary medical diagnosis")
    procedure: str = Field(..., description="Surgical or medical procedure performed")
    pre_existing: bool = Field(default=False, description="Whether the condition existed prior to policy inception")
    experimental: bool = Field(default=False, description="Whether treatment is experimental or unproven")
    hospital_room_unavailable: Optional[bool] = Field(default=None, description="Domiciliary condition: lack of hospital room")
    patient_cannot_be_moved: Optional[bool] = Field(default=None, description="Domiciliary condition: patient condition prevents transport")


class Expenses(BaseModel):
    model_config = ConfigDict(extra="allow")
    room: float = Field(default=0.0, description="Room rent and nursing expenses")
    doctor_fees: float = Field(default=0.0, description="Surgeon, anesthetist, medical practitioner fees")
    medicines_diagnostics: float = Field(default=0.0, description="Pharmacy, diagnostics, operation theatre charges")
    pre_hospitalization: float = Field(default=0.0, description="Expenses incurred prior to hospitalization")
    post_hospitalization: float = Field(default=0.0, description="Expenses incurred post-discharge")
    ambulance: float = Field(default=0.0, description="Emergency ambulance charges")


class EvidenceContext(BaseModel):
    model_config = ConfigDict(extra="allow")
    hospital_registered: Optional[bool] = None
    medical_necessity_confirmed: Optional[bool] = None
    hospital_minimum_criteria_documented: Optional[bool] = None


class ExpenseTiming(BaseModel):
    model_config = ConfigDict(extra="allow")
    pre_hospitalization_days_before_admission: Optional[int] = None
    post_hospitalization_days_after_discharge: Optional[int] = None
    same_condition_confirmed: Optional[bool] = None


class PriorPolicy(BaseModel):
    model_config = ConfigDict(extra="allow")
    insurer_type: Optional[str] = None
    continuous_years: Optional[int] = 0
    database_and_claim_history_received: Optional[bool] = None
    previous_sum_insured_inr: Optional[float] = 0.0


class ClaimCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str = Field(..., description="Unique case identifier (e.g. PUB-001)")
    policy_id: str = Field(..., description="Policy wording identifier")
    policy_start_date: str = Field(..., description="Policy inception date (YYYY-MM-DD)")
    claim_date: str = Field(..., description="Date of claim/admission (YYYY-MM-DD)")
    sum_insured_inr: float = Field(..., description="Total Sum Insured under policy in INR")
    continuous_coverage_months: int = Field(default=0, description="Continuous coverage months with current insurer")
    prior_insurer_continuous_years: int = Field(default=0, description="Continuous coverage years with prior insurer (portability)")
    patient: Patient
    hospital: Hospital
    treatment: Treatment
    expenses_inr: Expenses
    documents: List[str] = Field(default_factory=list, description="List of submitted claim documents")
    task: str = Field(..., description="Specific adjudication task / investigation focus")

    evidence_context: Optional[EvidenceContext] = None
    expense_timing: Optional[ExpenseTiming] = None
    prior_policy: Optional[PriorPolicy] = None
