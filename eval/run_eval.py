import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Setup root path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.models.claim import ClaimCase
from src.agents.workflow import ClaimAdjudicationPipeline
from src.rag.retriever import HybridRetriever
from src.config import settings

# Ground truth benchmarks established from policy terms & assignment specifications
EXPECTED_OUTCOMES: Dict[str, Dict[str, Any]] = {
    # 12 Public Cases
    "PUB-001": {
        "expected_decision": "ADMISSIBLE_WITH_LIMITS",
        "expected_limits": ["Room rent", "Proportionate deduction", "Ambulance"],
        "description": "Acute appendicitis; room rent exceeds 1% SI/day, triggers proportionate deduction"
    },
    "PUB-002": {
        "expected_decision": "NOT_ADMISSIBLE",
        "expected_limits": [],
        "description": "Viral fever on day 20 with 0 continuous months; excluded under 30-day initial waiting period"
    },
    "PUB-003": {
        "expected_decision": "NOT_ADMISSIBLE",
        "expected_limits": [],
        "description": "Pre-existing thyroid condition at 27 months continuous coverage; requires 48 months"
    },
    "PUB-004": {
        "expected_decision": "ADMISSIBLE_WITH_LIMITS",
        "expected_limits": ["Domiciliary treatment sub-limit"],
        "description": "Domiciliary treatment meeting room unavailability condition; capped at 20% SI"
    },
    "PUB-005": {
        "expected_decision": "ADMISSIBLE",
        "expected_limits": [],
        "description": "Cataract day care procedure with 29 months coverage; eligible under technological advancement"
    },
    "PUB-006": {
        "expected_decision": "NEEDS_REVIEW",
        "expected_limits": [],
        "description": "Acute infection; unverified hospital registration and unconfirmed medical necessity"
    },
    "PUB-007": {
        "expected_decision": "ADMISSIBLE_WITH_LIMITS",
        "expected_limits": ["Room rent", "Proportionate deduction", "Ambulance"],
        "description": "Cancer treatment; room rent overshoot triggers proportionate deduction and ambulance cap"
    },
    "PUB-008": {
        "expected_decision": "NOT_ADMISSIBLE",
        "expected_limits": [],
        "description": "Cosmetic surgery; strictly excluded under policy general exclusions"
    },
    "PUB-009": {
        "expected_decision": "ADMISSIBLE",
        "expected_limits": [],
        "description": "Pre/post hospitalization expenses within 30-day pre and 60-day post windows"
    },
    "PUB-010": {
        "expected_decision": "ADMISSIBLE",
        "expected_limits": [],
        "description": "Cataract surgery with portability credit from prior insurer; waiting period reduction applies"
    },
    "PUB-011": {
        "expected_decision": "NEEDS_REVIEW",
        "expected_limits": [],
        "description": "Facility does not document compliance with statutory Hospital definition"
    },
    "PUB-012": {
        "expected_decision": "NOT_ADMISSIBLE",
        "expected_limits": [],
        "description": "Experimental therapy; strictly excluded under experimental/unproven treatment exclusion"
    },

    # 5 Candidate-Created Cases
    "CAND-001": {
        "expected_decision": "NOT_ADMISSIBLE",
        "expected_limits": [],
        "description": "Total Knee Replacement at 8 months; specific 1-year joint replacement waiting period applies"
    },
    "CAND-002": {
        "expected_decision": "NEEDS_REVIEW",
        "expected_limits": [],
        "description": "Laparoscopic cholecystectomy in nursing home with unverified hospital statutory criteria"
    },
    "CAND-003": {
        "expected_decision": "ADMISSIBLE",
        "expected_limits": [],
        "description": "Accidental fracture on day 12; trauma is explicit exception to 30-day initial waiting period"
    },
    "CAND-004": {
        "expected_decision": "NEEDS_REVIEW",
        "expected_limits": [],
        "description": "Observation claim with missing discharge summary and unverified clinic criteria"
    },
    "CAND-005": {
        "expected_decision": "ADMISSIBLE_WITH_LIMITS",
        "expected_limits": ["Domiciliary treatment sub-limit"],
        "description": "Domiciliary pneumonia treatment for elderly patient; capped at 20% sum insured sub-limit"
    }
}


def run_evaluation():
    print("=" * 75)
    print("  POLICY-AWARE MULTI-AGENT RAG CLAIM DECISION ENGINE - EVALUATION SUITE  ")
    print("=" * 75)

    # Initialize retriever and pipeline
    print("\n[1/3] Initializing Knowledge Base and Multi-Agent Pipeline...")
    t0_init = time.time()
    retriever = HybridRetriever(settings.policy_chunks_path)
    pipeline = ClaimAdjudicationPipeline(retriever=retriever)
    print(f"      Pipeline initialized in {(time.time() - t0_init)*1000:.1f}ms with {len(retriever.chunks)} policy chunks.")

    # Load test cases
    public_file = (
        root_dir / "data" / "candidate_data" / "public_test_cases.json"
        if (root_dir / "data" / "candidate_data" / "public_test_cases.json").exists()
        else root_dir.parent / "candidate_data" / "public_test_cases.json"
    )
    candidate_file = root_dir / "eval" / "candidate_test_cases.json"

    cases: List[Dict[str, Any]] = []
    if public_file.exists():
        with open(public_file, "r", encoding="utf-8") as f:
            cases.extend(json.load(f))
    if candidate_file.exists():
        with open(candidate_file, "r", encoding="utf-8") as f:
            cases.extend(json.load(f))

    print(f"\n[2/3] Evaluating {len(cases)} total cases ({len(cases)-5} public + 5 candidate-created)...")

    results: List[Dict[str, Any]] = []
    correct_decisions = 0
    total_citations = 0
    grounded_citations = 0
    abstention_expected = 0
    abstention_actual = 0
    latencies = []

    print("-" * 75)
    print(f"{'Case ID':<10} | {'Expected':<23} | {'Actual':<23} | {'Match':<6} | {'Time (ms)'}")
    print("-" * 75)

    for case_data in cases:
        cid = case_data["case_id"]
        exp = EXPECTED_OUTCOMES.get(cid, {})
        exp_decision = exp.get("expected_decision", "UNKNOWN")

        t_start = time.time()
        case_obj = ClaimCase(**case_data)
        result = pipeline.run(case_obj)
        elapsed_ms = (time.time() - t_start) * 1000.0
        latencies.append(elapsed_ms)

        act_decision = result.decision.value
        is_match = (act_decision == exp_decision)
        if is_match:
            correct_decisions += 1

        if exp_decision == "NEEDS_REVIEW":
            abstention_expected += 1
            if act_decision == "NEEDS_REVIEW":
                abstention_actual += 1

        # Citation quality
        case_cits = len(result.citations)
        total_citations += case_cits
        if result.validation.status == "PASS":
            grounded_citations += case_cits

        match_symbol = "✅" if is_match else "❌"
        print(f"{cid:<10} | {exp_decision:<23} | {act_decision:<23} | {match_symbol:<6} | {elapsed_ms:>8.1f}")

        results.append({
            "case_id": cid,
            "expected_decision": exp_decision,
            "actual_decision": act_decision,
            "decision_match": is_match,
            "confidence": result.confidence,
            "latency_ms": round(elapsed_ms, 2),
            "citations_count": len(result.citations),
            "validation_status": result.validation.status,
            "applicable_limits": result.applicable_limits,
            "missing_evidence": result.missing_evidence,
            "description": exp.get("description", "")
        })

    # Summary calculations
    total_cases = len(cases)
    accuracy = (correct_decisions / total_cases) * 100.0 if total_cases > 0 else 0.0
    abstention_fidelity = (abstention_actual / abstention_expected) * 100.0 if abstention_expected > 0 else 0.0
    citation_correctness = (grounded_citations / total_citations) * 100.0 if total_citations > 0 else 100.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    print("-" * 75)
    print("\n[3/3] EVALUATION METRICS SUMMARY:")
    print(f"  • Total Cases Evaluated:    {total_cases} (12 Public + 5 Candidate)")
    print(f"  • Decision Accuracy:        {accuracy:.1f}% ({correct_decisions}/{total_cases})")
    print(f"  • Abstention Fidelity:      {abstention_fidelity:.1f}% ({abstention_actual}/{abstention_expected} NEEDS_REVIEW cases)")
    print(f"  • Citation Correctness:     {citation_correctness:.1f}% ({grounded_citations}/{total_citations} grounded citations)")
    print(f"  • Average Case Latency:     {avg_latency:.1f} ms")
    print("=" * 75)

    # Save detailed evaluation report
    report_path = root_dir / "eval" / "evaluation_report.json"
    report_data = {
        "metrics": {
            "total_cases": total_cases,
            "decision_accuracy_pct": round(accuracy, 2),
            "abstention_fidelity_pct": round(abstention_fidelity, 2),
            "citation_correctness_pct": round(citation_correctness, 2),
            "average_latency_ms": round(avg_latency, 2)
        },
        "detailed_results": results
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\nDetailed evaluation report saved to: {report_path.resolve()}\n")
    return accuracy == 100.0


if __name__ == "__main__":
    success = run_evaluation()
    sys.exit(0 if success else 1)
