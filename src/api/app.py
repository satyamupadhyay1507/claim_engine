import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from ..config import settings
from ..models.claim import ClaimCase
from ..models.decision import AdjudicationResult
from ..rag.retriever import HybridRetriever
from ..agents.workflow import ClaimAdjudicationPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager

# Global pipeline instance
pipeline: ClaimAdjudicationPipeline = None


def get_pipeline() -> ClaimAdjudicationPipeline:
    global pipeline
    if pipeline is None:
        logger.info("Initializing Hybrid Retriever and Agent Pipeline...")
        retriever = HybridRetriever(settings.policy_chunks_path)
        pipeline = ClaimAdjudicationPipeline(retriever=retriever)
        logger.info("Pipeline ready.")
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()
    yield


app = FastAPI(
    title="Policy-Aware Claim Adjudication API",
    version="1.0.0",
    description="Automated, evidence-grounded health insurance claim adjudication engine using multi-agent RAG.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, Any]:
    """Health and readiness probe."""
    p = get_pipeline()
    return {
        "status": "healthy",
        "app_env": settings.app_env,
        "policy_chunks_loaded": len(p.retriever.chunks) if p else 0,
        "version": "1.0.0"
    }


@app.post("/analyze", response_model=AdjudicationResult, status_code=status.HTTP_200_OK)
def analyze_claim(case: ClaimCase) -> AdjudicationResult:
    """
    Adjudicate a single health-insurance claim case against the policy knowledge base.
    Returns structured decision, confidence, key findings, limits, citations, and execution trace.
    """
    p = get_pipeline()
    try:
        logger.info(f"Processing claim case {case.case_id} (Diagnosis: {case.treatment.diagnosis})")
        result = p.run(case)
        return result
    except Exception as e:
        logger.error(f"Error adjudicating case {case.case_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Adjudication error: {str(e)}"
        )


@app.get("/cases", response_model=List[Dict[str, Any]])
def list_sample_cases() -> List[Dict[str, Any]]:
    """Helper endpoint to list public sample cases for demonstration and testing."""
    local_path = Path(__file__).resolve().parent.parent / "data" / "candidate_data" / "public_test_cases.json"
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)
    parent_path = Path(__file__).resolve().parent.parent.parent.parent / "candidate_data" / "public_test_cases.json"
    if parent_path.exists():
        with open(parent_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []
