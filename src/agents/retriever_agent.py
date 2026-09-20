import time
from typing import List, Dict, Any, Set
from .state import ClaimState, EvidenceItem
from ..models.decision import TraceEntry
from ..rag.retriever import HybridRetriever


class PolicyEvidenceAgent:
    """
    Policy Evidence Agent:
    Executes hybrid retrieval (BM25 + Dense similarity) and cross-relevance reranking
    for each dimension identified by the Case Analysis Agent.
    """

    def __init__(self, retriever: HybridRetriever):
        self.name = "Policy Evidence Agent"
        self.retriever = retriever

    def process(self, state: ClaimState) -> ClaimState:
        t0 = time.time()
        plan = state.investigation_plan

        if not plan or not plan.dimensions:
            # Fallback general query
            queries = [f"{state.case.treatment.diagnosis} {state.case.treatment.procedure} coverage"]
        else:
            queries = plan.dimensions

        seen_chunk_ids: Set[str] = set()
        evidence_list: List[EvidenceItem] = []

        total_retrieved_raw = 0

        for q in queries:
            results = self.retriever.retrieve(query=q, top_k=3, rerank=True)
            total_retrieved_raw += len(results)

            for r in results:
                cid = r["chunk_id"]
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    evidence_list.append(EvidenceItem(
                        chunk_id=cid,
                        source=r.get("source", "policy.pdf"),
                        page=r.get("page", 1),
                        section=r.get("section", "General"),
                        clause=r.get("clause", "Policy Term"),
                        text=r.get("text", ""),
                        score=r.get("rerank_score", 0.0)
                    ))

        # Sort evidence by relevance score
        evidence_list.sort(key=lambda e: e.score, reverse=True)
        state.retrieved_evidence = evidence_list

        elapsed_ms = (time.time() - t0) * 1000.0
        state.trace.append(TraceEntry(
            agent=self.name,
            action=f"Executed hybrid retrieval + reranking across {len(queries)} dimensions",
            timestamp_ms=round(elapsed_ms, 2),
            details={
                "queries_executed": len(queries),
                "total_candidates_evaluated": total_retrieved_raw,
                "deduplicated_evidence_chunks": len(evidence_list)
            }
        ))
        return state
