import pytest
from src.rag.retriever import HybridRetriever
from src.config import settings


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever(settings.policy_chunks_path)


def test_retriever_sparse_and_dense(retriever):
    results = retriever.retrieve("30 days waiting period", top_k=3, rerank=True)
    assert len(results) > 0
    top = results[0]
    assert "waiting" in top["text"].lower() or "waiting" in top["section"].lower() or "waiting" in top["clause"].lower()
    assert "rerank_score" in top


def test_retriever_hospital_definition(retriever):
    results = retriever.retrieve("hospital definition registered beds nursing", top_k=3, rerank=True)
    assert len(results) > 0
    top = results[0]
    assert "hospital" in top["text"].lower() or "hospital" in top["clause"].lower()


def test_retriever_exclusions(retriever):
    results = retriever.retrieve("cosmetic surgery aesthetic exclusion", top_k=3, rerank=True)
    assert len(results) > 0
    top = results[0]
    assert "cosmetic" in top["text"].lower() or "exclusion" in top["section"].lower()
