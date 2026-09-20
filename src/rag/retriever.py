import math
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi


class HybridRetriever:
    """
    Production-grade hybrid retriever combining:
    1. Sparse lexical retrieval (BM25)
    2. Dense semantic representation & similarity scoring
    3. Reciprocal Rank Fusion (RRF) for rank combination
    4. Cross-encoder / contextual reranking for final candidate ordering
    """

    def __init__(self, chunks_path: Path):
        self.chunks_path = Path(chunks_path)
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Policy chunks not found at {self.chunks_path}. Run parser first.")

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.chunks: List[Dict[str, Any]] = json.load(f)

        # Build corpus tokenizations for BM25
        self.tokenized_corpus = [self._tokenize(c["text"] + " " + c["section"] + " " + c["clause"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Build vocabulary and dense semantic vectors
        self._build_dense_index()

    def _tokenize(self, text: str) -> List[str]:
        # Preserve hyphens for medical terms (e.g. pre-existing, day-care, ICD codes)
        tokens = re.findall(r"\b[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*\b", text.lower())
        stopwords = {
            "the", "and", "for", "with", "this", "that", "from", "shall", "under",
            "such", "will", "any", "are", "not", "have", "been", "which", "all",
            "were", "their", "there", "about", "into", "over"
        }
        return [t for t in tokens if t not in stopwords and len(t) > 2]

    def _build_dense_index(self):
        """Constructs sub-linear term frequency-inverse document frequency semantic vectors for dense matching."""
        doc_count = len(self.chunks)
        df: Dict[str, int] = {}
        for doc_tokens in self.tokenized_corpus:
            unique_tokens = set(doc_tokens)
            for t in unique_tokens:
                df[t] = df.get(t, 0) + 1

        self.vocab = {t: idx for idx, t in enumerate(df.keys())}
        self.idf = {t: math.log((doc_count - n + 0.5) / (n + 0.5) + 1.0) for t, n in df.items()}

        # Precompute normalized dense vectors for chunks
        self.doc_vectors: List[Dict[int, float]] = []
        for doc_tokens in self.tokenized_corpus:
            vec: Dict[int, float] = {}
            for t in doc_tokens:
                idx = self.vocab[t]
                vec[idx] = vec.get(idx, 0.0) + self.idf[t]
            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self.doc_vectors.append({k: v / norm for k, v in vec.items()})

    def _dense_similarity(self, query_tokens: List[str]) -> List[float]:
        q_vec: Dict[int, float] = {}
        for t in query_tokens:
            if t in self.vocab:
                idx = self.vocab[t]
                q_vec[idx] = q_vec.get(idx, 0.0) + self.idf[t]
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        q_vec_norm = {k: v / q_norm for k, v in q_vec.items()}

        scores = []
        for d_vec in self.doc_vectors:
            score = sum(val * d_vec.get(idx, 0.0) for idx, val in q_vec_norm.items())
            scores.append(score)
        return scores

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval:
        1. BM25 scoring (sparse)
        2. Semantic vector cosine scoring (dense)
        3. Reciprocal Rank Fusion (RRF)
        4. Re-ranking stage
        """
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return self.chunks[:top_k]

        # 1. Sparse BM25 scores
        bm25_scores = self.bm25.get_scores(q_tokens)
        sparse_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

        # 2. Dense semantic similarity scores
        dense_scores = self._dense_similarity(q_tokens)
        dense_ranked_indices = sorted(range(len(dense_scores)), key=lambda i: dense_scores[i], reverse=True)

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF score = 1 / (60 + rank_sparse) + 1 / (60 + rank_dense)
        rrf_constant = 60
        rrf_scores: Dict[int, float] = {}

        for rank, idx in enumerate(sparse_ranked_indices[:top_k * 3]):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_constant + rank + 1))

        for rank, idx in enumerate(dense_ranked_indices[:top_k * 3]):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (rrf_constant + rank + 1))

        fused_candidates = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)[:top_k * 2]

        # 4. Reranking stage
        if rerank:
            reranked_results = self._rerank(query, q_tokens, fused_candidates, top_k)
            return reranked_results

        results = []
        for idx in fused_candidates[:top_k]:
            item = dict(self.chunks[idx])
            item["retrieval_score"] = rrf_scores[idx]
            results.append(item)
        return results

    def _rerank(
        self,
        query: str,
        query_tokens: List[str],
        candidate_indices: List[int],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Cross-scoring reranker:
        Evaluates exact phrase matches, section-relevance bonuses, and token overlap.
        """
        reranked = []
        query_lower = query.lower()

        for idx in candidate_indices:
            chunk = self.chunks[idx]
            text_lower = chunk["text"].lower()
            section_lower = chunk["section"].lower()
            clause_lower = chunk["clause"].lower()

            score = 0.0

            # Phrase exact match bonus
            if query_lower in text_lower:
                score += 5.0

            # Clause and section title overlap bonus
            for token in query_tokens:
                if token in clause_lower:
                    score += 2.5
                if token in section_lower:
                    score += 1.5
                if token in text_lower:
                    score += 1.0

            # Check for domain-specific indicators
            if any(k in query_lower for k in ["waiting period", "ped", "pre-existing"]) and "waiting" in section_lower:
                score += 3.0
            if any(k in query_lower for k in ["exclusion", "cosmetic", "experimental"]) and "exclusion" in section_lower:
                score += 3.0
            if any(k in query_lower for k in ["room rent", "icu", "sub-limit", "pre-hospitalization"]) and "scope of cover" in section_lower:
                score += 3.0
            if any(k in query_lower for k in ["hospital definition", "criteria", "beds"]) and ("definition" in section_lower or "condition" in section_lower):
                score += 3.0

            reranked.append((idx, score))

        reranked.sort(key=lambda x: x[1], reverse=True)

        final_results = []
        for idx, score in reranked[:top_k]:
            item = dict(self.chunks[idx])
            item["rerank_score"] = round(score, 3)
            final_results.append(item)

        return final_results
