import time
import logging
from typing import List, Dict, Any, Optional
from src.models import Chunk, SearchQuery, SearchResponse
from src.core.config import get_settings
from src.db.connection import BaseStorage, get_storage
from src.pipeline.embeddings import EmbeddingPipeline
from src.pipeline.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """
    Bộ máy tìm kiếm lai Hybrid Search kết hợp:
    1. Dense Vector Retrieval (pgvector Cosine Similarity)
    2. Sparse Keyword Retrieval (Postgres tsvector / BM25)
    3. Thuật toán Reciprocal Rank Fusion (RRF)
    4. Tái xếp hạng Cross-Encoder Reranker (Top-20 -> Top-5)
    """

    def __init__(
        self,
        storage: Optional[BaseStorage] = None,
        embeddings: Optional[EmbeddingPipeline] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.settings = get_settings()
        self.storage = storage or get_storage()
        self.embeddings = embeddings or EmbeddingPipeline()
        self.reranker = reranker or CrossEncoderReranker()

    def search(self, query_request: SearchQuery) -> SearchResponse:
        t0 = time.perf_counter()
        query = query_request.query.strip()
        top_k = query_request.top_k
        candidate_pool = query_request.candidate_pool or self.settings.rerank_candidate_pool
        use_reranker = query_request.use_reranker

        # 1. DENSE RETRIEVAL (Vector Cosine Similarity qua pgvector HNSW)
        query_vec = self.embeddings.embed_query(query)
        dense_results = self.storage.dense_search(query_vector=query_vec, top_k=candidate_pool)

        # 2. SPARSE RETRIEVAL (BM25 / tsvector qua Postgres Full-text Search)
        sparse_results = self.storage.sparse_search(query=query, top_k=candidate_pool)

        # 3. RECIPROCAL RANK FUSION (RRF)
        fused_candidates = self._reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=self.settings.rrf_k,
        )

        # 4. CROSS-ENCODER RERANKER (Top-20 -> Top-5)
        if use_reranker and fused_candidates:
            raw_top_candidates = [item["chunk"] for item in fused_candidates[:candidate_pool]]
            reranked_tuples = self.reranker.rerank(
                query=query,
                candidates=raw_top_candidates,
                top_k=top_k,
            )

            # Gán điểm chi tiết vào model Chunk
            final_chunks: List[Chunk] = []
            lookup = {item["chunk"].id: item for item in fused_candidates}

            for ch, r_score in reranked_tuples:
                original = lookup.get(ch.id, {})
                ch.dense_score = original.get("dense_score")
                ch.bm25_score = original.get("bm25_score")
                ch.rrf_score = original.get("rrf_score")
                ch.rerank_score = r_score
                ch.final_score = r_score
                final_chunks.append(ch)
        else:
            final_chunks = []
            for item in fused_candidates[:top_k]:
                ch = item["chunk"]
                ch.dense_score = item.get("dense_score")
                ch.bm25_score = item.get("bm25_score")
                ch.rrf_score = item.get("rrf_score")
                ch.final_score = item.get("rrf_score", 0.0)
                final_chunks.append(ch)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return SearchResponse(
            query=query,
            total_candidates=len(fused_candidates),
            results=final_chunks,
            execution_time_ms=round(elapsed_ms, 2),
        )

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Any],
        sparse_results: List[Any],
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        RRF Score(d) = sum_{m in M} 1 / (k + rank_m(d))
        """
        rrf_table: Dict[str, Dict[str, Any]] = {}

        # Xử lý Dense ranking
        for rank, (chunk, score) in enumerate(dense_results, start=1):
            if chunk.id not in rrf_table:
                rrf_table[chunk.id] = {
                    "chunk": chunk,
                    "rrf_score": 0.0,
                    "dense_score": float(score),
                    "bm25_score": None,
                }
            rrf_table[chunk.id]["rrf_score"] += 1.0 / (k + rank)

        # Xử lý Sparse ranking
        for rank, (chunk, score) in enumerate(sparse_results, start=1):
            if chunk.id not in rrf_table:
                rrf_table[chunk.id] = {
                    "chunk": chunk,
                    "rrf_score": 0.0,
                    "dense_score": None,
                    "bm25_score": float(score),
                }
            else:
                rrf_table[chunk.id]["bm25_score"] = float(score)
            rrf_table[chunk.id]["rrf_score"] += 1.0 / (k + rank)

        sorted_results = sorted(rrf_table.values(), key=lambda x: x["rrf_score"], reverse=True)
        return sorted_results
