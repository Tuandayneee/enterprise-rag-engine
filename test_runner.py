import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tests.test_week1_chunker import (
    test_fixed_size_chunker,
    test_recursive_character_chunker,
    test_semantic_chunker,
    test_chunker_factory_all_strategies,
    test_chunking_benchmark,
)
from tests.test_week2_hybrid_search import (
    test_dual_storage_dense_and_sparse,
    test_hybrid_search_with_rrf_and_reranker,
)
from tests.test_week3_rag_chain import (
    test_anti_hallucination_guard_and_citations,
    test_redis_cache_and_rate_limiter,
    test_rag_chain_response,
    test_rag_chain_streaming,
)
from tests.test_week4_evaluator import (
    test_ragas_evaluator_metrics,
    test_ragas_evaluator_unsupported_refusal,
    test_api_evaluate_and_delete_endpoints,
)


def run_all_tests():
    print("=" * 70)
    print("🧪 AUTOMATED 4-WEEK ENTERPRISE RAG TEST SUITE")
    print("=" * 70)

    test_cases = [
        ("Week 1: Fixed-size Chunker", test_fixed_size_chunker),
        ("Week 1: Recursive Character Chunker", test_recursive_character_chunker),
        ("Week 1: Semantic Chunker with Cosine Shift", test_semantic_chunker),
        (
            "Week 1: ChunkerFactory Multi-Strategy & Embeddings",
            test_chunker_factory_all_strategies,
        ),
        ("Week 1: Chunking Strategy Benchmark", test_chunking_benchmark),
        (
            "Week 2: Dual Storage (Dense + BM25 + Metadata Filter)",
            test_dual_storage_dense_and_sparse,
        ),
        (
            "Week 2: Hybrid Search with RRF & Cross-Encoder",
            test_hybrid_search_with_rrf_and_reranker,
        ),
        (
            "Week 3: Guardrails (Anti-Hallucination, Prompt Injection, PII)",
            test_anti_hallucination_guard_and_citations,
        ),
        (
            "Week 3: Redis Cache-Aside, Invalidation & Rate Limiter",
            test_redis_cache_and_rate_limiter,
        ),
        ("Week 3: Custom RAG Chain Response", test_rag_chain_response),
        (
            "Week 3: SSE Streaming Event Generator",
            test_rag_chain_streaming,
        ),
        (
            "Week 4: RAGAS Quality Metrics (Faithfulness, Relevance, Recall)",
            test_ragas_evaluator_metrics,
        ),
        (
            "Week 4: RAGAS Negative Constraint Refusal",
            test_ragas_evaluator_unsupported_refusal,
        ),
        (
            "Week 4: API Evaluate & Delete Document Endpoints",
            test_api_evaluate_and_delete_endpoints,
        ),
    ]

    passed = 0
    failed = 0

    for name, test_func in test_cases:
        try:
            test_func()
            print(f"  ✅ [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ [FAIL] {name}: {str(e)}")
            failed += 1

    print("-" * 70)
    print(f"KẾT QUẢ TỔNG HỢP: {passed} passed, {failed} failed.")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
