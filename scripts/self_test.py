import sys
import asyncio
import time
from pathlib import Path

# Add root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.models import (
    IngestionRequest,
    SearchQuery,
    ChatQuery,
    ChunkStrategy,
)
from src.api.routes import (
    health_check,
    ingest_document,
    search_hybrid,
    chat_rag,
    chat_rag_stream,
)
from src.api.middleware import get_cache_manager


async def test_full_pipeline():
    print("=" * 75)
    print("🧪 BẮT ĐẦU QUY TRÌNH TỰ KIỂM THỬ TOÀN DIỆN (COMPREHENSIVE SELF-TEST)")
    print("=" * 75)

    # 1. TEST HEALTH ENDPOINT
    print("\n1️⃣ Kiểm tra /api/v1/health...")
    health = await health_check()
    print(f"   -> Status: {health.status.upper()}")
    print(f"   -> Version: {health.version}")
    print(f"   -> Pipeline: {health.active_pipeline}")
    assert health.status in ["healthy", "degraded"], "Health check thất bại!"
    print("   ✅ Health Check PASSED!")

    # 2. TEST INGESTION
    print("\n2️⃣ Kiểm tra /api/v1/ingest (Offline Ingestion Pipeline)...")
    sample_text = (
        "Enterprise RAG Engine là hệ thống hỏi đáp thông minh kết hợp cơ sở tri thức doanh nghiệp. "
        "Dữ liệu được nạp qua thuật toán Semantic Chunking bảo toàn trọn vẹn ranh giới câu và ý nghĩa văn bản. "
        "PostgreSQL pgvector cùng chỉ mục HNSW đảm bảo tốc độ truy vấn vector tương đồng cực nhanh. "
        "Mô hình ngôn ngữ lớn Qwen2.5-7B hoặc Llama-3.1-8B chịu trách nhiệm tổng hợp câu trả lời kèm trích dẫn."
    )
    ingest_req = IngestionRequest(
        content=sample_text,
        document_id="self_test_enterprise_doc",
        strategy=ChunkStrategy.SEMANTIC,
        metadata={"source": "Enterprise_Whitepaper.pdf", "page": 1, "department": "AI-R&D"},
    )
    ingest_res = await ingest_document(ingest_req, _="127.0.0.1")
    print(f"   -> Success: {ingest_res.success}")
    print(f"   -> Document ID: {ingest_res.document_id}")
    print(f"   -> Total Chunks Created: {ingest_res.total_chunks}")
    assert ingest_res.success is True, "Ingestion thất bại!"
    assert ingest_res.total_chunks > 0, "Không có chunk nào được tạo!"
    print("   ✅ Ingestion PASSED!")

    # 3. TEST HYBRID SEARCH & RRF & RERANKER
    print("\n3️⃣ Kiểm tra /api/v1/search (Dense + BM25 + RRF + Cross-Encoder)...")
    search_req = SearchQuery(
        query="PostgreSQL pgvector chỉ mục HNSW",
        top_k=2,
        use_reranker=True,
    )
    s_res = await search_hybrid(search_req, _="127.0.0.1")
    print(f"   -> Query: '{s_res.query}'")
    print(f"   -> Total Candidates Fused via RRF: {s_res.total_candidates}")
    print(f"   -> Execution Time: {s_res.execution_time_ms} ms")
    assert len(s_res.results) > 0, "Không tìm thấy kết quả tìm kiếm lai!"
    top_chunk = s_res.results[0]
    print(f"   -> Top Chunk ID: {top_chunk.id} (Final Score: {top_chunk.final_score:.4f}, RRF: {top_chunk.rrf_score:.4f})")
    assert top_chunk.final_score > 0, "Điểm xếp hạng không hợp lệ!"
    print("   ✅ Hybrid Search PASSED!")

    # 4. TEST CHAT & REDIS CACHE-ASIDE PATTERN
    print("\n4️⃣ Kiểm tra /api/v1/chat & Redis Cache-Aside Pattern...")
    chat_req = ChatQuery(
        question="PostgreSQL sử dụng gì để đảm bảo tốc độ truy vấn vector?",
        top_k=2,
    )

    # Lần 1: Cache Miss
    t0 = time.perf_counter()
    chat_res_1 = await chat_rag(chat_req, _="127.0.0.1")
    lat_1 = (time.perf_counter() - t0) * 1000.0
    print(f"   [Lần 1 - Cache Miss] Latency: {lat_1:.2f} ms | Cached: {chat_res_1.cached}")
    print(f"     -> Answer: {chat_res_1.answer[:90]}...")
    print(f"     -> Citations: {len(chat_res_1.citations)} nguồn trích dẫn")
    assert chat_res_1.cached is False, "Lần đầu tiên phải là Cache Miss!"
    assert len(chat_res_1.citations) > 0, "Phải có nguồn trích dẫn!"

    # Lần 2: Cache Hit
    t1 = time.perf_counter()
    chat_res_2 = await chat_rag(chat_req, _="127.0.0.1")
    lat_2 = (time.perf_counter() - t1) * 1000.0
    print(f"   [Lần 2 - Cache Hit ] Latency: {lat_2:.2f} ms | Cached: {chat_res_2.cached}")
    assert chat_res_2.cached is True, "Lần thứ hai cùng câu hỏi phải là Cache Hit!"
    assert lat_2 < lat_1, "Cache Hit phải nhanh hơn Cache Miss!"
    print(f"     -> Tốc độ phản hồi tăng tốc {lat_1 / max(lat_2, 0.01):.1f}x nhờ Redis Cache-Aside!")
    print("   ✅ Redis Cache-Aside PASSED!")

    # 5. TEST SSE STREAMING
    print("\n5️⃣ Kiểm tra /api/v1/chat/stream (Server-Sent Events Streaming)...")
    streaming_response = await chat_rag_stream(
        q="Semantic Chunking bảo toàn điều gì?",
        top_k=2,
        _="127.0.0.1",
    )
    received_events = []
    async for packet in streaming_response.body_iterator:
        received_events.append(packet)

    citation_events = [e for e in received_events if "event: citation" in e]
    token_events = [e for e in received_events if "event: token" in e]
    done_events = [e for e in received_events if "event: done" in e]

    print(f"   -> Tổng packets SSE nhận được: {len(received_events)}")
    print(f"   -> Citation Event: {'Có' if citation_events else 'Không'}")
    print(f"   -> Token Events: {len(token_events)} tokens đã stream")
    print(f"   -> Done Event: {'Có' if done_events else 'Không'}")

    assert len(citation_events) > 0, "Thiếu event citation trong luồng SSE!"
    assert len(token_events) > 0, "Không có token nào được stream!"
    assert len(done_events) > 0, "Thiếu event done kết thúc luồng SSE!"
    print("   ✅ Server-Sent Events (SSE) Streaming PASSED!")

    # 6. TEST RATE LIMITER
    print("\n6️⃣ Kiểm tra Sliding Window Rate Limiter...")
    cache_mgr = get_cache_manager()
    test_client = "stress_test_client_ip"
    allowed_count = 0
    blocked_count = 0

    # Bắn 70 requests liên tục (ngưỡng limit là 60/phút)
    for _ in range(70):
        allowed, _ = cache_mgr.check_rate_limit(test_client)
        if allowed:
            allowed_count += 1
        else:
            blocked_count += 1

    print(f"   -> Đã gửi: 70 requests | Cho phép: {allowed_count} | Chặn (Throttled): {blocked_count}")
    assert allowed_count == 60, f"Rate limiter cho phép {allowed_count} thay vì đúng 60!"
    assert blocked_count == 10, f"Rate limiter phải chặn đúng 10 requests vượt ngưỡng!"
    print("   ✅ Sliding Window Rate Limiter PASSED!")

    print("\n" + "=" * 75)
    print("🎉 TẤT CẢ 6 HẠNG MỤC KIỂM THỬ HỆ THỐNG ĐỀU ĐẠT CHUẨN 100%!")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
