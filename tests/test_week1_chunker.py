from src.models import ChunkStrategy
from src.pipeline.chunker import (
    FixedSizeChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
    ChunkerFactory,
    ChunkingBenchmark,
)


def test_fixed_size_chunker():
    chunker = FixedSizeChunker(chunk_size=100, overlap=20)
    text = "Đây là một đoạn văn bản mẫu dùng để kiểm thử tính năng cắt cố định (Fixed-size Chunking) trong hệ thống RAG doanh nghiệp."
    chunks = chunker.chunk(text, document_id="doc_test_fixed")
    assert len(chunks) > 0
    assert chunks[0].document_id == "doc_test_fixed"
    assert len(chunks[0].text) <= 100


def test_recursive_character_chunker():
    chunker = RecursiveCharacterChunker(chunk_size=60, overlap=10)
    text = (
        "Đoạn 1 có nội dung ngắn gọn.\n\n"
        "Đoạn 2 có câu văn dài hơn để kiểm tra khả năng tách tự nhiên theo cấu trúc đoạn văn bản và câu văn."
    )
    chunks = chunker.chunk(text, document_id="doc_test_recur")
    assert len(chunks) >= 2


def test_semantic_chunker():
    chunker = SemanticChunker(min_chunk_size=50, max_chunk_size=300)
    text = (
        "Trí tuệ nhân tạo đang thay đổi thế giới mạnh mẽ. "
        "Mô hình ngôn ngữ lớn có khả năng sinh văn bản tự nhiên. "
        "Tuy nhiên hiện tượng ảo giác vẫn là một thách thức lớn. "
        "Kiến trúc RAG ra đời để giải quyết bài toán chống ảo giác này."
    )
    chunks = chunker.chunk(text, document_id="doc_test_semantic")
    assert len(chunks) > 0
    assert chunks[0].embedding is not None
    assert len(chunks[0].embedding) > 0


def test_chunker_factory_all_strategies():
    text = "Kiểm thử ChunkerFactory tự động sinh vector embedding cho cả 3 chiến lược."
    for strat in [
        ChunkStrategy.FIXED_SIZE,
        ChunkStrategy.RECURSIVE,
        ChunkStrategy.SEMANTIC,
    ]:
        chunks = ChunkerFactory.chunk_document(
            text=text, document_id=f"doc_{strat.value}", strategy=strat
        )
        assert len(chunks) > 0
        assert chunks[0].strategy == strat
        assert chunks[0].embedding is not None
        assert len(chunks[0].embedding) > 0


def test_chunking_benchmark():
    sample = "Câu số một ở đây. Câu số hai tiếp theo. Câu số ba kết thúc đoạn."
    results = ChunkingBenchmark.run_benchmark(sample, document_id="bench_test")
    assert len(results) == 3
    for r in results:
        assert r.total_chunks > 0
        assert r.context_completeness_score >= 0.0
