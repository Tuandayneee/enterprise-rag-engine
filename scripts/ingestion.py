import sys
import uuid
from pathlib import Path
from typing import Optional, List

# Thêm root vào sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.models import Chunk, IngestionResponse
from src.pipeline.extractor import DocumentExtractor
from src.pipeline.chunker import SemanticChunker
from src.pipeline.embeddings import EmbeddingPipeline
from src.db.connection import get_storage, BaseStorage


class OfflineIngestionPipeline:
    """
    [OFFLINE INGESTION PIPELINE]
    Documents (PDF/MD/Logs) -> Semantic Chunking -> Bi-Encoder (Dense) -> pgvector (HNSW)
                                                  -> BM25 / TSVector  -> Postgres Full-Text Index
    """

    def __init__(self, storage: Optional[BaseStorage] = None):
        self.storage = storage or get_storage()
        self.embeddings = EmbeddingPipeline()
        self.chunker = SemanticChunker(embedding_pipeline=self.embeddings)

    def ingest_file(self, file_path: str, document_id: Optional[str] = None) -> IngestionResponse:
        doc_id = document_id or f"doc_{uuid.uuid4().hex[:8]}"
        print(f"📥 Đang bóc tách file: {file_path}")

        text, meta = DocumentExtractor.extract_from_file(file_path)
        chunks: List[Chunk] = self.chunker.chunk(text, document_id=doc_id, metadata=meta)

        if not chunks:
            return IngestionResponse(
                success=False,
                document_id=doc_id,
                total_chunks=0,
                strategy="semantic",
                message="Tài liệu trống hoặc không tạo được chunk.",
            )

        # Nạp vào PostgreSQL (pgvector + tsvector)
        self.storage.upsert_chunks(chunks)
        print(f"✅ Đã nạp thành công {len(chunks)} chunks của tài liệu '{doc_id}' vào Postgres/Vector Storage.")

        return IngestionResponse(
            success=True,
            document_id=doc_id,
            total_chunks=len(chunks),
            strategy="semantic",
            message=f"Đã xử lý và nạp thành công {len(chunks)} chunks.",
        )

    def ingest_raw_text(self, text: str, document_id: Optional[str] = None, source_name: str = "raw_text") -> IngestionResponse:
        doc_id = document_id or f"doc_{uuid.uuid4().hex[:8]}"
        meta = {"source": source_name, "type": "text"}
        chunks: List[Chunk] = self.chunker.chunk(text, document_id=doc_id, metadata=meta)
        self.storage.upsert_chunks(chunks)
        return IngestionResponse(
            success=True,
            document_id=doc_id,
            total_chunks=len(chunks),
            strategy="semantic",
            message=f"Đã nạp {len(chunks)} chunks từ chuỗi văn bản.",
        )


def main():
    print("=" * 70)
    print("🚀 [OFFLINE INGESTION PIPELINE] NẠP DỮ LIỆU VÀO PGVECTOR & FULL-TEXT")
    print("=" * 70)

    pipeline = OfflineIngestionPipeline()

    sample_documents = [
        (
            "doc_arch_spec",
            (
                "Kiến trúc hệ thống RAG doanh nghiệp triển khai tách biệt 2 đường ống độc lập. "
                "Đường ống Ingestion xử lý bóc tách tài liệu PDF, Markdown và Logs. "
                "Cơ sở dữ liệu PostgreSQL sử dụng pgvector với chỉ mục HNSW để lưu trữ vector 1024 chiều. "
                "Đồng thời, cột tsvector và index GIN phục vụ cho việc tìm kiếm từ khóa BM25 chuẩn xác."
            ),
            "System_Architecture_Spec.md",
        ),
        (
            "doc_hybrid_algo",
            (
                "Hybrid Search kết hợp hai nguồn thông tin: Dense Vector Cosine Similarity và Sparse Keyword BM25. "
                "Thuật toán Reciprocal Rank Fusion (RRF) kết hợp bảng xếp hạng bằng công thức 1 / (60 + rank). "
                "Sau đó, Cross-Encoder Reranker dựa trên bge-reranker-large sẽ chấm điểm lại Top-20 ứng viên để chọn ra Top-5 đoạn văn bản tốt nhất."
            ),
            "Hybrid_Search_Algorithm.pdf",
        ),
    ]

    for doc_id, text, src in sample_documents:
        res = pipeline.ingest_raw_text(text=text, document_id=doc_id, source_name=src)
        print(f"  -> Ingested: {doc_id} | Chunks: {res.total_chunks} | Status: {res.message}")

    print("=" * 70)
    print("✨ Quá trình Ingestion hoàn tất!")


if __name__ == "__main__":
    main()
