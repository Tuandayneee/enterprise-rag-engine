import re
import time
from typing import List, Dict, Any, Optional
import numpy as np
from src.models import Chunk, ChunkStrategy, ChunkingBenchmarkResult
from src.pipeline.embeddings import EmbeddingPipeline
from src.core.config import get_settings


class FixedSizeChunker:
    """
    Chiến lược 1: Cắt cố định (Fixed-size Chunking)
    Cắt chuỗi ký tự theo kích thước cố định (ví dụ 512 chars/tokens) kèm overlap.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, document_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        base_meta = metadata or {}
        chunks: List[Chunk] = []
        start = 0
        text_len = len(text)
        idx = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            sub_text = text[start:end].strip()
            if sub_text:
                chunks.append(
                    Chunk(
                        id=f"{document_id}_fixed_{idx}",
                        document_id=document_id,
                        text=sub_text,
                        strategy=ChunkStrategy.FIXED_SIZE,
                        metadata={**base_meta, "chunk_index": idx, "start_char": start, "end_char": end},
                    )
                )
                idx += 1
            if end >= text_len:
                break
            start += self.chunk_size - self.overlap

        return chunks


class RecursiveCharacterChunker:
    """
    Chiến lược 2: Cắt ký tự đệ quy (Recursive Character Chunking)
    Cắt theo vạch phân cách tự nhiên từ lớn đến nhỏ (Paragraph -> Sentence -> Word -> Char).
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str, document_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        base_meta = metadata or {}
        raw_chunks = self._recursive_split(text, self.separators)
        chunks: List[Chunk] = []

        for idx, sub_text in enumerate(raw_chunks):
            cleaned = sub_text.strip()
            if cleaned:
                chunks.append(
                    Chunk(
                        id=f"{document_id}_recur_{idx}",
                        document_id=document_id,
                        text=cleaned,
                        strategy=ChunkStrategy.RECURSIVE,
                        metadata={**base_meta, "chunk_index": idx, "char_length": len(cleaned)},
                    )
                )
        return chunks

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)
        good_splits: List[str] = []

        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_chunks = self._recursive_split(s, new_separators)
                    final_chunks.extend(other_chunks)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0

        for s in splits:
            length = len(s) + (len(separator) if current_doc else 0)
            if total + length > self.chunk_size:
                if current_doc:
                    doc = separator.join(current_doc)
                    if doc.strip():
                        docs.append(doc)
                    # Keep overlap if possible
                    while total > self.overlap and current_doc:
                        total -= len(current_doc[0]) + len(separator)
                        current_doc.pop(0)
                current_doc.append(s)
                total += len(s)
            else:
                current_doc.append(s)
                total += length

        if current_doc:
            doc = separator.join(current_doc)
            if doc.strip():
                docs.append(doc)
        return docs


class SemanticChunker:
    """
    Chiến lược 3: Phân đoạn ngữ nghĩa (Semantic Chunking)
    Cắt dựa trên biến thiên khoảng cách Cosine giữa các câu liên tiếp bằng Sentence-Transformers.
    """

    def __init__(
        self,
        embedding_pipeline: Optional[EmbeddingPipeline] = None,
        similarity_threshold: Optional[float] = None,
        min_chunk_size: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
    ):
        settings = get_settings()
        self.embedding = embedding_pipeline or EmbeddingPipeline()
        self.similarity_threshold = similarity_threshold or settings.chunk_similarity_threshold
        self.min_chunk_size = min_chunk_size or settings.semantic_min_chunk_size
        self.max_chunk_size = max_chunk_size or settings.semantic_max_chunk_size

    def split_sentences(self, text: str) -> List[str]:
        # Phân rã câu hỗ trợ dấu câu Tiếng Việt và Tiếng Anh
        raw = re.split(r"(?<=[.!?。！？])\s+|\n{2,}", text)
        return [s.strip() for s in raw if s.strip()]

    def chunk(self, text: str, document_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        base_meta = metadata or {}
        sentences = self.split_sentences(text)
        if not sentences:
            return []

        if len(sentences) == 1 or len(text) <= self.max_chunk_size:
            emb = self.embedding.embed_query(text)
            return [
                Chunk(
                    id=f"{document_id}_semantic_0",
                    document_id=document_id,
                    text=text,
                    strategy=ChunkStrategy.SEMANTIC,
                    metadata={**base_meta, "chunk_index": 0, "sentence_count": 1},
                    embedding=emb,
                )
            ]

        # 1. Sinh vector cho từng câu
        sentence_vectors = np.array(self.embedding.embed_documents(sentences))

        # 2. Đo Cosine Distance giữa các câu liên kề
        similarities = []
        for i in range(len(sentence_vectors) - 1):
            va = sentence_vectors[i]
            vb = sentence_vectors[i + 1]
            na = np.linalg.norm(va)
            nb = np.linalg.norm(vb)
            sim = float(np.dot(va, vb) / (na * nb)) if na > 0 and nb > 0 else 1.0
            similarities.append(sim)

        # 3. Gom cụm theo Breakpoint
        chunks_text: List[str] = []
        current: List[str] = [sentences[0]]
        current_len = len(sentences[0])

        for i, sim in enumerate(similarities):
            next_sent = sentences[i + 1]
            should_break = False

            # Ngưỡng biến thiên ngữ nghĩa
            if sim < self.similarity_threshold and current_len >= self.min_chunk_size:
                should_break = True
            elif current_len + len(next_sent) > self.max_chunk_size:
                should_break = True

            if should_break:
                chunks_text.append(" ".join(current))
                current = [next_sent]
                current_len = len(next_sent)
            else:
                current.append(next_sent)
                current_len += len(next_sent) + 1

        if current:
            chunks_text.append(" ".join(current))

        # 4. Sinh vector cho các chunks đã gom
        chunk_embeddings = self.embedding.embed_documents(chunks_text)
        chunks: List[Chunk] = []

        for idx, (txt, emb) in enumerate(zip(chunks_text, chunk_embeddings)):
            chunks.append(
                Chunk(
                    id=f"{document_id}_semantic_{idx}",
                    document_id=document_id,
                    text=txt,
                    strategy=ChunkStrategy.SEMANTIC,
                    metadata={**base_meta, "chunk_index": idx, "char_length": len(txt)},
                    embedding=emb,
                )
            )

        return chunks


class ChunkingBenchmark:
    """
    Công cụ đo lường và đánh giá so sánh 3 chiến lược chunking.
    Đo:
      - Context Completeness (Độ toàn vẹn ngữ cảnh - tỷ lệ giữ nguyên ranh giới câu/đoạn)
      - Std Dev Length (Độ phân tán độ dài chunk)
      - Information Loss (Tỷ lệ mất mát thông tin ký tự)
      - Execution Latency (Thời gian xử lý ms)
    """

    @classmethod
    def run_benchmark(cls, sample_text: str, document_id: str = "benchmark_doc") -> List[ChunkingBenchmarkResult]:
        embedding = EmbeddingPipeline()
        strategies = [
            ("Fixed-size (512 tokens)", FixedSizeChunker(chunk_size=512, overlap=50)),
            ("Recursive Character", RecursiveCharacterChunker(chunk_size=512, overlap=50)),
            ("Semantic (Cosine Shift)", SemanticChunker(embedding_pipeline=embedding)),
        ]

        results = []
        original_len = len(sample_text)

        for name, chunker in strategies:
            t0 = time.perf_counter()
            chunks = chunker.chunk(sample_text, document_id=document_id)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            lengths = [len(c.text) for c in chunks] if chunks else [0]
            avg_len = float(np.mean(lengths))
            std_len = float(np.std(lengths))

            # Context completeness: Kiểm tra các chunk có kết thúc bằng dấu chấm kết câu không
            complete_boundaries = sum(
                1 for c in chunks if c.text.endswith((".", "!", "?", "。", "！", "？", "\n"))
            )
            completeness = complete_boundaries / max(len(chunks), 1)

            # Information loss: Đo tỷ lệ khác biệt tổng ký tự
            total_chunk_chars = sum(lengths)
            info_loss = max(0.0, 1.0 - (total_chunk_chars / max(original_len, 1)))

            results.append(
                ChunkingBenchmarkResult(
                    strategy=name,
                    total_chunks=len(chunks),
                    avg_chunk_length=round(avg_len, 1),
                    std_chunk_length=round(std_len, 1),
                    context_completeness_score=round(completeness, 3),
                    information_loss_ratio=round(info_loss, 3),
                    execution_time_ms=round(elapsed_ms, 2),
                )
            )

        return results
