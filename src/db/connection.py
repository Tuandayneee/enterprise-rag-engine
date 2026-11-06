import json
import logging
import re
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from src.core.config import get_settings
from src.models import Chunk, ChunkStrategy

logger = logging.getLogger(__name__)


class BaseStorage:
    def upsert_chunks(self, chunks: List[Chunk]) -> bool:
        raise NotImplementedError

    def dense_search(self, query_vector: List[float], top_k: int = 20) -> List[Tuple[Chunk, float]]:
        raise NotImplementedError

    def sparse_search(self, query: str, top_k: int = 20) -> List[Tuple[Chunk, float]]:
        raise NotImplementedError

    def delete_by_document(self, document_id: str) -> bool:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class InMemoryDualStore(BaseStorage):
    """
    Bộ lưu trữ kép In-Memory dự phòng (Dense Cosine + Okapi BM25).
    Đảm bảo kiểm thử và phát triển cục bộ trơn tru khi chưa bật Postgres.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: Dict[str, Chunk] = {}

    def upsert_chunks(self, chunks: List[Chunk]) -> bool:
        for c in chunks:
            self.chunks[c.id] = c
        return True

    def dense_search(self, query_vector: List[float], top_k: int = 20) -> List[Tuple[Chunk, float]]:
        if not self.chunks:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        results = []
        for ch in self.chunks.values():
            if not ch.embedding:
                continue
            c_vec = np.array(ch.embedding, dtype=np.float32)
            c_norm = np.linalg.norm(c_vec)
            if c_norm > 0:
                sim = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))
            else:
                sim = 0.0
            results.append((ch, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def sparse_search(self, query: str, top_k: int = 20) -> List[Tuple[Chunk, float]]:
        if not self.chunks:
            return []

        q_tokens = [t.lower() for t in re.findall(r"\w+", query)]
        if not q_tokens:
            return []

        # Tính toán BM25 đơn giản hóa
        doc_lengths = {cid: len(ch.text.split()) for cid, ch in self.chunks.items()}
        avgdl = sum(doc_lengths.values()) / max(len(doc_lengths), 1)

        scores = []
        for cid, ch in self.chunks.items():
            text_lower = ch.text.lower()
            d_len = doc_lengths[cid]
            score = 0.0
            for qt in q_tokens:
                tf = text_lower.count(qt)
                if tf > 0:
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * (d_len / (avgdl or 1.0)))
                    score += numerator / (denominator or 1.0)
            if score > 0:
                scores.append((ch, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def delete_by_document(self, document_id: str) -> bool:
        to_del = [cid for cid, ch in self.chunks.items() if ch.document_id == document_id]
        for cid in to_del:
            del self.chunks[cid]
        return True

    def count(self) -> int:
        return len(self.chunks)


class PostgresVectorStorage(BaseStorage):
    """
    Bộ lưu trữ chính thức PostgreSQL tích hợp pgvector (HNSW) & tsvector (Full-text Search).
    """

    def __init__(self):
        self.settings = get_settings()
        self._connected = False
        self._conn = None
        self._init_db()

    def _is_port_open(self, host: str, port: int, timeout: float = 0.05) -> bool:
        try:
            import socket
            target_host = "127.0.0.1" if host in ["localhost", "127.0.0.1"] else host
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                return s.connect_ex((target_host, port)) == 0
        except Exception:
            return False

    def _init_db(self):
        if not self._is_port_open(self.settings.postgres_host, self.settings.postgres_port):
            logger.info("PostgreSQL offline. Kích hoạt InMemoryDualStore dự phòng.")
            self._connected = False
            return

        try:
            import psycopg2
            from pgvector.psycopg2 import register_vector

            self._conn = psycopg2.connect(
                host=self.settings.postgres_host,
                port=self.settings.postgres_port,
                dbname=self.settings.postgres_db,
                user=self.settings.postgres_user,
                password=self.settings.postgres_password,
                connect_timeout=2,
            )
            self._conn.autocommit = True
            register_vector(self._conn)

            # Khởi tạo bảng và extension
            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id VARCHAR(128) PRIMARY KEY,
                        document_id VARCHAR(128) NOT NULL,
                        strategy VARCHAR(32) DEFAULT 'semantic',
                        content TEXT NOT NULL,
                        metadata JSONB DEFAULT '{}'::jsonb,
                        embedding vector(1024),
                        tsv_content tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            self._connected = True
            logger.info("Đã kết nối thành công PostgreSQL pgvector.")
        except Exception as e:
            logger.info(f"Không thể kết nối Postgres ({e}). Dùng InMemoryDualStore.")
            self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def upsert_chunks(self, chunks: List[Chunk]) -> bool:
        if not self._connected or not self._conn:
            return False
        import psycopg2.extras
        with self._conn.cursor() as cur:
            query = """
                INSERT INTO document_chunks (id, document_id, strategy, content, metadata, embedding)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding;
            """
            data = [
                (
                    c.id,
                    c.document_id,
                    c.strategy.value if hasattr(c.strategy, "value") else str(c.strategy),
                    c.text,
                    json.dumps(c.metadata),
                    c.embedding,
                )
                for c in chunks
            ]
            psycopg2.extras.execute_values(cur, query, data)
        return True

    def dense_search(self, query_vector: List[float], top_k: int = 20) -> List[Tuple[Chunk, float]]:
        if not self._connected or not self._conn:
            return []
        with self._conn.cursor() as cur:
            # 1 - (embedding <=> query_vector) quy đổi khoảng cách Cosine sang điểm tương đồng Cosine
            cur.execute("""
                SELECT id, document_id, strategy, content, metadata,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s;
            """, (query_vector, query_vector, top_k))
            rows = cur.fetchall()

        results = []
        for r in rows:
            ch = Chunk(
                id=r[0],
                document_id=r[1],
                strategy=ChunkStrategy(r[2]) if r[2] in [e.value for e in ChunkStrategy] else ChunkStrategy.SEMANTIC,
                text=r[3],
                metadata=r[4] if isinstance(r[4], dict) else json.loads(r[4]),
            )
            results.append((ch, float(r[5])))
        return results

    def sparse_search(self, query: str, top_k: int = 20) -> List[Tuple[Chunk, float]]:
        if not self._connected or not self._conn:
            return []
        with self._conn.cursor() as cur:
            # Sử dụng ts_rank_cd kết hợp plainto_tsquery cho Full-Text Search
            cur.execute("""
                SELECT id, document_id, strategy, content, metadata,
                       ts_rank_cd(tsv_content, plainto_tsquery('english', %s)) AS rank
                FROM document_chunks
                WHERE tsv_content @@ plainto_tsquery('english', %s)
                ORDER BY rank DESC
                LIMIT %s;
            """, (query, query, top_k))
            rows = cur.fetchall()

        results = []
        for r in rows:
            ch = Chunk(
                id=r[0],
                document_id=r[1],
                strategy=ChunkStrategy(r[2]) if r[2] in [e.value for e in ChunkStrategy] else ChunkStrategy.SEMANTIC,
                text=r[3],
                metadata=r[4] if isinstance(r[4], dict) else json.loads(r[4]),
            )
            results.append((ch, float(r[5])))
        return results

    def delete_by_document(self, document_id: str) -> bool:
        if not self._connected or not self._conn:
            return False
        with self._conn.cursor() as cur:
            cur.execute("DELETE FROM document_chunks WHERE document_id = %s;", (document_id,))
        return True

    def count(self) -> int:
        if not self._connected or not self._conn:
            return 0
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM document_chunks;")
            return cur.fetchone()[0]


def get_storage() -> BaseStorage:
    pg = PostgresVectorStorage()
    if pg.is_connected():
        return pg
    return InMemoryDualStore()
