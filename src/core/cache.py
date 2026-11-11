import time
import json
import hashlib
import logging
from typing import Optional, Tuple
from src.core.config import get_settings

logger = logging.getLogger(__name__)


class RedisCacheManager:
    """
    Quản lý bộ nhớ đệm (Cache-Aside Pattern) và giới hạn tần suất (Sliding Window Rate Limiter).
    Hỗ trợ tự động fallback in-memory nếu Redis offline.
    """

    def __init__(self):
        self.settings = get_settings()
        self.redis_client = None
        self._in_memory_cache = {}
        self._in_memory_rate_limits = {}
        self._init_redis()

    def _is_port_open(self, host: str, port: int, timeout: float = 0.05) -> bool:
        try:
            import socket

            target_host = "127.0.0.1" if host in ["localhost", "127.0.0.1"] else host
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                return s.connect_ex((target_host, port)) == 0
        except Exception:
            return False

    def _init_redis(self):
        if not self._is_port_open(self.settings.redis_host, self.settings.redis_port):
            logger.info(
                "Redis server offline. Kích hoạt In-Memory Cache & Rate Limiter."
            )
            self.redis_client = None
            return

        try:
            import redis

            self.redis_client = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                password=self.settings.redis_password,
                db=self.settings.redis_db,
                socket_connect_timeout=0.5,
                socket_timeout=0.5,
                decode_responses=True,
            )
            self.redis_client.ping()
            logger.info(
                f"Kết nối Redis thành công tại {self.settings.redis_host}:{self.settings.redis_port}"
            )
        except Exception as e:
            logger.info(f"Lỗi khởi tạo Redis ({e}). Dùng In-Memory.")
            self.redis_client = None

    def is_connected(self) -> bool:
        if not self.redis_client:
            return False
        try:
            return self.redis_client.ping()
        except Exception:
            return False

    # ------------------ CACHE-ASIDE PATTERN ------------------
    def get_cached_response(
        self, query: str, config_prefix: str = ""
    ) -> Optional[dict]:
        """
        Kiểm tra Semantic / Hash Cache. Nếu Hit -> Trả về ngay lập tức (P99 < 10ms).
        """
        digest = hashlib.md5((query + config_prefix).encode("utf-8")).hexdigest()
        key = f"rag_query_cache:{digest}"
        if self.redis_client:
            try:
                cached = self.redis_client.get(key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass
        else:
            item = self._in_memory_cache.get(key)
            if item and item["expires_at"] > time.time():
                return item["data"]
        return None

    def set_cached_response(self, query: str, data: dict, config_prefix: str = ""):
        digest = hashlib.md5((query + config_prefix).encode("utf-8")).hexdigest()
        key = f"rag_query_cache:{digest}"
        ttl = self.settings.redis_cache_ttl
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, json.dumps(data, ensure_ascii=False))
            except Exception:
                pass
        else:
            self._in_memory_cache[key] = {
                "data": data,
                "expires_at": time.time() + ttl,
            }

    def invalidate_all(self):
        """
        Xóa toàn bộ query cache khi có tài liệu mới được nạp hoặc xóa khỏi kho tri thức.
        """
        if self.redis_client:
            try:
                keys = self.redis_client.keys("rag_query_cache:*")
                if keys:
                    self.redis_client.delete(*keys)
            except Exception:
                pass
        self._in_memory_cache.clear()

    # ------------------ SLIDING WINDOW RATE LIMITER ------------------
    def check_rate_limit(self, client_id: str) -> Tuple[bool, int]:
        """
        Sliding Window Rate Limiter bằng Redis Sorted Sets (ZREMRANGEBYSCORE, ZADD, ZCARD).
        """
        limit = self.settings.rate_limit_requests
        window = self.settings.rate_limit_window_seconds
        now = time.time()

        if self.redis_client:
            try:
                key = f"rate_limit:{client_id}"
                pipe = self.redis_client.pipeline()
                pipe.zremrangebyscore(key, 0, now - window)
                pipe.zadd(key, {str(now): now})
                pipe.zcard(key)
                pipe.expire(key, window)
                res = pipe.execute()

                count = res[2]
                remaining = max(0, limit - count)
                return (count <= limit), remaining
            except Exception:
                pass

        # In-Memory fallback
        if client_id not in self._in_memory_rate_limits:
            self._in_memory_rate_limits[client_id] = []

        timestamps = [
            ts for ts in self._in_memory_rate_limits[client_id] if ts > now - window
        ]
        self._in_memory_rate_limits[client_id] = timestamps

        if len(timestamps) < limit:
            timestamps.append(now)
            return True, limit - len(timestamps)
        else:
            return False, 0
