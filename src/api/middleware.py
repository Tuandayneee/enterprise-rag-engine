from fastapi import Request, HTTPException
from src.core.cache import RedisCacheManager

_cache_manager = RedisCacheManager()


def get_cache_manager() -> RedisCacheManager:
    return _cache_manager


def rate_limit_dependency(request: Request):
    client_ip = request.client.host if request.client else "127.0.0.1"
    allowed, remaining = _cache_manager.check_rate_limit(client_ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Tần suất truy vấn vượt quá giới hạn (Rate limit exceeded). Vui lòng thử lại sau.",
        )
    return client_ip
