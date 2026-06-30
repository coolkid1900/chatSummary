"""Redis 连接（限流令牌桶、去重缓存、结果缓存共用）。"""
from __future__ import annotations

from functools import lru_cache

import redis

from app.config import get_settings


@lru_cache
def get_redis() -> redis.Redis:
    settings = get_settings()
    return redis.Redis(
        host=settings.redis_host,
        port=settings.redis_port,
        db=settings.redis_db,
        decode_responses=True,
        protocol=2,  # Redis 5.0.3 不支持 RESP3 的 HELLO，强制 RESP2
    )
