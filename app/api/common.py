"""各路由共用的鉴权与参数校验。"""
from __future__ import annotations

from datetime import datetime

from fastapi import HTTPException

from app.config import get_settings
from app.redis_client import get_redis

RUN_LOCK_TTL = 7200  # 最长预期运行时间（秒），防止进程崩溃后锁永不释放


def run_lock_key(date_str: str) -> str:
    """同日期的全公司与部门热点共用一把锁（与 obs.RunState 释放的键一致）。"""
    return f"pipeline:lock:{date_str}"


def acquire_run_lock(date_str: str, run_id: str, force: bool = False) -> bool:
    """普通启动使用 SET NX EX 保证同日期互斥；force=True 时 SET EX 抢占锁。"""
    redis = get_redis()
    if force:
        return redis.set(run_lock_key(date_str), run_id, ex=RUN_LOCK_TTL) is not None
    return redis.set(run_lock_key(date_str), run_id, nx=True, ex=RUN_LOCK_TTL) is not None


def release_run_lock(date_str: str, run_id: str) -> None:
    """只删自己持有的锁。"""
    redis = get_redis()
    if redis.get(run_lock_key(date_str)) == run_id:
        redis.delete(run_lock_key(date_str))


def check_auth(token: str | None) -> None:
    expected = get_settings().trigger_token
    if expected and token != expected:
        raise HTTPException(status_code=401, detail="无效的触发 token")


def validate_date(d: str) -> str:
    try:
        datetime.strptime(d, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"日期格式应为 YYYY-MM-DD: {d}")
    return d
