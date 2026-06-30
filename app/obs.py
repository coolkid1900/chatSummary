"""可观测性（技术优化5）：结构化日志 + 流水线运行状态/指标落库。

- setup_logging：统一日志格式（时间/级别/模块），各模块用 logging.getLogger(__name__)。
- RunState：在 pipeline_runs 表为本次运行维护**单行**记录，running→success/failed 原地更新，
  附各步计数与耗时，便于排查、监控、判断「某日期是否有任务正在跑」（HTTP 触发去重）。
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime

from sqlalchemy import update

from app.db import get_engine, get_session
from app.models import Base, PipelineRun
from app.redis_client import get_redis

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

log = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt="%H:%M:%S")


class RunState:
    """一次运行的指标累加器 + pipeline_runs 单行 upsert。"""

    def __init__(self, date_str: str, backend: str, run_id: str | None = None):
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.date_str = date_str
        self.backend = backend
        self.t0 = time.time()
        self.metrics: dict[str, int] = {
            "n_messages": 0,
            "n_sessions": 0,
            "n_topics": 0,
            "embed_api_calls": 0,
            "embed_cache_hits": 0,
            "llm_failures": 0,
        }
        Base.metadata.create_all(get_engine())

    def set(self, **kw: int) -> None:
        self.metrics.update(kw)

    def start(self) -> None:
        """插入 running 行。"""
        session = get_session()
        try:
            session.add(
                PipelineRun(
                    run_id=self.run_id,
                    stat_date=self.date_str,
                    status="running",
                    cluster_backend=self.backend,
                    created_at=datetime.utcnow(),
                    **self.metrics,
                )
            )
            session.commit()
        finally:
            session.close()

    def _finalize(self, status: str, error: str = "") -> None:
        """原地更新本次运行的状态、指标与耗时，并释放 Redis 分布式锁。"""
        session = get_session()
        try:
            session.execute(
                update(PipelineRun)
                .where(PipelineRun.run_id == self.run_id)
                .values(
                    status=status,
                    duration_sec=round(time.time() - self.t0, 2),
                    error=error[:2000],
                    **self.metrics,
                )
            )
            session.commit()
        finally:
            session.close()
        # 释放分布式锁，允许该日期的下一次触发
        # 只删自己持有的锁（run_id 匹配），防止误删其他进程的锁
        redis = get_redis()
        lock_key = f"pipeline:lock:{self.date_str}"
        if redis.get(lock_key) == self.run_id:
            redis.delete(lock_key)

    def success(self) -> None:
        self._finalize("success")

    def failed(self, error: str) -> None:
        self._finalize("failed", error)
