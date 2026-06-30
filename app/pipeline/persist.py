"""步骤 10：结果落库 MySQL + Redis 缓存（§13）。"""
from __future__ import annotations

import json

from sqlalchemy import text

from app.config import get_settings
from app.db import get_engine, get_session
from app.models import Base, DailyHotTopic
from app.redis_client import get_redis

CACHE_TTL = 86400  # 当日缓存 1 天


def _cache_key(date_str: str) -> str:
    return f"topics:{date_str}"


_MIGRATIONS = [
    ("business_words", "ADD COLUMN business_words TEXT NOT NULL AFTER hot_words"),
    ("customer_count", "ADD COLUMN customer_count INT NOT NULL DEFAULT 0 AFTER heat"),
    ("prev_heat", "ADD COLUMN prev_heat INT NOT NULL DEFAULT 0 AFTER customer_count"),
    ("heat_change_pct", "ADD COLUMN heat_change_pct FLOAT NULL AFTER prev_heat"),
    ("is_new", "ADD COLUMN is_new TINYINT(1) NOT NULL DEFAULT 0 AFTER heat_change_pct"),
    ("is_surge", "ADD COLUMN is_surge TINYINT(1) NOT NULL DEFAULT 0 AFTER is_new"),
]


def _ensure_columns() -> None:
    """对已存在的旧表做幂等迁移（MySQL 8 不支持 ADD COLUMN IF NOT EXISTS）。"""
    engine = get_engine()
    with engine.begin() as conn:
        for col, ddl in _MIGRATIONS:
            exists = conn.execute(
                text(
                    "SELECT COUNT(*) FROM information_schema.columns "
                    "WHERE table_schema = DATABASE() AND table_name = 'daily_hot_topics' "
                    "AND column_name = :col"
                ),
                {"col": col},
            ).scalar()
            if not exists:
                conn.execute(text(f"ALTER TABLE daily_hot_topics {ddl}"))


def save_results(date_str: str, topics: list[dict]) -> None:
    """topics: 已排序的热点列表（含 rank）。写 MySQL（覆盖当日）+ Redis 缓存。"""
    settings = get_settings()
    Base.metadata.create_all(get_engine())
    _ensure_columns()

    session = get_session()
    try:
        session.query(DailyHotTopic).filter(
            DailyHotTopic.stat_date == date_str
        ).delete(synchronize_session=False)
        for t in topics:
            session.add(
                DailyHotTopic(
                    stat_date=date_str,
                    rank=t["rank"],
                    topic_id=t["topic_id"],
                    heat=t["heat"],
                    customer_count=t.get("customer_count", 0),
                    prev_heat=t.get("prev_heat", 0),
                    heat_change_pct=t.get("heat_change_pct"),
                    is_new=t.get("is_new", False),
                    is_surge=t.get("is_surge", False),
                    hot_words=json.dumps(t["hot_words"], ensure_ascii=False),
                    business_words=json.dumps(
                        t.get("business_words", []), ensure_ascii=False
                    ),
                    customer_intent=t["customer_intent"],
                    representative_docs=json.dumps(
                        t["representative_docs"], ensure_ascii=False
                    ),
                    cluster_backend=settings.cluster_backend,
                )
            )
        session.commit()
    finally:
        session.close()

    # 当日 TOP N 写 Redis 缓存，供 FastAPI 快速返回
    get_redis().set(
        _cache_key(date_str),
        json.dumps(topics, ensure_ascii=False),
        ex=CACHE_TTL,
    )
