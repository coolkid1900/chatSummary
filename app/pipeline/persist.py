"""步骤 10：结果落库 MySQL + Redis 缓存（§13）。"""
from __future__ import annotations

import json

from sqlalchemy import select, text

from app.config import get_settings
from app.db import get_engine, get_session
from app.models import Base, DailyHotTopic, DailyTopicCatalog
from app.redis_client import get_redis

CACHE_TTL = 86400  # 当日缓存 1 天

GLOBAL_SCOPE = ("global", "all")  # (scope_type, scope_id)；部门为 ("department", 部门 id)


def cache_key(date_str: str, scope: tuple[str, str] = GLOBAL_SCOPE) -> str:
    if scope == GLOBAL_SCOPE:
        return f"topics:{date_str}"
    return f"topics:{date_str}:{scope[0]}:{scope[1]}"


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


def save_results(date_str: str, topics: list[dict], *, scope: tuple[str, str] = GLOBAL_SCOPE,
                 run_id: str = "", catalog: list[dict] | None = None) -> None:
    """topics: 已排序的热点列表（含 rank）。覆盖该日期该范围的榜单 + Redis 缓存。

    catalog 非空时同一事务整天替换主题目录（只有全公司运行会传）。
    """
    settings = get_settings()
    Base.metadata.create_all(get_engine())
    _ensure_columns()

    session = get_session()
    try:
        session.query(DailyHotTopic).filter(
            DailyHotTopic.stat_date == date_str,
            DailyHotTopic.scope_type == scope[0],
            DailyHotTopic.scope_id == scope[1],
        ).delete(synchronize_session=False)
        for t in topics:
            session.add(
                DailyHotTopic(
                    stat_date=date_str,
                    scope_type=scope[0],
                    scope_id=scope[1],
                    run_id=run_id,
                    rank=t["rank"],
                    topic_id=t["topic_id"],
                    parent_topic_id=t.get("parent_topic_id"),
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
                    intent_source=t.get("intent_source", "global"),
                    representative_docs=json.dumps(
                        t["representative_docs"], ensure_ascii=False
                    ),
                    cluster_backend=settings.cluster_backend,
                )
            )
        if catalog is not None:
            session.query(DailyTopicCatalog).filter(
                DailyTopicCatalog.stat_date == date_str
            ).delete(synchronize_session=False)
            for c in catalog:
                session.add(
                    DailyTopicCatalog(
                        stat_date=date_str,
                        topic_id=c["topic_id"],
                        session_count=c["session_count"],
                        heat=c["heat"],
                        customer_count=c["customer_count"],
                        hot_words=json.dumps(c["hot_words"], ensure_ascii=False),
                        business_words=json.dumps(c["business_words"], ensure_ascii=False),
                        customer_intent=c["customer_intent"],
                        representative_docs=json.dumps(c["representative_docs"], ensure_ascii=False),
                        run_id=run_id,
                        cluster_backend=settings.cluster_backend,
                    )
                )
        session.commit()
    finally:
        session.close()

    # 当日 TOP N 写 Redis 缓存，供 FastAPI 快速返回
    get_redis().set(
        cache_key(date_str, scope),
        json.dumps(topics, ensure_ascii=False),
        ex=CACHE_TTL,
    )


def _row_dict(r: DailyHotTopic) -> dict:
    return {
        "date": r.stat_date,
        "scope_type": r.scope_type,
        "scope_id": r.scope_id,
        "rank": r.rank,
        "topic_id": r.topic_id,
        "parent_topic_id": r.parent_topic_id,
        "heat": r.heat,
        "customer_count": r.customer_count,
        "prev_heat": r.prev_heat,
        "heat_change_pct": r.heat_change_pct,
        "is_new": r.is_new,
        "is_surge": r.is_surge,
        "hot_words": json.loads(r.hot_words),
        "business_words": json.loads(r.business_words or "[]"),
        "customer_intent": r.customer_intent,
        "intent_source": r.intent_source,
        "representative_docs": json.loads(r.representative_docs),
        "cluster_backend": r.cluster_backend,
    }


def load_topics(date_str: str, scope: tuple[str, str] = GLOBAL_SCOPE) -> list[dict]:
    """读榜单：优先 Redis 缓存，未命中回查 MySQL 并回填。"""
    key = cache_key(date_str, scope)
    cached = get_redis().get(key)
    if cached is not None:
        return json.loads(cached)
    session = get_session()
    try:
        rows = session.execute(
            select(DailyHotTopic)
            .where(DailyHotTopic.stat_date == date_str, DailyHotTopic.scope_type == scope[0],
                   DailyHotTopic.scope_id == scope[1])
            .order_by(DailyHotTopic.rank)
        ).scalars().all()
        data = [_row_dict(r) for r in rows]
    finally:
        session.close()
    if data:
        get_redis().set(key, json.dumps(data, ensure_ascii=False), ex=CACHE_TTL)
    return data
