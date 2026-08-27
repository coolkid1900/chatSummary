"""ORM 模型：结果表 + 运行状态 + 词库。

原始聊天记录在 20 张分表 user_chat_record_sharding_*（见 app/sharding.py），
由业务方/DBA 统一建表，应用不建表（本地开发建表见 sql/shard_tables.sql）。
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DailyHotTopic(Base):
    """每日 TOP 热点结果（§13）。按 date 查询；hot_words / representative_docs 存 JSON 文本。"""

    __tablename__ = "daily_hot_topics"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    stat_date: Mapped[str] = mapped_column(String(10), index=True)  # YYYY-MM-DD
    rank: Mapped[int] = mapped_column(Integer)
    topic_id: Mapped[int] = mapped_column(Integer)
    heat: Mapped[int] = mapped_column(Integer)  # 该主题客户消息数
    customer_count: Mapped[int] = mapped_column(Integer, default=0)  # 涉及去重客户数（广度）
    prev_heat: Mapped[int] = mapped_column(Integer, default=0)       # 昨日同主题热度（环比基数）
    heat_change_pct: Mapped[float | None] = mapped_column(Float, nullable=True)  # 环比涨幅，新主题 NULL
    is_new: Mapped[bool] = mapped_column(Boolean, default=False)     # 昨日无匹配（新出现）
    is_surge: Mapped[bool] = mapped_column(Boolean, default=False)   # 突增标记
    hot_words: Mapped[str] = mapped_column(Text)  # JSON: [["提前还款",0.41],...]
    # 大模型从 hot_words 中筛出的「有业务意义」热词。JSON: ["提前还款","违约金"]
    business_words: Mapped[str] = mapped_column(Text, default="[]")
    customer_intent: Mapped[str] = mapped_column(Text)
    representative_docs: Mapped[str] = mapped_column(Text)  # JSON: ["...","..."]
    cluster_backend: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("uq_date_rank", "stat_date", "rank", unique=True),
    )


class Lexicon(Base):
    """可维护词库：业务词典(term)、停用词(stopword)、寒暄词(chitchat)（§10.1）。

    首次使用时从 data/dict/*.txt 自动播种，之后通过 API 维护；改动在下次流水线生效。
    """

    __tablename__ = "lexicon"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)  # term | stopword | chitchat
    word: Mapped[str] = mapped_column(String(128))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)  # 软禁用，不删
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint("kind", "word", name="uq_kind_word"),
    )


class PipelineRun(Base):
    """每次批处理运行的状态与指标（可观测性 / 断点续跑，技术优化5）。"""

    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    stat_date: Mapped[str] = mapped_column(String(10), index=True)
    status: Mapped[str] = mapped_column(String(16))  # running | success | failed
    cluster_backend: Mapped[str] = mapped_column(String(32), default="")
    n_messages: Mapped[int] = mapped_column(Integer, default=0)
    n_sessions: Mapped[int] = mapped_column(Integer, default=0)
    n_topics: Mapped[int] = mapped_column(Integer, default=0)
    embed_api_calls: Mapped[int] = mapped_column(Integer, default=0)
    embed_cache_hits: Mapped[int] = mapped_column(Integer, default=0)
    llm_failures: Mapped[int] = mapped_column(Integer, default=0)
    duration_sec: Mapped[float] = mapped_column(default=0.0)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
