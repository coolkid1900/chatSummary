"""ORM 模型：结果表 + 运行状态 + 词库。

原始聊天记录在 20 张分表 user_chat_record_sharding_*（见 app/sharding.py），
由业务方/DBA 统一建表，应用不建表（本地开发建表见 sql/shard_tables.sql）。
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
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


# ===== 经理×客户接触总结（两层：日笔记 → 月度总结）=====
# 工号/客户 ID 与聊天分表一致使用 utf8mb4_bin，保证大小写敏感的精确匹配。

def _manager_col():
    return mapped_column(String(100, collation="utf8mb4_bin"))


def _customer_col():
    return mapped_column(String(128, collation="utf8mb4_bin"))


class ContactDailyStat(Base):
    """每对(经理,客户)每天的沟通统计，由代码从原始聊天计算（含寒暄）。"""

    __tablename__ = "contact_daily_stats"

    manager_id: Mapped[str] = mapped_column(String(100, collation="utf8mb4_bin"), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(128, collation="utf8mb4_bin"), primary_key=True)
    stat_date: Mapped[date] = mapped_column(Date, primary_key=True)
    msg_count: Mapped[int] = mapped_column(Integer, default=0)
    customer_msg_count: Mapped[int] = mapped_column(Integer, default=0)
    manager_msg_count: Mapped[int] = mapped_column(Integer, default=0)
    # 跨零点的会话归入开始那天，次日的延续段不重复计数
    session_count: Mapped[int] = mapped_column(Integer, default=0)
    customer_initiated: Mapped[int] = mapped_column(Integer, default=0)
    first_msg_time: Mapped[datetime] = mapped_column(DateTime)
    last_msg_time: Mapped[datetime] = mapped_column(DateTime)
    source_fp: Mapped[str] = mapped_column(String(64))  # 当天消息哈希，变化说明有补录
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (Index("idx_cds_mgr_date", "manager_id", "stat_date"),)


class ContactDailyNote(Base):
    """Prompt A 产出的日笔记。只有寒暄的天也写一行（no_business），以区分“未处理”。"""

    __tablename__ = "contact_daily_notes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    manager_id: Mapped[str] = _manager_col()
    customer_id: Mapped[str] = _customer_col()
    note_date: Mapped[date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(16))  # ok | no_business | failed
    topics: Mapped[list] = mapped_column(JSON, default=list)
    customer_points: Mapped[list] = mapped_column(JSON, default=list)
    manager_actions: Mapped[list] = mapped_column(JSON, default=list)
    facts: Mapped[list] = mapped_column(JSON, default=list)
    pending: Mapped[list] = mapped_column(JSON, default=list)
    source_fp: Mapped[str] = mapped_column(String(64), default="")
    prompt_version: Mapped[str] = mapped_column(String(32), default="")
    model: Mapped[str] = mapped_column(String(64), default="")
    run_id: Mapped[str] = mapped_column(String(64), default="")
    llm_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint("manager_id", "customer_id", "note_date", name="uq_cdn_pair_date"),
        Index("idx_cdn_mgr_date", "manager_id", "note_date"),
    )


class ContactSignalEvent(Base):
    """信号开启/关闭事件，信号状态的唯一来源；与日笔记同一事务写入。"""

    __tablename__ = "contact_signal_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    note_id: Mapped[int] = mapped_column(BigInteger)
    manager_id: Mapped[str] = _manager_col()
    customer_id: Mapped[str] = _customer_col()
    event_date: Mapped[date] = mapped_column(Date)
    signal_code: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(8))  # open | closed
    evidence: Mapped[str] = mapped_column(String(200), default="")
    # 证据须为脱敏原文的子串；未通过的只留作审计，不参与时间线汇总
    evidence_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_cse_pair_date", "manager_id", "customer_id", "event_date"),
        # 跨客户信号查询按 (经理, 信号, 日期范围) 过滤；status 在代码里汇总，不参与查询
        Index("idx_cse_mgr_code_date", "manager_id", "signal_code", "event_date"),
    )


class ContactNoteRun(Base):
    """日笔记批次：夜间全量（manager_id=''）或为某位经理补数。"""

    __tablename__ = "contact_note_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    stat_date: Mapped[date] = mapped_column(Date)
    manager_id: Mapped[str] = mapped_column(String(100, collation="utf8mb4_bin"), default="")
    trigger_type: Mapped[str] = mapped_column(String(16))  # nightly | backfill | manual
    force: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    pair_count: Mapped[int] = mapped_column(Integer, default=0)
    llm_pair_count: Mapped[int] = mapped_column(Integer, default=0)
    skipped_count: Mapped[int] = mapped_column(Integer, default=0)
    reused_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    attempt_token: Mapped[str] = mapped_column(String(64), default="")
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_cnr_date_mgr", "stat_date", "manager_id", "status"),
        Index("idx_cnr_status", "status", "created_at"),
    )


class ContactSummary(Base):
    """Prompt B 产出的月度总结，按 (经理, 客户, 窗口截止日) 唯一。"""

    __tablename__ = "contact_summaries"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    manager_id: Mapped[str] = _manager_col()
    customer_id: Mapped[str] = _customer_col()
    window_start: Mapped[date] = mapped_column(Date)
    window_end: Mapped[date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(16))  # success | template | review
    stats_json: Mapped[dict] = mapped_column(JSON, default=dict)
    signals_json: Mapped[list] = mapped_column(JSON, default=list)
    pending_json: Mapped[list] = mapped_column(JSON, default=list)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    main_topics: Mapped[list] = mapped_column(JSON, default=list)
    suggestions: Mapped[list] = mapped_column(JSON, default=list)
    incomplete_days: Mapped[int] = mapped_column(Integer, default=0)
    input_fp: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[str] = mapped_column(String(32), default="")
    model: Mapped[str] = mapped_column(String(64), default="")
    job_id: Mapped[str] = mapped_column(String(64), default="")
    llm_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    validation_error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint("manager_id", "customer_id", "window_end", name="uq_cs_pair_window"),
        Index("idx_cs_mgr_window", "manager_id", "window_end"),
    )


class ContactSummaryJob(Base):
    """按需生成某经理月度总结的异步任务。"""

    __tablename__ = "contact_summary_jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    manager_id: Mapped[str] = _manager_col()
    window_end: Mapped[date] = mapped_column(Date)
    customer_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    force: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    stage: Mapped[str] = mapped_column(String(32), default="queued")
    total_customers: Mapped[int] = mapped_column(Integer, default=0)
    done_customers: Mapped[int] = mapped_column(Integer, default=0)
    reused_count: Mapped[int] = mapped_column(Integer, default=0)
    template_count: Mapped[int] = mapped_column(Integer, default=0)
    review_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    attempt_token: Mapped[str] = mapped_column(String(64), default="")
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_csj_mgr_window", "manager_id", "window_end", "status"),
        Index("idx_csj_status", "status", "created_at"),
    )


class LlmCall(Base):
    """大模型调用日志：校准 token 估算、排查限流与统计成本。不保存 prompt 原文。"""

    __tablename__ = "llm_calls"

    call_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    purpose: Mapped[str] = mapped_column(String(16))  # note | summary
    manager_id: Mapped[str] = mapped_column(String(100, collation="utf8mb4_bin"), default="")
    customer_id: Mapped[str] = mapped_column(String(128, collation="utf8mb4_bin"), default="")
    est_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    http_status: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(16))  # ok | failed | rate_limited | invalid
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    prompt_version: Mapped[str] = mapped_column(String(32), default="")
    model: Mapped[str] = mapped_column(String(64), default="")
    error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("idx_llm_created", "created_at"),)
