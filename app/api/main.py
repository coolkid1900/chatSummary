"""FastAPI 服务（§4）。

- 查询接口：仅读已算好的结果（优先 Redis 缓存，未命中回查 MySQL），不在请求里现算。
- 触发接口：POST /pipeline/run 将同步流水线放到 asyncio 后台任务的工作线程执行，
  立即返回 run_id；用 /pipeline/runs/{run_id} 轮询状态。
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import date as date_cls, datetime, timedelta

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.config import get_settings
from app.db import get_session
from app import lexicon
from app.models import DailyHotTopic, PipelineRun
from app.pipeline.run_pipeline import run as run_pipeline
from app.redis_client import get_redis
from app.api.seed_data import seed_data

app = FastAPI(title="客户经理聊天热点总结", version="0.1.0")
log = logging.getLogger(__name__)

# 保留强引用，确保 create_task 创建的任务在完成前不会被回收。
_pipeline_tasks: set[asyncio.Task[None]] = set()


def _from_db(date_str: str) -> list[dict]:
    session = get_session()
    try:
        rows = session.execute(
            select(DailyHotTopic)
            .where(DailyHotTopic.stat_date == date_str)
            .order_by(DailyHotTopic.rank)
        ).scalars().all()
        return [
            {
                "date": r.stat_date,
                "rank": r.rank,
                "topic_id": r.topic_id,
                "heat": r.heat,
                "customer_count": r.customer_count,
                "prev_heat": r.prev_heat,
                "heat_change_pct": r.heat_change_pct,
                "is_new": r.is_new,
                "is_surge": r.is_surge,
                "hot_words": json.loads(r.hot_words),
                "business_words": json.loads(r.business_words or "[]"),
                "customer_intent": r.customer_intent,
                "representative_docs": json.loads(r.representative_docs),
                "cluster_backend": r.cluster_backend,
            }
            for r in rows
        ]
    finally:
        session.close()


_TOPICS_CACHE_TTL = 86400  # 热点结果缓存 1 天


def _get_topics(date_str: str) -> list[dict]:
    key = f"topics:{date_str}"
    cached = get_redis().get(key)
    if cached is not None:
        return json.loads(cached)
    data = _from_db(date_str)
    if data:
        get_redis().set(key, json.dumps(data, ensure_ascii=False), ex=_TOPICS_CACHE_TTL)
    return data


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/topics/{date_str}")
def topics_by_date(date_str: str):
    _validate_date(date_str)
    topics = _get_topics(date_str)
    if not topics:
        raise HTTPException(status_code=404, detail=f"无 {date_str} 的热点结果")
    return {"date": date_str, "count": len(topics), "topics": topics}


# ===== 批处理触发 / 状态 =====

class TriggerReq(BaseModel):
    date: str | None = None   # 默认上一天（昨天）
    force: bool = False       # true 忽略已存在分片强制重嵌（不影响并发互斥）


class SeedDataReq(BaseModel):
    date: str | None = None
    customers: int = Field(default=300, ge=1, le=100_000)
    seed: int | None = None


_RUN_LOCK_TTL = 7200  # 最长预期运行时间（秒），防止进程崩溃后锁永不释放


def _acquire_run_lock(date_str: str, run_id: str) -> bool:
    """原子占位：SET NX EX，检查与占位在同一个 Redis 命令内完成，消除竞态窗口。
    返回 True 表示抢到锁（可以启动），False 表示该日期已有进程在跑。
    """
    key = f"pipeline:lock:{date_str}"
    return get_redis().set(key, run_id, nx=True, ex=_RUN_LOCK_TTL) is not None


def _check_auth(token: str | None) -> None:
    expected = get_settings().trigger_token
    if expected and token != expected:
        raise HTTPException(status_code=401, detail="无效的触发 token")


def _validate_date(d: str) -> str:
    try:
        datetime.strptime(d, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"日期格式应为 YYYY-MM-DD: {d}")
    return d


def _run_to_dict(r: PipelineRun) -> dict:
    return {
        "run_id": r.run_id,
        "date": r.stat_date,
        "status": r.status,
        "cluster_backend": r.cluster_backend,
        "n_messages": r.n_messages,
        "n_sessions": r.n_sessions,
        "n_topics": r.n_topics,
        "embed_api_calls": r.embed_api_calls,
        "embed_cache_hits": r.embed_cache_hits,
        "llm_failures": r.llm_failures,
        "duration_sec": r.duration_sec,
        "error": r.error,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


@app.post("/seed-data")
async def create_seed_data(
    req: SeedDataReq = SeedDataReq(),
    x_trigger_token: str | None = Header(default=None),
):
    """覆盖写入指定日期的模拟聊天数据，默认生成当天 300 个客户的会话。"""
    _check_auth(x_trigger_token)
    target = _validate_date(req.date) if req.date else date_cls.today().isoformat()
    result = await asyncio.to_thread(seed_data, target, req.customers, req.seed)
    return {
        "status": "success",
        **result,
        "note": "模拟数据已写入；重新统计该日期时请使用 force=true。",
    }


async def _run_pipeline_in_background(date_str: str, force: bool, run_id: str) -> None:
    """在线程中运行同步流水线，避免阻塞 FastAPI 的事件循环。"""
    try:
        await asyncio.to_thread(run_pipeline, date_str, force, run_id)
    except Exception:
        # run_pipeline 会将失败信息更新到 pipeline_runs；这里仅补充 API 日志。
        log.exception("pipeline background task failed: run_id=%s", run_id)


@app.post("/pipeline/run")
async def trigger_pipeline(
    req: TriggerReq = TriggerReq(),
    x_trigger_token: str | None = Header(default=None),
):
    """触发批处理。默认处理**上一天**数据，可用 body.date 指定日期（YYYY-MM-DD）。"""
    _check_auth(x_trigger_token)
    target = _validate_date(req.date) if req.date else (date_cls.today() - timedelta(days=1)).isoformat()

    run_id = uuid.uuid4().hex[:12]
    # 并发互斥与「是否强制重嵌」解耦：任何情况下同一日期都不允许两个 pipeline 并跑，
    # force 仅控制重嵌（见 --force），不再跳过此处的锁。残留锁靠 TTL 自动过期，
    # 需人工清理时 `redis-cli del pipeline:lock:{date}`。
    if not _acquire_run_lock(target, run_id):
        raise HTTPException(status_code=409, detail=f"{target} 已有正在运行的任务")

    # create_task 使 HTTP 请求立即返回；to_thread 保证同步重计算不阻塞事件循环。
    task = asyncio.create_task(
        _run_pipeline_in_background(target, req.force, run_id),
        name=f"pipeline-{run_id}",
    )
    _pipeline_tasks.add(task)
    task.add_done_callback(_pipeline_tasks.discard)

    return {"run_id": run_id, "date": target, "status": "started", "force": req.force}


@app.get("/pipeline/runs/{run_id}")
def get_run(run_id: str):
    session = get_session()
    try:
        r = session.execute(
            select(PipelineRun).where(PipelineRun.run_id == run_id)
            .order_by(PipelineRun.id.desc()).limit(1)
        ).scalars().first()
        if not r:
            raise HTTPException(status_code=404, detail=f"无此 run_id: {run_id}")
        return _run_to_dict(r)
    finally:
        session.close()


@app.get("/pipeline/runs")
def list_runs(limit: int = 20):
    session = get_session()
    try:
        rows = session.execute(
            select(PipelineRun).order_by(PipelineRun.id.desc()).limit(min(limit, 100))
        ).scalars().all()
        return {"count": len(rows), "runs": [_run_to_dict(r) for r in rows]}
    finally:
        session.close()


# ===== 词库维护（业务词典 / 停用词，§10.1）=====
# 改动在**下次流水线运行**时生效（分词器按运行加载数据库词库）。

class WordsReq(BaseModel):
    words: list[str]


def _check_kind(kind: str) -> str:
    if kind not in lexicon.KINDS:
        raise HTTPException(status_code=400, detail=f"kind 须为 {lexicon.KINDS} 之一")
    return kind


@app.get("/lexicon/{kind}")
def lexicon_list(kind: str, include_disabled: bool = False):
    _check_kind(kind)
    words = lexicon.list_words(kind, include_disabled=include_disabled)
    return {"kind": kind, "count": len(words), "words": words}


@app.post("/lexicon/{kind}")
def lexicon_add(kind: str, req: WordsReq, x_trigger_token: str | None = Header(default=None)):
    _check_kind(kind)
    _check_auth(x_trigger_token)
    added = lexicon.add_words(kind, req.words)
    return {"kind": kind, "added": added, "note": "下次流水线运行生效"}


@app.delete("/lexicon/{kind}/{word}")
def lexicon_delete(kind: str, word: str, x_trigger_token: str | None = Header(default=None)):
    _check_kind(kind)
    _check_auth(x_trigger_token)
    if not lexicon.delete_word(kind, word):
        raise HTTPException(status_code=404, detail=f"{kind} 中无此启用词: {word}")
    return {"kind": kind, "disabled": word, "note": "下次流水线运行生效"}
