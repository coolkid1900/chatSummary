"""FastAPI 服务（§4）。

- 查询接口：仅读已算好的结果（优先 Redis 缓存，未命中回查 MySQL），不在请求里现算。
- 触发接口：POST /pipeline/run 将同步流水线放到 asyncio 后台任务的工作线程执行，
  立即返回 run_id；用 /pipeline/runs/{run_id} 轮询状态。
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import date as date_cls, timedelta

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db import get_engine, get_session
from app import lexicon
from app.api.common import (acquire_run_lock as _acquire_run_lock, check_auth as _check_auth,
                            validate_date as _validate_date)
from app.api.contacts import router as contacts_router
from app.api.departments import DepartmentIds, mock_router, router as departments_router
from app.models import Base, PipelineRun
from app.obs import setup_logging
from app.pipeline.persist import load_topics
from app.pipeline.run_pipeline import run as run_pipeline
from app.api.seed_data import seed_data

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app):
    # 后台流水线与部门热点的运行情况只写日志，API 进程需输出 INFO。
    setup_logging()
    # 建表兜底（幂等）：API 可能先于 worker 启动。失败不阻止启动，查询时再报错。
    try:
        await asyncio.to_thread(Base.metadata.create_all, get_engine())
    except Exception:
        log.exception("启动时建表失败")
    yield


app = FastAPI(title="客户经理聊天热点总结", version="0.1.0", lifespan=lifespan)
app.include_router(contacts_router)
app.include_router(departments_router)
app.include_router(mock_router)

# 保留强引用，确保 create_task 创建的任务在完成前不会被回收。
_pipeline_tasks: set[asyncio.Task[None]] = set()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/topics/{date_str}")
def topics_by_date(date_str: str):
    _validate_date(date_str)
    topics = load_topics(date_str)
    if not topics:
        raise HTTPException(status_code=404, detail=f"无 {date_str} 的热点结果")
    return {"date": date_str, "count": len(topics), "topics": topics}


# ===== 批处理触发 / 状态 =====

class TriggerReq(BaseModel):
    date: str | None = None   # 默认上一天（昨天）
    force: bool = False       # true 覆盖运行锁并忽略已有分片，强制启动完整流水线
    department_ids: DepartmentIds = []  # 父部门 id；全公司完成后按部门名单切片统计


class SeedDataReq(BaseModel):
    date: str | None = None
    customers: int = Field(default=300, ge=1, le=100_000)
    seed: int | None = None


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


async def _run_pipeline_in_background(date_str: str, force: bool, run_id: str,
                                     department_ids: list[str]) -> None:
    """在线程中运行同步流水线，避免阻塞 FastAPI 的事件循环。"""
    try:
        await asyncio.to_thread(run_pipeline, date_str, force, run_id, department_ids or None)
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
    # 普通启动保持同日期互斥；force=true 时覆盖已有锁并立即启动，同时强制重嵌。
    # RunState 只释放 run_id 与自己相同的锁，因此被抢占的旧任务不会误删新锁。
    if not _acquire_run_lock(target, run_id, force=req.force):
        raise HTTPException(status_code=409, detail=f"{target} 已有正在运行的任务")

    # create_task 使 HTTP 请求立即返回；to_thread 保证同步重计算不阻塞事件循环。
    task = asyncio.create_task(
        _run_pipeline_in_background(target, req.force, run_id, req.department_ids),
        name=f"pipeline-{run_id}",
    )
    _pipeline_tasks.add(task)
    task.add_done_callback(_pipeline_tasks.discard)

    return {"run_id": run_id, "date": target, "status": "started", "force": req.force,
            "department_ids": req.department_ids}


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
