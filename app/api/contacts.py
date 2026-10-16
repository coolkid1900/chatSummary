"""经理×客户接触总结接口：全部为 POST，参数放在 body（工号、客户 ID 不出现在 URL 与访问日志）。

参数校验失败统一返回 400 {"detail": "..."}，与现有接口一致。
鉴权沿用 X-Trigger-Token（服务级）；按操作人的数据权限需由调用方网关保证。
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Annotated, Literal

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.api.common import check_auth
from app.config import get_settings
from app.contacts import jobs
from app.contacts.prompts import SIGNALS, signal_name
from app.contacts.summary import build_signal_timeline, window_bounds
from app.contacts.transcript import period_label
from app.db import get_session
from app.models import (ContactDailyNote, ContactDailyStat, ContactNoteRun, ContactSignalEvent,
                        ContactSummary, ContactSummaryJob)


class _BadRequestRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def wrapped(request: Request):
            try:
                return await handler(request)
            except RequestValidationError as exc:
                detail = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:]) or 'body'}: {e['msg']}"
                                   for e in exc.errors())
                return JSONResponse(status_code=400, content={"detail": detail})

        return wrapped


router = APIRouter(prefix="/contacts", tags=["contacts"], route_class=_BadRequestRoute)

ManagerId = Annotated[str, Field(min_length=1, max_length=100)]
CustomerId = Annotated[str, Field(min_length=1, max_length=128)]


def _yesterday() -> date:
    return date.today() - timedelta(days=1)


def _past_date(value: date | None, field: str) -> date:
    value = value or _yesterday()
    if value >= date.today():
        raise HTTPException(400, f"{field} 须早于今天（日笔记次日凌晨生成）")
    return value


def _iso(value) -> str | None:
    return value.isoformat(sep=" ") if hasattr(value, "hour") else value.isoformat() if value else None


def _page(items: list, page: int, page_size: int) -> list:
    return items[(page - 1) * page_size: page * page_size]


# ===== 1. 提交总结任务 =====

class SummarySubmitReq(BaseModel):
    manager_id: ManagerId
    end_date: date | None = None
    customer_ids: list[str] | None = Field(default=None, max_length=200)
    force: bool = False


@router.post("/summary/submit", status_code=202)
def summary_submit(req: SummarySubmitReq, x_trigger_token: str | None = Header(default=None)):
    check_auth(x_trigger_token)
    end = _past_date(req.end_date, "end_date")
    job, deduplicated = jobs.enqueue_summary_job(req.manager_id, end, req.customer_ids, req.force)
    start, end = window_bounds(end, get_settings().contact_window_days)
    return {"job_id": job.job_id, "manager_id": req.manager_id, "window_start": start.isoformat(),
            "window_end": end.isoformat(), "status": job.status, "deduplicated": deduplicated}


# ===== 2. 查询总结任务 =====

class JobQueryReq(BaseModel):
    job_id: str = Field(min_length=1, max_length=64)


def _eta_seconds(job: ContactSummaryJob) -> int | None:
    """粗略估算：剩余客户数 × 已完成客户的平均 token ÷ 每秒可用 token。"""
    if job.status != "running" or job.stage != "summarizing":
        return None
    settings = get_settings()
    remaining = max(0, job.total_customers - job.done_customers)
    called = max(1, job.done_customers - job.reused_count - job.template_count)
    avg = job.input_tokens / called if job.input_tokens else 1800
    per_sec = settings.contact_llm_tokens_per_min * settings.contact_llm_token_safety / 60
    return int(remaining * avg / per_sec)


def job_dict(job: ContactSummaryJob) -> dict:
    start, end = window_bounds(job.window_end, get_settings().contact_window_days)
    return {
        "job_id": job.job_id, "manager_id": job.manager_id,
        "window_start": start.isoformat(), "window_end": end.isoformat(),
        "status": job.status, "stage": job.stage,
        "progress": {"total_customers": job.total_customers, "done_customers": job.done_customers,
                     "reused": job.reused_count, "template": job.template_count,
                     "review": job.review_count, "failed": job.failed_count},
        "input_tokens": job.input_tokens, "eta_seconds": _eta_seconds(job),
        "error": job.error or None, "created_at": _iso(job.created_at),
        "started_at": _iso(job.started_at), "finished_at": _iso(job.finished_at),
    }


@router.post("/summary-job/query")
def summary_job_query(req: JobQueryReq, x_trigger_token: str | None = Header(default=None)):
    check_auth(x_trigger_token)
    session = get_session()
    try:
        job = session.get(ContactSummaryJob, req.job_id)
    finally:
        session.close()
    if job is None:
        raise HTTPException(404, f"任务不存在: {req.job_id}")
    return job_dict(job)


# ===== 3. 客户总结列表 =====

class SummaryListReq(BaseModel):
    manager_id: ManagerId
    end_date: date | None = None
    signal: str | None = None
    signal_status: Literal["open", "closed"] | None = None
    status: Literal["success", "template", "review"] | None = None
    sort: Literal["last_contact", "session_count", "signal_change"] = "last_contact"
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


def _signal_brief(row: ContactSummary) -> tuple[list[dict], list[dict]]:
    opened, closed = [], []
    for s in row.signals_json or []:
        if s["code"] not in SIGNALS:
            continue  # 代码表调整前生成的旧结果
        brief = {"code": s["code"], "name": s["name"], "category": SIGNALS[s["code"]].category}
        if s["status"] == "open":
            if "confirmed" in s:
                brief["confirmed"] = s["confirmed"]
            opened.append(brief)
        else:
            closed.append({**brief, "closed_period": period_label(date.fromisoformat(s["closed_date"]))})
    return opened, closed


def _list_item(row: ContactSummary) -> dict:
    opened, closed = _signal_brief(row)
    stats = row.stats_json or {}
    return {"customer_id": row.customer_id, "status": row.status, "summary": row.summary,
            "main_topics": row.main_topics,
            "stats": {k: stats.get(k) for k in ("session_count", "customer_initiated",
                                                 "active_days", "last_contact")},
            "open_signals": opened, "closed_signals": closed,
            "incomplete_days": row.incomplete_days, "updated_at": _iso(row.updated_at)}


def _latest_job(session, manager_id: str, end: date) -> ContactSummaryJob | None:
    return session.execute(select(ContactSummaryJob).where(
        ContactSummaryJob.manager_id == manager_id, ContactSummaryJob.window_end == end)
        .order_by(ContactSummaryJob.created_at.desc()).limit(1)).scalars().first()


@router.post("/summary/list")
def summary_list(req: SummaryListReq, x_trigger_token: str | None = Header(default=None)):
    check_auth(x_trigger_token)
    if req.signal and req.signal not in SIGNALS:
        raise HTTPException(400, f"未知信号代码: {req.signal}")
    if req.signal_status and not req.signal:
        raise HTTPException(400, "signal_status 须与 signal 一起使用")
    end = _past_date(req.end_date, "end_date")
    start, end = window_bounds(end, get_settings().contact_window_days)
    session = get_session()
    try:
        rows = session.execute(select(ContactSummary).where(
            ContactSummary.manager_id == req.manager_id, ContactSummary.window_end == end)).scalars().all()
        job = _latest_job(session, req.manager_id, end)
    finally:
        session.close()
    if not rows and job is None:
        raise HTTPException(404, "该经理在此窗口尚无总结，请先提交生成任务")

    if req.status:
        rows = [r for r in rows if r.status == req.status]
    if req.signal:
        def matches(r):
            s = next((x for x in r.signals_json or [] if x["code"] == req.signal), None)
            return s is not None and (req.signal_status is None or s["status"] == req.signal_status)
        rows = [r for r in rows if matches(r)]
    last = lambda r: (r.stats_json or {}).get("last_contact") or ""
    if req.sort == "session_count":
        rows.sort(key=lambda r: ((r.stats_json or {}).get("session_count", 0), last(r)), reverse=True)
    elif req.sort == "signal_change":
        rows.sort(key=lambda r: (any(s["status"] == "closed" for s in r.signals_json or []), last(r)),
                  reverse=True)
    else:
        rows.sort(key=last, reverse=True)
    return {
        "manager_id": req.manager_id, "window_start": start.isoformat(), "window_end": end.isoformat(),
        "job": None if job is None else {"job_id": job.job_id, "status": job.status,
                                         "done_customers": job.done_customers,
                                         "total_customers": job.total_customers},
        "total": len(rows), "page": req.page, "page_size": req.page_size,
        "items": [_list_item(r) for r in _page(rows, req.page, req.page_size)],
    }


# ===== 4. 单个客户总结详情 =====

class SummaryDetailReq(BaseModel):
    manager_id: ManagerId
    customer_id: CustomerId
    end_date: date | None = None


@router.post("/summary/detail")
def summary_detail(req: SummaryDetailReq, x_trigger_token: str | None = Header(default=None)):
    check_auth(x_trigger_token)
    end = _past_date(req.end_date, "end_date")
    session = get_session()
    try:
        row = session.execute(select(ContactSummary).where(
            ContactSummary.manager_id == req.manager_id, ContactSummary.customer_id == req.customer_id,
            ContactSummary.window_end == end)).scalars().first()
    finally:
        session.close()
    if row is None:
        raise HTTPException(404, "该客户在此窗口尚无总结")
    return {
        "manager_id": row.manager_id, "customer_id": row.customer_id,
        "window_start": row.window_start.isoformat(), "window_end": row.window_end.isoformat(),
        "status": row.status, "summary": row.summary, "main_topics": row.main_topics,
        "suggestions": row.suggestions, "stats": row.stats_json, "signals": row.signals_json,
        "pending": row.pending_json, "incomplete_days": row.incomplete_days,
        "validation_error": row.validation_error,
        "meta": {"prompt_version": row.prompt_version, "model": row.model or None,
                 "generated_at": _iso(row.updated_at)},
    }


# ===== 5. 单个客户日笔记 =====

class NoteListReq(BaseModel):
    manager_id: ManagerId
    customer_id: CustomerId
    start_date: date
    end_date: date
    include_no_business: bool = False


@router.post("/note/list")
def note_list(req: NoteListReq, x_trigger_token: str | None = Header(default=None)):
    check_auth(x_trigger_token)
    if req.end_date < req.start_date or (req.end_date - req.start_date).days > 30:
        raise HTTPException(400, "查询区间须为 1～31 天")
    pair = dict(manager_id=req.manager_id, customer_id=req.customer_id)
    session = get_session()
    try:
        notes = session.execute(select(ContactDailyNote).filter_by(**pair).where(
            ContactDailyNote.note_date.between(req.start_date, req.end_date))
            .order_by(ContactDailyNote.note_date)).scalars().all()
        stats = {r.stat_date: r for r in session.execute(select(ContactDailyStat).filter_by(**pair).where(
            ContactDailyStat.stat_date.between(req.start_date, req.end_date))).scalars()}
        events = session.execute(select(ContactSignalEvent).filter_by(**pair).where(
            ContactSignalEvent.event_date.between(req.start_date, req.end_date))
            .order_by(ContactSignalEvent.id)).scalars().all()
    finally:
        session.close()
    by_note: dict[int, list] = {}
    for e in events:
        by_note.setdefault(e.note_id, []).append(
            {"code": e.signal_code, "status": e.status, "evidence": e.evidence,
             "verified": e.evidence_verified})
    done = {n.note_date for n in notes if n.status in ("ok", "no_business")}
    items = []
    for n in notes:
        if n.status == "no_business" and not req.include_no_business:
            continue
        s = stats.get(n.note_date)
        items.append({
            "date": n.note_date.isoformat(), "status": n.status,
            "stats": {"session_count": s.session_count, "msg_count": s.msg_count} if s else None,
            "topics": n.topics, "customer_points": n.customer_points,
            "manager_actions": n.manager_actions, "facts": n.facts, "pending": n.pending,
            "signals": by_note.get(n.id, []), "error": n.error,
        })
    return {**pair, "start_date": req.start_date.isoformat(), "end_date": req.end_date.isoformat(),
            "items": items, "missing_dates": sorted(d.isoformat() for d in stats if d not in done)}


# ===== 6. 跨客户信号查询 =====

class SignalListReq(BaseModel):
    manager_id: ManagerId
    code: str
    status: Literal["open", "closed"] | None = None
    end_date: date | None = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


@router.post("/signal/list")
def signal_list(req: SignalListReq, x_trigger_token: str | None = Header(default=None)):
    check_auth(x_trigger_token)
    if req.code not in SIGNALS:
        raise HTTPException(400, f"未知信号代码: {req.code}")
    end = _past_date(req.end_date, "end_date")
    start, end = window_bounds(end, get_settings().contact_window_days)
    session = get_session()
    try:
        events = session.execute(
            select(ContactSignalEvent)
            .join(ContactDailyNote, ContactDailyNote.id == ContactSignalEvent.note_id)
            .where(ContactSignalEvent.manager_id == req.manager_id,
                   ContactSignalEvent.signal_code == req.code,
                   ContactSignalEvent.event_date.between(start, end),
                   ContactSignalEvent.evidence_verified.is_(True),
                   ContactDailyNote.status == "ok")
            .order_by(ContactSignalEvent.event_date, ContactSignalEvent.id)).scalars().all()
    finally:
        session.close()
    by_customer: dict[str, list] = {}
    for e in events:
        by_customer.setdefault(e.customer_id, []).append(e)
    items = []
    for customer, evs in by_customer.items():
        (entry,) = build_signal_timeline(evs)
        if req.status and entry["status"] != req.status:
            continue
        items.append({"customer_id": customer, "status": entry["status"],
                      "first_date": entry["first_date"], "last_date": entry["last_date"],
                      "open_days": entry["open_days"], "closed_date": entry.get("closed_date"),
                      "latest_evidence": entry.get("latest_evidence") or entry.get("evidence")})
    items.sort(key=lambda x: x["last_date"], reverse=True)
    return {"manager_id": req.manager_id, "code": req.code, "name": signal_name(req.code),
            "status": req.status, "window_start": start.isoformat(), "window_end": end.isoformat(),
            "total": len(items), "page": req.page, "page_size": req.page_size,
            "items": _page(items, req.page, req.page_size)}


# ===== 7. 触发日笔记批次 =====

class NoteRunSubmitReq(BaseModel):
    date: date
    manager_id: str | None = Field(default=None, min_length=1, max_length=100)
    force: bool = False


@router.post("/note-run/submit", status_code=202)
def note_run_submit(req: NoteRunSubmitReq, x_trigger_token: str | None = Header(default=None)):
    check_auth(x_trigger_token)
    day = _past_date(req.date, "date")
    try:
        run = jobs.enqueue_note_run(day, req.manager_id or "", req.force, trigger_type="manual")
    except jobs.Conflict as exc:
        raise HTTPException(409, str(exc))
    return {"run_id": run.run_id, "date": day.isoformat(), "manager_id": req.manager_id,
            "status": run.status}


# ===== 8. 查询日笔记批次 =====

class NoteRunQueryReq(BaseModel):
    run_id: str = Field(min_length=1, max_length=64)


@router.post("/note-run/query")
def note_run_query(req: NoteRunQueryReq, x_trigger_token: str | None = Header(default=None)):
    check_auth(x_trigger_token)
    session = get_session()
    try:
        run = session.get(ContactNoteRun, req.run_id)
    finally:
        session.close()
    if run is None:
        raise HTTPException(404, f"批次不存在: {req.run_id}")
    return {
        "run_id": run.run_id, "date": run.stat_date.isoformat(), "manager_id": run.manager_id or None,
        "trigger": run.trigger_type, "force": run.force, "status": run.status,
        "pair_count": run.pair_count, "llm_pair_count": run.llm_pair_count,
        "skipped_count": run.skipped_count, "reused_count": run.reused_count,
        "failed_count": run.failed_count, "input_tokens": run.input_tokens,
        "started_at": _iso(run.started_at), "finished_at": _iso(run.finished_at),
        "error": run.error or None,
    }
