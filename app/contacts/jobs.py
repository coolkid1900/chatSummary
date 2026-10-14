"""MySQL 持久化队列：日笔记批次（contact_note_runs）与总结任务（contact_summary_jobs）。

领取任务时生成 attempt_token，之后的心跳与状态写入都以 token 为条件（fencing）：
- 后台线程每 lease/4 秒续约；租约过期的 running 任务可被其他 worker 接管。
- 进程正常停止时把任务放回 queued，不标记失败。
- 接管后旧进程的写入因 token 不符被拒绝；笔记和总结按唯一键 upsert，重复执行结果一致。
"""
from __future__ import annotations

import hashlib
import threading
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta

from sqlalchemy import and_, or_, select, text, update

from app.config import get_settings
from app.db import get_engine, get_session
from app.models import ContactDailyNote, ContactNoteRun, ContactSummaryJob

ACTIVE = ("queued", "running")
DONE_OK = ("success", "partial_success")


class LeaseLost(RuntimeError):
    pass


class Conflict(ValueError):
    pass


@contextmanager
def named_lock(name: str, timeout: int = 10):
    """MySQL 命名锁，保证“检查是否已有任务 + 创建任务”原子执行。

    锁属于数据库连接，必须用独立连接持有：Session 提交后会把连接还给连接池，
    在 Session 上释放可能落到另一条连接，导致锁泄漏。
    """
    name = "contact:" + hashlib.sha1(name.encode("utf-8")).hexdigest()  # 锁名上限 64 字符
    conn = get_engine().connect()
    try:
        if conn.execute(text("SELECT GET_LOCK(:n, :t)"), {"n": name, "t": timeout}).scalar() != 1:
            raise RuntimeError(f"获取锁超时: {name}")
        try:
            yield
        finally:
            conn.execute(text("SELECT RELEASE_LOCK(:n)"), {"n": name})
    finally:
        conn.close()


def _now() -> datetime:
    return datetime.utcnow()


# ===== 提交 =====

def enqueue_note_run(stat_date: date, manager_id: str = "", force: bool = False,
                     trigger_type: str = "manual") -> ContactNoteRun:
    session = get_session()
    try:
        with named_lock(f"contact-note-run:{stat_date}:{manager_id}"):
            busy = session.execute(select(ContactNoteRun.run_id).where(
                ContactNoteRun.stat_date == stat_date, ContactNoteRun.manager_id == manager_id,
                ContactNoteRun.status.in_(ACTIVE))).scalars().first()
            if busy:
                raise Conflict(f"{stat_date} 已有排队或运行中的日笔记批次 {busy}")
            run = ContactNoteRun(run_id=f"nr_{uuid.uuid4().hex[:12]}", stat_date=stat_date,
                                 manager_id=manager_id, trigger_type=trigger_type, force=force,
                                 status="queued", error="")
            session.add(run)
            session.commit()
            return run
    finally:
        session.close()


def ensure_nightly_run(stat_date: date) -> ContactNoteRun | None:
    """夜间批次每个日期只自动创建一次；失败后由人工通过接口重跑。"""
    session = get_session()
    try:
        with named_lock(f"contact-note-run:{stat_date}:"):
            exists = session.execute(select(ContactNoteRun.run_id).where(
                ContactNoteRun.stat_date == stat_date, ContactNoteRun.manager_id == "",
                ContactNoteRun.trigger_type == "nightly")).scalars().first()
            if exists:
                return None
            run = ContactNoteRun(run_id=f"nr_{uuid.uuid4().hex[:12]}", stat_date=stat_date,
                                 manager_id="", trigger_type="nightly", status="queued", error="")
            session.add(run)
            session.commit()
            return run
    finally:
        session.close()


def enqueue_summary_job(manager_id: str, window_end: date, customer_ids: list[str] | None,
                        force: bool) -> tuple[ContactSummaryJob, bool]:
    """返回 (任务, 是否复用已有任务)。已有任务覆盖本次请求范围时直接复用。"""
    wanted = sorted(set(customer_ids)) if customer_ids else None
    session = get_session()
    try:
        with named_lock(f"contact-summary-job:{manager_id}:{window_end}"):
            for job in session.execute(select(ContactSummaryJob).where(
                    ContactSummaryJob.manager_id == manager_id,
                    ContactSummaryJob.window_end == window_end,
                    ContactSummaryJob.status.in_(ACTIVE))).scalars():
                covers = job.customer_ids is None or (wanted is not None and set(wanted) <= set(job.customer_ids))
                if covers and (job.force or not force):
                    return job, True
            job = ContactSummaryJob(job_id=f"csj_{uuid.uuid4().hex[:12]}", manager_id=manager_id,
                                    window_end=window_end, customer_ids=wanted, force=force,
                                    status="queued", stage="queued", error="")
            session.add(job)
            session.commit()
            return job, False
    finally:
        session.close()


# ===== 领取与租约 =====

def _claim(model, key_attr: str):
    lease_cutoff = _now() - timedelta(seconds=get_settings().contact_job_lease_seconds)
    session = get_session()
    try:
        cond = or_(model.status == "queued",
                   and_(model.status == "running", model.heartbeat_at < lease_cutoff))
        row = session.execute(select(model).where(cond).order_by(model.created_at)
                              .with_for_update(skip_locked=True).limit(1)).scalars().first()
        if row is None:
            return None
        row.status = "running"
        row.attempt_token = uuid.uuid4().hex
        row.heartbeat_at = _now()
        row.started_at = row.started_at or _now()
        session.commit()
        return Lease(model, key_attr, getattr(row, key_attr), row.attempt_token), row
    finally:
        session.close()


def claim_note_run():
    # 补数批次由总结任务创建并直接执行，只有租约过期后才会被这里接管。
    return _claim(ContactNoteRun, "run_id")


def claim_summary_job():
    return _claim(ContactSummaryJob, "job_id")


class Lease:
    def __init__(self, model, key_attr: str, key: str, token: str):
        self.model, self.key_attr, self.key, self.token = model, key_attr, key, token
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _write(self, values: dict, require_running: bool = True) -> None:
        col = getattr(self.model, self.key_attr)
        cond = [col == self.key, self.model.attempt_token == self.token]
        if require_running:
            cond.append(self.model.status == "running")
        session = get_session()
        try:
            res = session.execute(update(self.model).where(*cond).values(**values))
            session.commit()
        finally:
            session.close()
        if res.rowcount == 0:
            self.lost.set()
            raise LeaseLost(f"{self.key} 已被其他 worker 接管或已结束")

    def update(self, **values) -> None:
        if self.lost.is_set():
            raise LeaseLost(f"{self.key} 租约已丢失")
        self._write({**values, "heartbeat_at": _now()})

    def finish(self, status: str, **values) -> None:
        self._stop.set()
        self._write({**values, "status": status, "finished_at": _now(), "heartbeat_at": _now()})

    def release(self) -> None:
        """正常停止：放回队列，下一个 worker 立即可以领取。"""
        self._stop.set()
        self._write({"status": "queued", "attempt_token": "", "heartbeat_at": None})

    def keepalive(self) -> "Lease":
        interval = max(1.0, get_settings().contact_job_lease_seconds / 4)

        def loop():
            while not self._stop.wait(interval):
                try:
                    self.update()
                except LeaseLost:
                    return
                except Exception:
                    continue  # 数据库短暂异常时下一轮再续约

        self._thread = threading.Thread(target=loop, name=f"lease-{self.key}", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()


# ===== 覆盖情况 =====

def missing_note_dates(manager_id: str, dates: list[date]) -> list[date]:
    """窗口内需要补数的日期：没有成功批次覆盖，或该经理有失败笔记。"""
    if not dates:
        return []
    session = get_session()
    try:
        covered = set(session.execute(select(ContactNoteRun.stat_date).where(
            ContactNoteRun.stat_date.in_(dates),
            ContactNoteRun.manager_id.in_(["", manager_id]),
            ContactNoteRun.status.in_(DONE_OK))).scalars())
        failed = set(session.execute(select(ContactDailyNote.note_date).where(
            ContactDailyNote.manager_id == manager_id,
            ContactDailyNote.note_date.between(min(dates), max(dates)),
            ContactDailyNote.status == "failed")).scalars())
        return sorted((set(dates) - covered) | failed)
    finally:
        session.close()


def create_backfill_runs(manager_id: str, dates: list[date], token: str) -> list[Lease]:
    """为总结任务创建补数批次并直接置为 running，由该任务内联执行。"""
    session = get_session()
    leases = []
    try:
        for d in dates:
            run = ContactNoteRun(run_id=f"nr_bf_{uuid.uuid4().hex[:12]}", stat_date=d,
                                 manager_id=manager_id, trigger_type="backfill", status="running",
                                 attempt_token=token, heartbeat_at=_now(), started_at=_now(), error="")
            session.add(run)
            leases.append(Lease(ContactNoteRun, "run_id", run.run_id, token))
        session.commit()
        return leases
    finally:
        session.close()
