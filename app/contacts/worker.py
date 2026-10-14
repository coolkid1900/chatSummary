"""接触总结 worker：两个线程分别消费日笔记批次与按需总结任务。

- 日笔记（low 优先级）：夜间全量或手动补数；可按配置每天自动创建前一天的夜间批次。
- 总结任务（high 优先级）：先补齐窗口内缺失的日笔记，再逐个客户生成月度总结。
两类任务共用 Redis 滑动窗口限流；high 等待期间 low 不发起新调用。
收到 SIGTERM/SIGINT 后停止发起新调用，已发出的调用结束后把任务放回队列。
可部署多个副本：租约过期的任务会被其他副本接管，结果按唯一键 upsert。
"""
from __future__ import annotations

import logging
import signal
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime, timedelta

from app.config import get_settings
from app.contacts import jobs
from app.contacts.jobs import Lease, LeaseLost
from app.contacts.llm import ContactLlm
from app.contacts.notes import DateCounts, run_notes
from app.contacts.reader import dates_between
from app.contacts.summary import summarize_pair, window_bounds, window_customers
from app.db import get_engine
from app.models import Base
from app.obs import setup_logging

log = logging.getLogger(__name__)


def run_status(failed: int, total: int) -> str:
    if failed == 0:
        return "success"
    return "failed" if failed >= total else "partial_success"


def _end_on_error(lease: Lease, exc: BaseException) -> None:
    """中断 → 放回队列；租约已丢失 → 不再写；其他异常 → 标记失败。"""
    try:
        if isinstance(exc, InterruptedError):
            lease.release()
        elif not isinstance(exc, LeaseLost):
            lease.finish("failed", error=f"{type(exc).__name__}: {exc}"[:2000])
    except LeaseLost:
        pass
    finally:
        lease.close()


def execute_note_run(lease: Lease, row, stop: threading.Event) -> None:
    lease.keepalive()
    try:
        counts = run_notes(row.manager_id, {row.stat_date: row.run_id}, force=row.force,
                           priority="low", stop=stop,
                           on_progress=lambda c: lease.update(**asdict(c[row.stat_date])))
        c = counts.get(row.stat_date) or DateCounts()
        lease.finish(run_status(c.failed_count, c.pair_count), **asdict(c))
        log.info("日笔记批次完成 run_id=%s %s", row.run_id, asdict(c))
    except BaseException as exc:
        if not isinstance(exc, (InterruptedError, LeaseLost)):
            log.exception("日笔记批次失败 run_id=%s", row.run_id)
        _end_on_error(lease, exc)
        if not isinstance(exc, Exception):
            raise
    finally:
        lease.close()


def _backfill(job, lease: Lease, llm: ContactLlm, stop: threading.Event, dates: list[date]) -> int:
    run_leases = [l.keepalive() for l in jobs.create_backfill_runs(job.manager_id, dates, lease.token)]
    by_date = dict(zip(dates, run_leases))

    def progress(counts):
        lease.update()
        for d, l in by_date.items():
            l.update(**asdict(counts[d]))

    try:
        counts = run_notes(job.manager_id, {d: l.key for d, l in by_date.items()}, llm=llm,
                           priority="high", stop=stop, on_progress=progress)
    except BaseException as exc:
        for l in run_leases:
            _end_on_error(l, exc)
        raise
    for d, l in by_date.items():
        c = counts.get(d) or DateCounts()
        l.finish(run_status(c.failed_count, c.pair_count), **asdict(c))
        l.close()
    return sum(c.input_tokens for c in counts.values())


def execute_summary_job(lease: Lease, job, stop: threading.Event) -> None:
    settings = get_settings()
    lease.keepalive()
    llm = ContactLlm(stop=stop)
    start, end = window_bounds(job.window_end, settings.contact_window_days)
    try:
        lease.update(stage="backfill_notes")
        missing = jobs.missing_note_dates(job.manager_id, dates_between(start, end))
        tokens = _backfill(job, lease, llm, stop, missing) if missing else 0

        customers = window_customers(job.manager_id, start, end)
        if job.customer_ids:
            wanted = set(job.customer_ids)
            customers = [c for c in customers if c in wanted]
        lease.update(stage="summarizing", total_customers=len(customers), input_tokens=tokens)

        kinds: Counter = Counter()
        pool = ThreadPoolExecutor(max_workers=settings.contact_llm_concurrency,
                                  thread_name_prefix="contact-summary")
        try:
            futures = [pool.submit(summarize_pair, job.manager_id, c, end, job_id=job.job_id,
                                   force=job.force, llm=llm) for c in customers]
            for fut in as_completed(futures):
                outcome = fut.result()
                kinds[outcome.kind] += 1
                tokens += outcome.input_tokens
                lease.update(done_customers=sum(kinds.values()), reused_count=kinds["reused"],
                             template_count=kinds["template"], review_count=kinds["review"],
                             failed_count=kinds["failed"], input_tokens=tokens)
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        pool.shutdown(wait=True)
        lease.finish(run_status(kinds["failed"], len(customers)), stage="done")
        log.info("总结任务完成 job_id=%s customers=%d %s", job.job_id, len(customers), dict(kinds))
    except BaseException as exc:
        if not isinstance(exc, (InterruptedError, LeaseLost)):
            log.exception("总结任务失败 job_id=%s", job.job_id)
        _end_on_error(lease, exc)
        if not isinstance(exc, Exception):
            raise
    finally:
        lease.close()


def note_loop(stop: threading.Event) -> None:
    settings = get_settings()
    scheduled: date | None = None
    while not stop.is_set():
        try:
            target = date.today() - timedelta(days=1)
            if (settings.contact_nightly_enabled and scheduled != target
                    and datetime.now().hour >= settings.contact_nightly_hour):
                if jobs.ensure_nightly_run(target):
                    log.info("已创建 %s 的夜间日笔记批次", target)
                scheduled = target
            claimed = jobs.claim_note_run()
            if claimed:
                execute_note_run(*claimed, stop)
                continue
        except Exception:
            log.exception("日笔记循环异常，稍后重试")
        stop.wait(settings.contact_worker_poll_seconds)


def summary_loop(stop: threading.Event) -> None:
    settings = get_settings()
    while not stop.is_set():
        try:
            claimed = jobs.claim_summary_job()
            if claimed:
                execute_summary_job(*claimed, stop)
                continue
        except Exception:
            log.exception("总结任务循环异常，稍后重试")
        stop.wait(settings.contact_worker_poll_seconds)


def main() -> None:
    setup_logging()
    Base.metadata.create_all(get_engine())
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    threads = [threading.Thread(target=note_loop, args=(stop,), name="contact-notes"),
               threading.Thread(target=summary_loop, args=(stop,), name="contact-summaries")]
    for t in threads:
        t.start()
    log.info("接触总结 worker 已启动")
    while any(t.is_alive() for t in threads):
        for t in threads:
            t.join(timeout=1)
    log.info("接触总结 worker 已停止")


if __name__ == "__main__":
    main()
