"""第一层：日笔记（Prompt A）。

每个对话对每天：
1. 计算沟通统计并写入 contact_daily_stats（含寒暄，不调用模型）。
2. 输入未变（source_fp + prompt 版本相同）且已有成功笔记 → 复用，不调用模型。
3. 清洗后只剩寒暄 → 直接写 no_business，不调用模型。
4. 其余按对话对打包（冷启动时多天合并一次调用）→ Prompt A → 校验 → 笔记与信号事件同一事务写入。

信号证据必须是本次送入原文的子串，否则 evidence_verified=0，只留作审计。
调用失败的天写 status=failed，保留旧内容与旧信号，下次批次自动重试。
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable

from sqlalchemy import delete, select
from sqlalchemy.dialects.mysql import insert

from app.config import get_settings
from app.contacts.llm import ContactLlm, LlmCallError
from app.contacts.prompts import NOTE_PROMPT_VERSION, NOTE_SYSTEM, SIGNALS, build_note_user
from app.contacts.reader import ChatMsg, PairDayKey, read_day_partition, read_manager_range
from app.contacts.stats import DayStats, compute_day_stats
from app.contacts.dbutil import run_in_transaction
from app.contacts.transcript import DayTranscript, clean_day, estimate_tokens, load_chitchat, normalize
from app.db import get_session
from app.models import ContactDailyNote, ContactDailyStat, ContactSignalEvent

log = logging.getLogger(__name__)

_LIST_LIMITS = {"topics": 3, "customer_points": 3, "manager_actions": 2, "facts": 3, "pending": 3}
_ITEM_MAX_CHARS = 80
_EVIDENCE_MAX_CHARS = 60
_SIGNAL_LOOKBACK_DAYS = 90
_PROMPT_OVERHEAD = 300   # user 模板与未关闭信号列表的估算
_CHUNK = 500             # IN 查询与批量写入的分块大小


# ===== 模型输出校验（纯函数）=====

@dataclass
class ParsedSignal:
    code: str
    status: str
    evidence: str
    verified: bool


@dataclass
class ParsedDay:
    topics: list[str] = field(default_factory=list)
    customer_points: list[str] = field(default_factory=list)
    manager_actions: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    signals: list[ParsedSignal] = field(default_factory=list)

    @property
    def has_content(self) -> bool:
        return any(getattr(self, k) for k in _LIST_LIMITS) or bool(self.signals)


def _str_list(value, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    items = [" ".join(str(v).split())[:_ITEM_MAX_CHARS] for v in value if isinstance(v, (str, int, float))]
    return [v for v in items if v][:limit]


def parse_note_output(data: dict, transcripts: dict[date, DayTranscript]) -> dict[date, ParsedDay]:
    """只保留本批次日期、已知信号代码；证据须为该天送入原文的子串才算核实。"""
    days = data.get("days") if isinstance(data, dict) else None
    if not isinstance(days, list):
        raise ValueError("模型输出缺少 days 数组")
    result: dict[date, ParsedDay] = {}
    for item in days:
        if not isinstance(item, dict):
            continue
        try:
            d = date.fromisoformat(str(item.get("date", "")).strip())
        except ValueError:
            continue
        if d not in transcripts or d in result:
            continue
        source = normalize("\n".join(transcripts[d].lines))
        signals = []
        for s in item.get("signals") or []:
            if not isinstance(s, dict):
                continue
            code = str(s.get("code", "")).strip().upper()
            status = str(s.get("status", "")).strip().lower()
            if code not in SIGNALS or status not in ("open", "closed"):
                continue
            evidence = str(s.get("evidence") or "").strip().strip("“”\"'「」 ")[:_EVIDENCE_MAX_CHARS]
            key = normalize(evidence)
            signals.append(ParsedSignal(code, status, evidence, bool(key) and key in source))
        result[d] = ParsedDay(**{k: _str_list(item.get(k), n) for k, n in _LIST_LIMITS.items()},
                              signals=signals)
    return result


def open_signals_from_events(events) -> list[tuple[str, date]]:
    """events 按 (日期, id) 排序：(code, status, event_date)。返回仍未关闭的 (code, 本轮开启日期)。"""
    since: dict[str, date] = {}
    for code, status, day in events:
        if status == "open":
            since.setdefault(code, day)
        else:
            since.pop(code, None)
    return sorted(since.items(), key=lambda x: x[1])


def fit_transcript(tr: DayTranscript, budget: int) -> DayTranscript:
    """单天内容超出单次调用预算时保留前面的发言，并标注已截断。"""
    if estimate_tokens(tr.text) <= budget:
        return tr
    kept, used = [], estimate_tokens(f"[{tr.day.isoformat()}]") + 20
    for line in tr.lines:
        cost = estimate_tokens(line) + 1
        if used + cost > budget:
            break
        kept.append(line)
        used += cost
    return DayTranscript(tr.day, kept + ["（当天记录过长，后续内容已截断）"])


def pack_batches(items: list[tuple[date, DayTranscript, str]], budget: int):
    """按日期顺序把同一对话对的多天打包，每包估算 token 不超过 budget（单天超出时独占一包）。"""
    batch, used = [], 0
    for item in sorted(items, key=lambda x: x[0]):
        cost = estimate_tokens(item[1].text)
        if batch and used + cost > budget:
            yield batch
            batch, used = [], 0
        batch.append(item)
        used += cost
    if batch:
        yield batch


# ===== 执行 =====

@dataclass
class DateCounts:
    pair_count: int = 0
    llm_pair_count: int = 0
    skipped_count: int = 0
    reused_count: int = 0
    failed_count: int = 0
    input_tokens: int = 0


def _chunks(values: list, size: int = _CHUNK):
    for i in range(0, len(values), size):
        yield values[i:i + size]


class NoteProcessor:
    def __init__(self, run_ids: dict[date, str], *, force: bool = False, llm: ContactLlm | None = None,
                 priority: str = "low", stop: threading.Event | None = None,
                 on_progress: Callable[[], None] | None = None):
        self.settings = get_settings()
        self.run_ids = run_ids
        self.force = force
        self.llm = llm or ContactLlm(stop=stop)
        self.priority = priority
        self.stop = stop or threading.Event()
        self.on_progress = on_progress
        self.counts: dict[date, DateCounts] = defaultdict(DateCounts)
        self._lock = threading.Lock()
        self._abort = threading.Event()  # 本批次中止；不影响 worker 的全局 stop
        self._call_budget = (self.settings.contact_max_call_tokens
                             - estimate_tokens(NOTE_SYSTEM) - _PROMPT_OVERHEAD)

    def _count(self, d: date, **delta) -> None:
        with self._lock:
            c = self.counts[d]
            for k, v in delta.items():
                setattr(c, k, getattr(c, k) + v)

    # ---- 主流程 ----

    def process(self, pair_days: dict[PairDayKey, list[ChatMsg]]) -> None:
        pair_days = {k: v for k, v in pair_days.items() if k[2] in self.run_ids and v}
        if not pair_days:
            return
        chitchat = load_chitchat()
        stats = self._save_stats(pair_days)
        existing = self._existing_notes(list(pair_days))
        todo: dict[tuple[str, str], list] = defaultdict(list)
        no_business = []
        for key in sorted(pair_days, key=lambda k: k[2]):
            manager, customer, d = key
            fp = stats[key].source_fp
            self._count(d, pair_count=1)
            old = existing.get(key)
            if (not self.force and old is not None and old[0] in ("ok", "no_business")
                    and old[1] == fp and old[2] == NOTE_PROMPT_VERSION):
                self._count(d, reused_count=1)
                continue
            tr = clean_day(d, pair_days[key], chitchat)
            if not tr.lines:
                no_business.append((key, fp))
                self._count(d, skipped_count=1)
                continue
            todo[(manager, customer)].append((d, fit_transcript(tr, self._call_budget), fp))
        self._save_no_business(no_business)

        batches = [(m, c, b) for (m, c), items in todo.items()
                   for b in pack_batches(items, self.settings.contact_note_batch_tokens)]
        if not batches:
            return
        pool = ThreadPoolExecutor(max_workers=self.settings.contact_llm_concurrency,
                                  thread_name_prefix="contact-note")
        try:
            futures = [pool.submit(self._run_batch, *b) for b in batches]
            for fut in as_completed(futures):
                fut.result()
                if self.on_progress:
                    self.on_progress()
        except BaseException:
            # 租约丢失或停止时不再发起新调用；已发出的调用自然结束。
            self._abort.set()
            pool.shutdown(wait=True, cancel_futures=True)
            raise
        pool.shutdown(wait=True)

    def _run_batch(self, manager: str, customer: str, items) -> None:
        if self.stop.is_set() or self._abort.is_set():
            return
        transcripts = {d: tr for d, tr, _ in items}
        since = self._open_signals(manager, customer, items[0][0])
        user = build_note_user([(code, f"{d:%m-%d}") for code, d in since],
                               "\n\n".join(tr.text for _, tr, _ in items))
        try:
            res = self.llm.chat_json(
                purpose="note", system=NOTE_SYSTEM, user=user, prompt_version=NOTE_PROMPT_VERSION,
                max_tokens=min(4000, 500 + 400 * len(items)), priority=self.priority,
                manager_id=manager, customer_id=customer)
        except InterruptedError:
            return
        except LlmCallError as exc:
            log.warning("日笔记调用失败 %s/%s %s: %s", manager, customer,
                        [d.isoformat() for d, _, _ in items], exc)
            self._save_failed(manager, customer, items, str(exc))
            for d, _, _ in items:
                self._count(d, failed_count=1)
            return
        share, extra = divmod(res.prompt_tokens, len(items))
        for i, (d, _, _) in enumerate(items):
            self._count(d, input_tokens=share + (extra if i == 0 else 0))
        try:
            parsed = parse_note_output(res.data, transcripts)
        except ValueError as exc:
            self._save_failed(manager, customer, items, f"输出校验失败: {exc}")
            for d, _, _ in items:
                self._count(d, failed_count=1)
            return
        self._save_parsed(manager, customer, items, parsed, res.call_id)
        for d, _, _ in items:
            self._count(d, llm_pair_count=1)

    # ---- 读取 ----

    def _open_signals(self, manager: str, customer: str, before: date) -> list[tuple[str, date]]:
        session = get_session()
        try:
            rows = session.execute(
                select(ContactSignalEvent.signal_code, ContactSignalEvent.status,
                       ContactSignalEvent.event_date)
                .join(ContactDailyNote, ContactDailyNote.id == ContactSignalEvent.note_id)
                .where(ContactSignalEvent.manager_id == manager,
                       ContactSignalEvent.customer_id == customer,
                       ContactSignalEvent.event_date < before,
                       ContactSignalEvent.event_date >= before - timedelta(days=_SIGNAL_LOOKBACK_DAYS),
                       ContactSignalEvent.evidence_verified.is_(True),
                       ContactDailyNote.status == "ok")
                .order_by(ContactSignalEvent.event_date, ContactSignalEvent.id)).all()
        finally:
            session.close()
        return open_signals_from_events(rows)

    def _existing_notes(self, keys: list[PairDayKey]) -> dict[PairDayKey, tuple[str, str, str]]:
        managers = sorted({k[0] for k in keys})
        dates = sorted({k[2] for k in keys})
        wanted = set(keys)
        out = {}
        session = get_session()
        try:
            for chunk in _chunks(managers):
                rows = session.execute(select(
                    ContactDailyNote.manager_id, ContactDailyNote.customer_id, ContactDailyNote.note_date,
                    ContactDailyNote.status, ContactDailyNote.source_fp, ContactDailyNote.prompt_version,
                ).where(ContactDailyNote.manager_id.in_(chunk), ContactDailyNote.note_date.in_(dates)))
                for m, c, d, status, fp, version in rows:
                    if (m, c, d) in wanted:
                        out[(m, c, d)] = (status, fp, version)
        finally:
            session.close()
        return out

    # ---- 写入 ----

    def _save_stats(self, pair_days: dict[PairDayKey, list[ChatMsg]]) -> dict[PairDayKey, DayStats]:
        """按日期升序计算；跨零点判断优先用本批次前一天的结果，否则查库。"""
        gap = self.settings.session_gap_minutes
        prev_keys = {(m, c, d - timedelta(days=1)) for m, c, d in pair_days}
        prev_last: dict[PairDayKey, datetime] = {}
        managers = sorted({k[0] for k in pair_days})
        prev_dates = sorted({k[2] for k in prev_keys})
        session = get_session()
        try:
            for chunk in _chunks(managers):
                rows = session.execute(select(
                    ContactDailyStat.manager_id, ContactDailyStat.customer_id,
                    ContactDailyStat.stat_date, ContactDailyStat.last_msg_time,
                ).where(ContactDailyStat.manager_id.in_(chunk), ContactDailyStat.stat_date.in_(prev_dates)))
                for m, c, d, last in rows:
                    if (m, c, d) in prev_keys:
                        prev_last[(m, c, d)] = last
        finally:
            session.close()
        result: dict[PairDayKey, DayStats] = {}
        for key in sorted(pair_days, key=lambda k: k[2]):
            m, c, d = key
            prev_key = (m, c, d - timedelta(days=1))
            last = (pair_days[prev_key][-1].msg_time if prev_key in pair_days
                    else prev_last.get(prev_key))
            result[key] = compute_day_stats(pair_days[key], gap, last)
        rows = [dict(manager_id=m, customer_id=c, stat_date=d, **asdict(s), updated_at=datetime.utcnow())
                for (m, c, d), s in result.items()]
        update_keys = [k for k in rows[0] if k not in ("manager_id", "customer_id", "stat_date")]
        for chunk in _chunks(rows):
            def write(session, chunk=chunk):
                stmt = insert(ContactDailyStat).values(chunk)
                session.execute(stmt.on_duplicate_key_update({k: stmt.inserted[k] for k in update_keys}))
            run_in_transaction(write)
        return result

    def _upsert_note(self, session, manager: str, customer: str, d: date, values: dict,
                     update_keys: list[str]) -> int:
        now = datetime.utcnow()
        base = dict(topics=[], customer_points=[], manager_actions=[], facts=[], pending=[],
                    model=self.settings.llm_model, prompt_version=NOTE_PROMPT_VERSION,
                    llm_call_id=None, error=None, created_at=now, updated_at=now)
        stmt = insert(ContactDailyNote).values(manager_id=manager, customer_id=customer,
                                               note_date=d, **{**base, **values})
        session.execute(stmt.on_duplicate_key_update(
            {k: stmt.inserted[k] for k in [*update_keys, "updated_at"]}))
        return session.execute(select(ContactDailyNote.id).where(
            ContactDailyNote.manager_id == manager, ContactDailyNote.customer_id == customer,
            ContactDailyNote.note_date == d)).scalar_one()

    _CONTENT_KEYS = ["status", "topics", "customer_points", "manager_actions", "facts", "pending",
                     "source_fp", "prompt_version", "model", "run_id", "llm_call_id", "error"]

    def _clear_events(self, session, manager: str, customer: str, d: date) -> None:
        session.execute(delete(ContactSignalEvent).where(
            ContactSignalEvent.manager_id == manager, ContactSignalEvent.customer_id == customer,
            ContactSignalEvent.event_date == d))

    def _save_no_business(self, items) -> None:
        def write(session, chunk):
            for (m, c, d), fp in chunk:
                self._upsert_note(session, m, c, d, dict(status="no_business", source_fp=fp,
                                                          run_id=self.run_ids[d]), self._CONTENT_KEYS)
                self._clear_events(session, m, c, d)

        for chunk in _chunks(items, 100):
            run_in_transaction(lambda session, chunk=chunk: write(session, chunk))

    def _save_parsed(self, manager: str, customer: str, items, parsed: dict[date, ParsedDay],
                     call_id: str) -> None:
        def write(session):
            for d, _, fp in items:
                day = parsed.get(d) or ParsedDay()
                values = dict(status="ok" if day.has_content else "no_business", source_fp=fp,
                              run_id=self.run_ids[d], llm_call_id=call_id,
                              **{k: getattr(day, k) for k in _LIST_LIMITS})
                note_id = self._upsert_note(session, manager, customer, d, values, self._CONTENT_KEYS)
                self._clear_events(session, manager, customer, d)
                session.add_all(ContactSignalEvent(
                    note_id=note_id, manager_id=manager, customer_id=customer, event_date=d,
                    signal_code=s.code, status=s.status, evidence=s.evidence,
                    evidence_verified=s.verified) for s in day.signals)

        run_in_transaction(write)

    def _save_failed(self, manager: str, customer: str, items, error: str) -> None:
        def write(session):
            for d, _, fp in items:
                self._upsert_note(session, manager, customer, d,
                                  dict(status="failed", source_fp=fp, run_id=self.run_ids[d],
                                       error=error[:500]),
                                  ["status", "source_fp", "run_id", "error"])

        run_in_transaction(write)


def run_notes(manager_id: str, run_ids: dict[date, str], *, force: bool = False,
              llm: ContactLlm | None = None, priority: str = "low",
              stop: threading.Event | None = None,
              on_progress: Callable[[dict[date, DateCounts]], None] | None = None) -> dict[date, DateCounts]:
    """执行日笔记：manager_id 为空表示某一天的全量（按经理哈希分批读取）。"""
    stop = stop or threading.Event()
    proc = NoteProcessor(run_ids, force=force, llm=llm, priority=priority, stop=stop)
    if on_progress:
        proc.on_progress = lambda: on_progress(proc.counts)
    if manager_id:
        dates = sorted(run_ids)
        proc.process(read_manager_range(manager_id, dates[0], dates[-1]))
    else:
        if len(run_ids) != 1:
            raise ValueError("全量日笔记一次只处理一天")
        (day,) = run_ids
        parts = get_settings().contact_read_partitions
        for part in range(parts):
            if stop.is_set():
                break
            proc.process(read_day_partition(day, part, parts))
            if on_progress:
                on_progress(proc.counts)
    if stop.is_set():
        raise InterruptedError("日笔记执行被中断")
    return proc.counts
