"""第二层：月度总结（Prompt B）。

输入全部由代码拼装，不调用模型：
- 沟通统计：contact_daily_stats 求和（含寒暄）；
- 信号时间线：已核实、且所属笔记成功的信号事件，按日期汇总最终状态；
- 待跟进 / 每日笔记：status=ok 的日笔记。
input_fp 不变且已有成功结果时直接复用；窗口内所有天都已分析且没有业务内容时套模板，不调用模型；没有可用笔记又有未分析的天时不生成。
模型输出经代码校验（沟通次数、字数、数字来源、PII），失败重试一次，仍失败标记为 review。
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert

from app.config import get_settings
from app.contacts.dbutil import run_in_transaction
from app.contacts.llm import ContactLlm, LlmCallError
from app.contacts.prompts import (SUMMARY_PROMPT_VERSION, build_summary_system, build_summary_user,
                                  signal_name, window_label)
from app.contacts.stats import aggregate_window
from app.contacts.transcript import estimate_tokens, normalize, period_label
from app.db import get_session
from app.models import ContactDailyNote, ContactDailyStat, ContactSignalEvent, ContactSummary
from app.pipeline.pii import mask_pii

_MAX_SUMMARY_CHARS = 320


def window_bounds(window_end: date, days: int) -> tuple[date, date]:
    return window_end - timedelta(days=days - 1), window_end


# ===== 输入拼装（纯函数）=====

def build_signal_timeline(events) -> list[dict]:
    """events: 按 (日期, id) 排序的对象，含 event_date / signal_code / status / evidence。"""
    by_code: dict[str, list] = {}
    for e in events:
        by_code.setdefault(e.signal_code, []).append(e)
    timeline = []
    for code, items in by_code.items():
        open_dates = sorted({e.event_date for e in items if e.status == "open"})
        last = items[-1]
        entry = {"code": code, "name": signal_name(code), "status": last.status,
                 "first_date": items[0].event_date.isoformat(),
                 "last_date": last.event_date.isoformat(), "open_days": len(open_dates)}
        if last.status == "closed":
            entry.update(closed_date=last.event_date.isoformat(), evidence=last.evidence)
            if open_dates:
                entry["first_open_date"] = open_dates[0].isoformat()
        else:
            entry["latest_evidence"] = last.evidence
        timeline.append(entry)
    # 持续中的信号在前，其次按首次出现时间
    timeline.sort(key=lambda x: (x["status"] != "open", x["first_date"]))
    return timeline


def signal_lines(timeline: list[dict]) -> list[str]:
    lines = []
    for s in timeline:
        period = lambda key: period_label(date.fromisoformat(s[key]))
        if s["status"] == "open":
            lines.append(f"- {s['name']}：持续中；{period('first_date')}起出现，"
                         f"共{s['open_days']}天，最近{period('last_date')}")
        elif s.get("first_open_date"):
            lines.append(f"- {s['name']}：已关闭；{period('first_open_date')}出现 → "
                         f"{period('closed_date')}关闭，依据：“{s['evidence']}”")
        else:
            lines.append(f"- {s['name']}：已关闭；{period('closed_date')}关闭，依据：“{s['evidence']}”")
    return lines


def note_line(note, compact: bool = False) -> str:
    parts = [f"{note.note_date:%m-%d}"]
    if note.topics:
        parts.append("话题：" + "、".join(note.topics))
    if not compact:
        for label, key in (("客户", "customer_points"), ("经理", "manager_actions"), ("事实", "facts")):
            values = getattr(note, key)
            if values:
                parts.append(f"{label}：" + "；".join(values))
    return "｜".join(parts)


def fit_note_lines(notes: list, priority_dates: set[date], budget: int) -> list[str]:
    """超出预算时：先把非重点日压缩为只留话题，再从最早的非重点日删起，最后删最早的重点日。"""
    notes = sorted(notes, key=lambda n: n.note_date)
    lines = {n.note_date: note_line(n) for n in notes}
    cost = lambda: sum(estimate_tokens(v) + 1 for v in lines.values())
    if cost() <= budget:
        return list(lines.values())
    for n in notes:
        if n.note_date not in priority_dates:
            lines[n.note_date] = note_line(n, compact=True)
    for group in ([n for n in notes if n.note_date not in priority_dates],
                  [n for n in notes if n.note_date in priority_dates][:-1]):
        for n in group:
            if cost() <= budget:
                break
            lines.pop(n.note_date, None)
    return list(lines.values())


@dataclass
class SummaryInput:
    window_start: date
    window_end: date
    stats: dict
    business_days: int
    incomplete_days: int
    timeline: list[dict]
    pending: list[dict]
    user_prompt: str
    system_prompt: str
    input_fp: str = field(default="")


def build_summary_input(window_start: date, window_end: date, stat_rows, notes, events,
                        window_days: int, notes_budget: int, model: str) -> SummaryInput:
    """stat_rows / notes / events 为该对话对窗口内的记录；events 只含已核实且属于成功笔记的事件。"""
    agg = aggregate_window(stat_rows)
    ok_notes = [n for n in notes if n.status == "ok"]
    noted = {n.note_date for n in notes if n.status in ("ok", "no_business")}
    stat_dates = {r.stat_date for r in stat_rows}
    incomplete = len(stat_dates - noted)
    events = sorted(events, key=lambda e: (e.event_date, e.id))
    timeline = build_signal_timeline(events)
    pending = [{"date": n.note_date.isoformat(), "item": item, "verified": False}
               for n in sorted(ok_notes, key=lambda n: n.note_date) for item in n.pending]
    stats = {"session_count": agg.session_count, "customer_initiated": agg.customer_initiated,
             "active_days": agg.active_days, "business_days": len(ok_notes),
             "msg_count": agg.msg_count,
             "last_contact": agg.last_contact.isoformat() if agg.last_contact else None,
             "last_contact_period": period_label(agg.last_contact) if agg.last_contact else None}
    stats_line = (f"沟通{agg.session_count}次（客户主动发起{agg.customer_initiated}次），"
                  f"涉及{agg.active_days}天（其中{len(ok_notes)}天有业务内容）")
    if agg.last_contact:
        stats_line += f"，最近一次联系为{period_label(agg.last_contact)}"
    priority_dates = {e.event_date for e in events} | {n.note_date for n in ok_notes if n.facts or n.pending}
    user = build_summary_user(
        f"{window_start.isoformat()} ～ {window_end.isoformat()}（{window_label(window_days)}）",
        stats_line, signal_lines(timeline),
        [f"- {period_label(date.fromisoformat(p['date']))}：{p['item']}" for p in pending],
        fit_note_lines(ok_notes, priority_dates, notes_budget))
    system = build_summary_system(window_days)
    fp = hashlib.sha256(json.dumps([SUMMARY_PROMPT_VERSION, model, system, user],
                                   ensure_ascii=False).encode("utf-8")).hexdigest()
    return SummaryInput(window_start, window_end, stats, len(ok_notes), incomplete,
                        timeline, pending, user, system, fp)


def template_summary(inp: SummaryInput, window_days: int) -> str:
    return f"{window_label(window_days)}与客户沟通{inp.stats['session_count']}次，以日常问候为主，暂未涉及业务诉求。"


_NUMBER = re.compile(r"\d+(?:\.\d+)?\s*(?:%|％|万|元|亿)")
_FORBIDDEN = ("系统", "笔记", "持续中", "仍在持续")  # 内部用语不得出现在给经理看的正文里


def validate_summary(data: dict, inp: SummaryInput) -> tuple[str, list[str], list[str]]:
    summary = " ".join(str(data.get("summary") or "").split())
    if not summary:
        raise ValueError("summary 为空")
    n = inp.stats["session_count"]
    if f"沟通{n}次" not in summary:
        raise ValueError(f"沟通次数与统计（{n}次）不一致")
    min_chars = 50 if inp.business_days <= 2 else 100
    if not min_chars <= len(summary) <= _MAX_SUMMARY_CHARS:
        raise ValueError(f"字数 {len(summary)} 不在 {min_chars}～{_MAX_SUMMARY_CHARS} 范围")
    source = normalize(inp.user_prompt)
    unknown = [m for m in _NUMBER.findall(summary) if normalize(m) not in source]
    if unknown:
        raise ValueError(f"出现输入中没有的数字：{'、'.join(unknown)}")
    if mask_pii(summary) != summary:
        raise ValueError("包含疑似个人敏感信息")
    leaked = [w for w in _FORBIDDEN if w in summary]
    if leaked:
        raise ValueError(f"出现内部用语：{'、'.join(leaked)}")
    lists = []
    for key, limit in (("main_topics", 3), ("suggestions", 2)):
        value = data.get(key) or []
        lists.append([str(v).strip() for v in value if str(v).strip()][:limit] if isinstance(value, list) else [])
    return summary, lists[0], lists[1]


# ===== 执行 =====

def load_pair_records(manager_id: str, customer_id: str, start: date, end: date):
    session = get_session()
    try:
        stats = session.execute(select(ContactDailyStat).where(
            ContactDailyStat.manager_id == manager_id, ContactDailyStat.customer_id == customer_id,
            ContactDailyStat.stat_date.between(start, end))).scalars().all()
        notes = session.execute(select(ContactDailyNote).where(
            ContactDailyNote.manager_id == manager_id, ContactDailyNote.customer_id == customer_id,
            ContactDailyNote.note_date.between(start, end))).scalars().all()
        events = session.execute(
            select(ContactSignalEvent)
            .join(ContactDailyNote, ContactDailyNote.id == ContactSignalEvent.note_id)
            .where(ContactSignalEvent.manager_id == manager_id,
                   ContactSignalEvent.customer_id == customer_id,
                   ContactSignalEvent.event_date.between(start, end),
                   ContactSignalEvent.evidence_verified.is_(True),
                   ContactDailyNote.status == "ok")).scalars().all()
        return stats, notes, events
    finally:
        session.close()


def window_customers(manager_id: str, start: date, end: date) -> list[str]:
    session = get_session()
    try:
        return list(session.execute(select(ContactDailyStat.customer_id).distinct().where(
            ContactDailyStat.manager_id == manager_id,
            ContactDailyStat.stat_date.between(start, end)).order_by(ContactDailyStat.customer_id)).scalars())
    finally:
        session.close()


@dataclass
class PairOutcome:
    kind: str            # reused | template | success | review | failed | skipped
    input_tokens: int = 0


def summarize_pair(manager_id: str, customer_id: str, window_end: date, *, job_id: str = "",
                   force: bool = False, llm: ContactLlm | None = None,
                   priority: str = "high") -> PairOutcome:
    settings = get_settings()
    days = settings.contact_window_days
    start, end = window_bounds(window_end, days)
    stat_rows, notes, events = load_pair_records(manager_id, customer_id, start, end)
    if not stat_rows:
        return PairOutcome("skipped")
    inp = build_summary_input(start, end, stat_rows, notes, events, days,
                              settings.contact_summary_notes_tokens, settings.llm_model)
    if inp.business_days == 0 and inp.incomplete_days:
        # 没有任何可用笔记、又有未分析的天，无法判断是否只是寒暄：不生成总结，计为失败。
        return PairOutcome("failed")

    session = get_session()
    try:
        existing = session.execute(select(ContactSummary.input_fp, ContactSummary.status).where(
            ContactSummary.manager_id == manager_id, ContactSummary.customer_id == customer_id,
            ContactSummary.window_end == end)).first()
    finally:
        session.close()
    if not force and existing and existing[0] == inp.input_fp and existing[1] in ("success", "template"):
        return PairOutcome("reused")

    values = dict(window_start=start, stats_json=inp.stats, signals_json=inp.timeline,
                  pending_json=inp.pending, incomplete_days=inp.incomplete_days,
                  input_fp=inp.input_fp, prompt_version=SUMMARY_PROMPT_VERSION,
                  model=settings.llm_model, job_id=job_id, main_topics=[], suggestions=[],
                  llm_call_id=None, validation_error=None)
    tokens = 0
    if inp.business_days == 0:
        values.update(status="template", summary=template_summary(inp, days), model="")
        kind = "template"
    else:
        llm = llm or ContactLlm()
        kind, error = "review", None
        for _ in range(2):
            try:
                res = llm.chat_json(purpose="summary", system=inp.system_prompt, user=inp.user_prompt,
                                    prompt_version=SUMMARY_PROMPT_VERSION, max_tokens=800,
                                    priority=priority, manager_id=manager_id, customer_id=customer_id)
            except LlmCallError:
                # 调用本身失败（网络/限流/鉴权）不是内容问题：不覆盖已有结果，计入任务失败数。
                return PairOutcome("failed", tokens)
            tokens += res.prompt_tokens
            values["llm_call_id"] = res.call_id
            try:
                summary, topics, suggestions = validate_summary(res.data, inp)
            except ValueError as exc:
                error = f"校验失败：{exc}"
                continue
            values.update(status="success", summary=summary, main_topics=topics, suggestions=suggestions)
            kind = "success"
            break
        if kind == "review":
            values.update(status="review", summary=None, validation_error=(error or "")[:500])
    _save_summary(manager_id, customer_id, end, values)
    return PairOutcome(kind, tokens)


def _save_summary(manager_id: str, customer_id: str, window_end: date, values: dict) -> None:
    now = datetime.utcnow()

    def write(session):
        stmt = insert(ContactSummary).values(manager_id=manager_id, customer_id=customer_id,
                                             window_end=window_end, created_at=now, updated_at=now,
                                             **values)
        session.execute(stmt.on_duplicate_key_update(
            {k: stmt.inserted[k] for k in [*values, "updated_at"]}))

    run_in_transaction(write)
