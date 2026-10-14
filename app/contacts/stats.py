"""对话对每日沟通统计：纯代码计算，包含寒暄。

会话按 SESSION_GAP_MINUTES 切分，与热点流水线口径一致。跨零点的会话归入开始那天：
若当天首条消息距前一天最后一条不超过间隔，视为前一天会话的延续，不重复计数。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from app.contacts.reader import ChatMsg


@dataclass
class DayStats:
    msg_count: int
    customer_msg_count: int
    manager_msg_count: int
    session_count: int
    customer_initiated: int
    first_msg_time: datetime
    last_msg_time: datetime
    source_fp: str


def source_fingerprint(msgs: list[ChatMsg]) -> str:
    payload = [[m.id, m.from_customer, m.msg_time.isoformat(), m.content] for m in msgs]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


def compute_day_stats(msgs: list[ChatMsg], gap_minutes: int,
                      prev_last_time: datetime | None = None) -> DayStats:
    """msgs 须为同一对话对同一天、按时间排序的非空列表。"""
    gap = timedelta(minutes=gap_minutes)
    sessions = initiated = 0
    last = prev_last_time
    for m in msgs:
        if last is None or m.msg_time - last > gap:
            sessions += 1
            initiated += m.from_customer
        last = m.msg_time
    customer = sum(m.from_customer for m in msgs)
    return DayStats(
        msg_count=len(msgs),
        customer_msg_count=customer,
        manager_msg_count=len(msgs) - customer,
        session_count=sessions,
        customer_initiated=initiated,
        first_msg_time=msgs[0].msg_time,
        last_msg_time=msgs[-1].msg_time,
        source_fp=source_fingerprint(msgs),
    )


@dataclass
class WindowStats:
    session_count: int
    customer_initiated: int
    active_days: int
    msg_count: int
    last_contact: date | None


def aggregate_window(rows) -> WindowStats:
    """rows: ContactDailyStat 行（或同字段对象）。"""
    rows = list(rows)
    last = max((r.last_msg_time for r in rows), default=None)
    return WindowStats(
        session_count=sum(r.session_count for r in rows),
        customer_initiated=sum(r.customer_initiated for r in rows),
        active_days=len(rows),
        msg_count=sum(r.msg_count for r in rows),
        last_contact=last.date() if last else None,
    )
