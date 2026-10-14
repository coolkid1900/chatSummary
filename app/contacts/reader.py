"""读取经理与客户的双向文本消息。

聊天分表按 from_user 路由：经理发出的消息在经理的分表，客户发出的消息在客户的分表，
同一对话对的消息分散在两张表里，因此按 (经理, 客户, 日期) 在内存中归组。

- 全量（夜间）：按 CRC32(经理工号) % K 分 K 批读取当天消息，每批只占约 1/K 内存。
- 单经理（补数）：经理分表按 from_user（idx_from_user）+ 全部分表按 to_user（idx_to_user）。

按 msg_time 归属日期（BETWEEN 当天 00:00:00 AND 23:59:59，msg_time 为秒精度 DATETIME）。
不按 room_id 过滤；收发双方必须恰好一方是客户。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Iterable

from sqlalchemy import text

from app.config import get_settings
from app.db import get_session
from app.sharding import all_shard_tables, shard_table


@dataclass(frozen=True)
class ChatMsg:
    id: int
    manager_id: str
    customer_id: str
    from_customer: bool
    content: str
    msg_time: datetime


PairDayKey = tuple[str, str, date]  # (manager_id, customer_id, date)


def _prefixes() -> tuple[str, ...]:
    return tuple(p.strip() for p in get_settings().external_id_prefixes.split(",") if p.strip())


def _day_bounds(start: date, end: date) -> tuple[datetime, datetime]:
    return datetime.combine(start, time.min), datetime.combine(end, time(23, 59, 59))


def to_chat_msg(row, prefixes: tuple[str, ...]) -> ChatMsg | None:
    """把分表行转换为对话对消息；内部员工互发、客户互发等不属于经理-客户对话，返回 None。"""
    sender_is_customer = row.from_user.startswith(prefixes)
    receiver_is_customer = row.to_user.startswith(prefixes)
    if sender_is_customer == receiver_is_customer:
        return None
    manager, customer = (row.to_user, row.from_user) if sender_is_customer else (row.from_user, row.to_user)
    return ChatMsg(id=row.id, manager_id=manager, customer_id=customer,
                   from_customer=sender_is_customer, content=row.content or "",
                   msg_time=row.msg_time)


def group_pair_days(messages: Iterable[ChatMsg]) -> dict[PairDayKey, list[ChatMsg]]:
    grouped: dict[PairDayKey, list[ChatMsg]] = defaultdict(list)
    for m in messages:
        grouped[(m.manager_id, m.customer_id, m.msg_time.date())].append(m)
    for msgs in grouped.values():
        msgs.sort(key=lambda m: (m.msg_time, m.id))
    return dict(grouped)


_COLUMNS = "id, from_user, to_user, content, msg_time"


def _like_patterns(prefixes: tuple[str, ...]) -> dict[str, str]:
    escape = lambda p: p.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return {f"p{i}": escape(p) + "%" for i, p in enumerate(prefixes)}


def read_day_partition(day: date, part: int, parts: int) -> dict[PairDayKey, list[ChatMsg]]:
    """读取某天、经理工号哈希落在第 part 批的全部经理-客户消息。"""
    prefixes = _prefixes()
    patterns = _like_patterns(prefixes)
    is_customer = " OR ".join(f"from_user LIKE :{k}" for k in patterns) or "FALSE"
    where = "msg_time BETWEEN :start AND :end AND msg_type = 'text'"
    if parts > 1:
        where += (f" AND CRC32(CASE WHEN ({is_customer}) THEN to_user ELSE from_user END)"
                  f" % :parts = :part")
    start, end = _day_bounds(day, day)
    params = {"start": start, "end": end, "parts": parts, "part": part, **patterns}
    out: list[ChatMsg] = []
    session = get_session()
    try:
        for tbl in all_shard_tables():
            stmt = text(f"SELECT {_COLUMNS} FROM `{tbl}` WHERE {where}")
            for row in session.execute(stmt, params):
                msg = to_chat_msg(row, prefixes)
                if msg is not None:
                    out.append(msg)
    finally:
        session.close()
    return group_pair_days(out)


def read_manager_range(manager_id: str, start: date, end: date) -> dict[PairDayKey, list[ChatMsg]]:
    """读取某经理在 [start, end] 内与所有客户的双向消息。"""
    prefixes = _prefixes()
    lo, hi = _day_bounds(start, end)
    params = {"m": manager_id, "start": lo, "end": hi}
    time_filter = "msg_time BETWEEN :start AND :end AND msg_type = 'text'"
    out: list[ChatMsg] = []
    session = get_session()
    try:
        own = shard_table(manager_id)
        stmt = text(f"SELECT {_COLUMNS} FROM `{own}` WHERE from_user = :m AND {time_filter}")
        rows = list(session.execute(stmt, params))
        for tbl in all_shard_tables():
            stmt = text(f"SELECT {_COLUMNS} FROM `{tbl}` WHERE to_user = :m AND {time_filter}")
            rows.extend(session.execute(stmt, params))
    finally:
        session.close()
    for row in rows:
        msg = to_chat_msg(row, prefixes)
        if msg is not None and msg.manager_id == manager_id:
            out.append(msg)
    return group_pair_days(out)


def dates_between(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]
