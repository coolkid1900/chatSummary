"""步骤 1：从 MySQL 分片读取当日消息（§5.1，IO 型，可分片并行）。

本地 MVP 单进程顺序分片读取；K8s 下可按 id 区间分给多 worker。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

from sqlalchemy import select, text

from app.db import get_engine, get_session
from app.models import Message

# 取数查询的复合索引：WHERE role,msg_type,msg_time范围 + ORDER BY sender,receiver,msg_time,id。
# 没有它，当天百万行会走全量 filesort（DB 端临时排序）。等值列(role,msg_type)在前，
# 其后 sender,receiver,msg_time,id 与 ORDER BY 完全对齐 → 索引序扫描，免 filesort。
# 注：messages 多日累积时建议再按 msg_time 做日期分区，让日期过滤先做分区裁剪。
_INGEST_INDEX = "idx_ingest_order"
_index_checked = False


def ensure_indexes() -> None:
    """幂等创建取数复合索引（MySQL 8 不支持 ADD INDEX IF NOT EXISTS）。"""
    global _index_checked
    if _index_checked:
        return
    engine = get_engine()
    with engine.begin() as conn:
        exists = conn.execute(
            text(
                "SELECT COUNT(*) FROM information_schema.statistics "
                "WHERE table_schema = DATABASE() AND table_name = 'messages' "
                "AND index_name = :idx"
            ),
            {"idx": _INGEST_INDEX},
        ).scalar()
        if not exists:
            conn.execute(
                text(
                    f"ALTER TABLE messages ADD INDEX {_INGEST_INDEX} "
                    "(role, msg_type, sender, receiver, msg_time, id)"
                )
            )
    _index_checked = True


@dataclass
class RawMessage:
    id: int
    sender: str
    receiver: str
    role: str
    msg_type: str
    content: str
    msg_time: datetime


def iter_messages(date_str: str, shard_size: int = 5000) -> Iterator[RawMessage]:
    """按主键分片流式读取当日消息，内存只占一个分片。"""
    start = f"{date_str} 00:00:00"
    end = f"{date_str} 23:59:59"
    session = get_session()
    try:
        last_id = 0
        while True:
            rows = session.execute(
                select(Message)
                .where(
                    Message.msg_time >= start,
                    Message.msg_time <= end,
                    Message.id > last_id,
                )
                .order_by(Message.id)
                .limit(shard_size)
            ).scalars().all()
            if not rows:
                break
            for m in rows:
                yield RawMessage(
                    id=m.id,
                    sender=m.sender,
                    receiver=m.receiver,
                    role=m.role,
                    msg_type=m.msg_type,
                    content=m.content,
                    msg_time=m.msg_time,
                )
            last_id = rows[-1].id
    finally:
        session.close()


def iter_customer_messages(date_str: str, yield_per: int = 2000) -> Iterator[RawMessage]:
    """流式读取当日**客户侧文本**消息，按 (客户, 客户经理, 时间) 排序。

    用 MySQL 服务端游标（stream_results + yield_per）拉取，进程内存只占 yield_per
    行，避免把百万级消息一次性读进内存。排序保证同一 (客户,经理) 对的消息连续，
    供下游会话聚合按边界即时切分（§5.2 / 技术优化1）。
    """
    ensure_indexes()
    start = f"{date_str} 00:00:00"
    end = f"{date_str} 23:59:59"
    stmt = (
        select(Message)
        .where(
            Message.msg_time >= start,
            Message.msg_time <= end,
            Message.role == "customer",
            Message.msg_type == "text",
        )
        .order_by(Message.sender, Message.receiver, Message.msg_time, Message.id)
        .execution_options(yield_per=yield_per)  # 触发服务端流式游标
    )
    session = get_session()
    try:
        for m in session.execute(stmt).scalars():
            yield RawMessage(
                id=m.id,
                sender=m.sender,
                receiver=m.receiver,
                role=m.role,
                msg_type=m.msg_type,
                content=m.content,
                msg_time=m.msg_time,
            )
    finally:
        session.close()
