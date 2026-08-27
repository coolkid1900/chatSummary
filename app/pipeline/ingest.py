"""步骤 1：从分表读取当日聊天（§5.1，IO 型）。

聊天记录按 from_user 分 20 张表 user_chat_record_sharding_1..20（见 app/sharding.py）。
**关键**：分表键 = from_user，而我们按客户(from_user)聚合会话，所以同一客户的消息
必在同一张分表 → 可**逐分表独立流式读取与聚合，无需跨分表全局排序**（也是天然的
并行/多 pod 工作单元）。

角色判定：新表无 role 列，按企业微信账号格式区分——外部联系人(微信客户)的
external_userid 以 wm/wo 开头(EXTERNAL_ID_PREFIXES)，企业成员(客户经理)是 userid(工号)。
只有客户侧文本消息进入热点分析。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterator

from sqlalchemy import text

from app.config import get_settings
from app.db import get_session
from app.sharding import all_shard_tables


@dataclass
class RawMessage:
    id: int
    sender: str        # from_user（发送方）
    receiver: str      # to_user（接收方）
    role: str          # customer | staff（由 from_user 前缀推导）
    msg_type: str
    content: str
    msg_time: datetime


def _customer_prefixes() -> tuple[str, ...]:
    return tuple(
        p.strip() for p in get_settings().external_id_prefixes.split(",") if p.strip()
    )


def _is_customer(from_user: str) -> bool:
    # 外部联系人(微信客户) external_userid 以 wm/wo 开头；企业成员(工号)则否
    return from_user.startswith(_customer_prefixes())


def iter_customer_messages(date_str: str, yield_per: int = 2000) -> Iterator[RawMessage]:
    """流式读取当日**客户侧文本**消息。

    逐分表用服务端游标（stream_results + yield_per）拉取，每张表按
    (from_user, to_user, msg_time, id) 排序，保证同一 (客户,经理) 对消息连续，
    供下游会话聚合按边界即时切分。跨分表顺序无关（同一客户不跨表）。
    """
    for tbl in all_shard_tables():
        stmt = text(
            f"SELECT id, from_user, to_user, msg_type, content, msg_time "
            f"FROM `{tbl}` "
            f"WHERE create_date = :d AND msg_type = 'text' "
            f"ORDER BY from_user, to_user, msg_time, id"
        ).execution_options(yield_per=yield_per, stream_results=True)
        session = get_session()
        try:
            for r in session.execute(stmt, {"d": date_str}):
                if not _is_customer(r.from_user):  # 只要客户侧
                    continue
                yield RawMessage(
                    id=r.id,
                    sender=r.from_user,
                    receiver=r.to_user,
                    role="customer",
                    msg_type=r.msg_type,
                    content=r.content,
                    msg_time=r.msg_time,
                )
        finally:
            session.close()
