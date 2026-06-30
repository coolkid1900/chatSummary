"""步骤 2：预处理 —— 规则过滤 + 会话聚合 + PII 脱敏（§5.2 / §8 / 合规）。

减量措施（乘法叠加，§8）：
  1. 规则预过滤：丢弃非文本/系统消息、纯寒暄、过短消息。
  2. 会话聚合：同一(客户经理,客户)对在时间窗内的**客户侧**消息拼成一段，只嵌一次。
  3. PII 脱敏：聚合时对每条内容掩码（手机号/卡号/身份证/邮箱），下游全链路用脱敏文本。

流式聚合（技术优化1）：输入需按 (客户, 客户经理, 时间) 有序（见 ingest.iter_customer_messages），
本模块在分组/时间窗边界即时 flush，进程内存只占「当前一个会话」，支持百万级。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import timedelta
from functools import lru_cache
from typing import Iterable, Iterator

from app.config import get_settings
from app.lexicon import load_words
from app.pipeline.ingest import RawMessage
from app.pipeline.pii import mask_pii


@lru_cache
def _load_chitchat() -> frozenset[str]:
    """从数据库加载寒暄词（首次自动从种子文件播种）。"""
    return frozenset(load_words("chitchat"))


@dataclass
class SessionUnit:
    session_id: str          # f"{staff}:{customer}:{seq}"
    staff: str
    customer: str
    text: str                # 聚合 + 脱敏后的客户侧文本
    msg_count: int           # 该会话客户侧消息条数（用于热度）
    text_hash: str = field(default="")

    def __post_init__(self):
        if not self.text_hash:
            self.text_hash = hashlib.md5(self.text.encode("utf-8")).hexdigest()


def _keep(msg: RawMessage, min_len: int) -> bool:
    if msg.msg_type != "text":
        return False
    body = msg.content.strip()
    if len(body) < min_len:
        return False
    if body in _load_chitchat():
        return False
    return True


def _flush(staff: str, customer: str, seq: int, texts: list[str]) -> SessionUnit | None:
    if not texts:
        return None
    joined = mask_pii(" ".join(texts))  # PII 脱敏（合规）
    return SessionUnit(
        session_id=f"{staff}:{customer}:{seq}",
        staff=staff,
        customer=customer,
        text=joined,
        msg_count=len(texts),
    )


def stream_sessions(messages: Iterable[RawMessage]) -> Iterator[SessionUnit]:
    """流式会话聚合：输入须按 (客户, 客户经理, 时间) 有序。

    在 (客户经理,客户) 分组变化或时间窗 > SESSION_GAP_MINUTES 时 flush 当前会话。
    内存只保留当前会话缓冲。
    """
    settings = get_settings()
    gap = timedelta(minutes=settings.session_gap_minutes)
    min_len = settings.min_text_len

    cur_key: tuple[str, str] | None = None  # (staff, customer)
    seq = 0
    cur_texts: list[str] = []
    last_time = None

    for m in messages:
        if m.role != "customer" or not _keep(m, min_len):
            continue
        key = (m.receiver, m.sender)  # (staff, customer)

        new_group = key != cur_key
        gapped = (
            not new_group and last_time is not None and m.msg_time - last_time > gap
        )
        if new_group or gapped:
            # flush 上一个会话，必须用「上一个会话的 seq」（此时尚未重置），
            # 否则换组时把上一组最后一个会话错标成 seq=0，与该组首个会话 id 冲突。
            unit = _flush(*(cur_key or ("", "")), seq, cur_texts)
            if unit:
                yield unit
            if new_group:
                cur_key = key
                seq = 0
            else:
                seq += 1
            cur_texts = []

        cur_texts.append(m.content.strip())
        last_time = m.msg_time

    if cur_texts and cur_key is not None:
        unit = _flush(cur_key[0], cur_key[1], seq, cur_texts)
        if unit:
            yield unit


def aggregate_sessions(messages: Iterable[RawMessage]) -> list[SessionUnit]:
    """非流式版本（保留供测试/小批量）：先按 (staff,customer) 排序再走流式聚合。"""
    ordered = sorted(
        (m for m in messages if m.role == "customer" and m.msg_type == "text"),
        key=lambda x: (x.receiver, x.sender, x.msg_time, x.id),
    )
    return list(stream_sessions(ordered))
