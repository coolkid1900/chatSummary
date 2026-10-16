"""把对话对消息清洗成送给 Prompt A 的文本，并提供 token 估算与旬的表述。

清洗：去寒暄（数据库寒暄词）、脱敏、单条截断、连续同一人发言合并成一行。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date

from app.config import get_settings
from app.contacts.reader import ChatMsg
from app.lexicon import load_words
from app.pipeline.pii import mask_pii


def load_chitchat() -> frozenset[str]:
    """每个批次加载一次，常驻 worker 无需重启即可使用最新寒暄词。"""
    return frozenset(load_words("chitchat"))


@dataclass
class DayTranscript:
    day: date
    lines: list[str]   # "HH:MM 客：..."，空列表表示清洗后无内容

    @property
    def text(self) -> str:
        return f"[{self.day.isoformat()}]\n" + "\n".join(self.lines)


def clean_day(day: date, msgs: list[ChatMsg], chitchat: frozenset[str]) -> DayTranscript:
    settings = get_settings()
    limit = settings.contact_line_max_chars
    lines: list[str] = []
    last_speaker = None
    for m in msgs:
        body = " ".join(m.content.split())
        if not body or body in chitchat:
            continue
        body = mask_pii(body)
        if len(body) > limit:
            body = body[:limit] + "…"
        speaker = "客" if m.from_customer else "经"
        if speaker == last_speaker:
            lines[-1] += f" / {body}"
        else:
            lines.append(f"{m.msg_time:%H:%M} {speaker}：{body}")
            last_speaker = speaker
    return DayTranscript(day, lines)


_CJK = re.compile(r"[　-〿㐀-鿿＀-￯]")


def estimate_tokens(text: str) -> int:
    """按字符类别估算输入 token；系数需用 llm_calls 中实际 prompt_tokens 校准。"""
    settings = get_settings()
    cjk = len(_CJK.findall(text))
    return math.ceil(cjk * settings.contact_tokens_per_cjk
                     + (len(text) - cjk) * settings.contact_tokens_per_other)


def normalize(text: str) -> str:
    """证据比对用：去掉空白，避免换行/空格差异导致误判。"""
    return re.sub(r"\s+", "", text)


_EVIDENCE_IGNORED = re.compile(r"[\s\[\]【】（）()“”\"'‘’「」《》<>，。、；：！？,.;:!?…·~～-]")


def evidence_key(text: str) -> str:
    """证据比对用：忽略空白、标点和括号（如脱敏占位“[他行]”被摘成“他行”），文字仍须逐字一致。"""
    return _EVIDENCE_IGNORED.sub("", text)


def period_label(d: date) -> str:
    part = "上旬" if d.day <= 10 else "中旬" if d.day <= 20 else "下旬"
    return f"{d.month}月{part}"
