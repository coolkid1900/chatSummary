"""PII 脱敏（业务合规）。

银行聊天必然含个人敏感信息（手机号、银行卡号、身份证、邮箱等）。在预处理的
**唯一入口**对每条消息内容掩码，使下游 embedding、parquet 落盘、representative_docs
展示、送 DeepSeek 的 prompt 全部使用脱敏后文本。

注意：金额/利率/期限等是业务热点本身，**保留不脱敏**。
"""
from __future__ import annotations

import re

# 身份证（18 位，末位可能 X）放在卡号之前匹配，避免被卡号规则吃掉
_ID_CARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
# 银行卡号 16~19 位连续数字
_BANK_CARD = re.compile(r"(?<!\d)\d{16,19}(?!\d)")
# 中国大陆手机号
_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
# 邮箱
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def mask_pii(text: str) -> str:
    """对文本中的 PII 掩码。顺序敏感：先身份证/邮箱，再卡号，最后手机号。"""
    if not text:
        return text
    text = _ID_CARD.sub("[身份证]", text)
    text = _EMAIL.sub("[邮箱]", text)
    text = _BANK_CARD.sub("[卡号]", text)
    text = _PHONE.sub("[手机号]", text)
    return text
