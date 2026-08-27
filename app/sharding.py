"""聊天记录分表路由（单一真源）。

聊天记录按 `from_user` 分 20 张表 user_chat_record_sharding_1..20，
路由 = Java `String.hashCode()`，与业务方 Java 系统保持一致。

- java_hashcode：精确复刻 Java String.hashCode（31 进制 + 32 位有符号溢出）。
- shard_index：`((h % N) + N) % N + 1` → 稳定落在 1..N。等价于业务方的
  `hashcode % 20 + 1`，并安全处理负 hashCode（标准做法）。写入(seed)与读取(ingest)
  共用本函数，保证同一 from_user 永远落同一张表。
"""
from __future__ import annotations

from app.config import get_settings

# 分表(user_chat_record_sharding_1..N)由业务方/DBA 统一建表，应用不负责建表。
# 本地开发的建表 DDL 见 sql/shard_tables.sql（docker MySQL 初始化时执行）。

_MASK32 = 0xFFFFFFFF


def java_hashcode(s: str) -> int:
    """复刻 Java String.hashCode()：h = 31*h + char，32 位有符号溢出。

    注：Java 按 UTF-16 code unit 迭代；对 BMP 内字符（含中文）ord(ch) 与其一致，
    账号一般为 ASCII，无代理对问题。
    """
    h = 0
    for ch in s:
        h = (31 * h + ord(ch)) & _MASK32
    if h >= 0x80000000:  # 转回 32 位有符号
        h -= 0x100000000
    return h


def shard_index(from_user: str) -> int:
    """返回 1..SHARD_COUNT 的分表序号。"""
    n = get_settings().shard_count
    h = java_hashcode(from_user)
    return (h % n + n) % n + 1


def shard_table(from_user: str) -> str:
    return f"{get_settings().shard_table_prefix}{shard_index(from_user)}"


def all_shard_tables() -> list[str]:
    prefix = get_settings().shard_table_prefix
    return [f"{prefix}{i}" for i in range(1, get_settings().shard_count + 1)]
