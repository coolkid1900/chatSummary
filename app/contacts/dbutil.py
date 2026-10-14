"""写事务工具：READ COMMITTED + 死锁重试。

并发批次会写同一经理下不同客户的相邻索引范围；REPEATABLE READ 下“先删后插”的空范围
删除也会加间隙锁，导致死锁（1213）。写事务改用 READ COMMITTED 关闭这类间隙锁，
并对死锁 / 锁等待超时整体重试（各写入均为幂等的 upsert + 先删后插）。
"""
from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

from sqlalchemy.exc import OperationalError

from app.db import get_session

T = TypeVar("T")

_RETRYABLE = (1213, 1205)  # 死锁、锁等待超时


def run_in_transaction(fn: Callable[..., T], attempts: int = 5) -> T:
    for attempt in range(1, attempts + 1):
        session = get_session()
        try:
            session.connection(execution_options={"isolation_level": "READ COMMITTED"})
            result = fn(session)
            session.commit()
            return result
        except OperationalError as exc:
            session.rollback()
            code = exc.orig.args[0] if getattr(exc.orig, "args", None) else None
            if code not in _RETRYABLE or attempt == attempts:
                raise
            time.sleep(random.uniform(0.05, 0.2) * attempt)
        finally:
            session.close()
    raise AssertionError("unreachable")
