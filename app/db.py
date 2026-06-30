"""MySQL 连接（SQLAlchemy 2.0 + PyMySQL）。"""
from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    # pool_pre_ping 避免连接被 MySQL 回收后报错；pool_recycle 防止 wait_timeout。
    return create_engine(
        settings.mysql_url,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=10,
        max_overflow=5,
        future=True,
    )


@lru_cache
def _session_factory() -> sessionmaker:
    return sessionmaker(bind=get_engine(), class_=Session, expire_on_commit=False)


def get_session() -> Session:
    return _session_factory()()
