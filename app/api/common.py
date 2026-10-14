"""各路由共用的鉴权与参数校验。"""
from __future__ import annotations

from datetime import datetime

from fastapi import HTTPException

from app.config import get_settings


def check_auth(token: str | None) -> None:
    expected = get_settings().trigger_token
    if expected and token != expected:
        raise HTTPException(status_code=401, detail="无效的触发 token")


def validate_date(d: str) -> str:
    try:
        datetime.strptime(d, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"日期格式应为 YYYY-MM-DD: {d}")
    return d
