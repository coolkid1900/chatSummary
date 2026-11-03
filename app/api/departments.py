"""部门热点接口：全部为 POST，参数放在 body。

- /departments/hotspot/run   ：对已完成全公司统计的日期，按部门名单重新切片统计（不重新聚类）。
- /departments/hotspot/query ：查询某部门某天的榜单。
- /mock/departments/managers ：本地联调用的名单接口（DEPARTMENT_MOCK_ENABLED=true 才可用）。
运行没有任务表，进度和结果看 API 日志（关键字“部门热点”）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import date as date_cls, timedelta
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException
from pydantic import AfterValidator, BaseModel, Field

from app.api.common import acquire_run_lock, check_auth, release_run_lock, validate_date
from app.api.contacts import _BadRequestRoute
from app.config import get_settings
from app.departments.hotspot import DepartmentDataMissing, check_ready, run_departments
from app.pipeline.persist import load_topics

log = logging.getLogger(__name__)


def _check_ids(ids: list[str]) -> list[str]:
    ids = list(dict.fromkeys(i.strip() for i in ids if i.strip()))
    limit = get_settings().department_max_per_request
    if len(ids) > limit:
        raise ValueError(f"一次最多 {limit} 个部门")
    return ids


DepartmentId = Annotated[str, Field(min_length=1, max_length=64)]
DepartmentIds = Annotated[list[DepartmentId], AfterValidator(_check_ids)]

router = APIRouter(prefix="/departments", tags=["departments"], route_class=_BadRequestRoute)
mock_router = APIRouter(prefix="/mock/departments", tags=["mock"], route_class=_BadRequestRoute)

# 保留强引用，确保 create_task 创建的任务在完成前不会被回收。
_tasks: set[asyncio.Task[None]] = set()


class HotspotRunReq(BaseModel):
    date: str | None = None  # 默认上一天
    department_ids: DepartmentIds


class HotspotQueryReq(BaseModel):
    date: str
    department_id: DepartmentId


class RosterPageReq(BaseModel):
    department_id: DepartmentId
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=500, ge=1, le=5000)


async def _run_in_background(date_str: str, department_ids: list[str], run_id: str) -> None:
    try:
        await asyncio.to_thread(run_departments, date_str, department_ids, run_id)
    except Exception:
        log.exception("部门热点后台任务失败 run_id=%s", run_id)
    finally:
        release_run_lock(date_str, run_id)


@router.post("/hotspot/run")
async def run_hotspot(req: HotspotRunReq, x_trigger_token: str | None = Header(default=None)):
    """按部门重新统计某天热点；当天须已完成全公司统计（聚类产物随向量保留 VECTOR_TTL_DAYS 天）。"""
    check_auth(x_trigger_token)
    target = validate_date(req.date) if req.date else (date_cls.today() - timedelta(days=1)).isoformat()
    if not req.department_ids:
        raise HTTPException(400, "department_ids 不能为空")
    try:
        await asyncio.to_thread(check_ready, target)
    except DepartmentDataMissing as exc:
        raise HTTPException(409, str(exc))
    run_id = f"dh_{uuid.uuid4().hex[:10]}"
    if not acquire_run_lock(target, run_id):
        raise HTTPException(409, f"{target} 已有正在运行的热点任务")
    task = asyncio.create_task(_run_in_background(target, req.department_ids, run_id),
                               name=f"departments-{run_id}")
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return {"run_id": run_id, "date": target, "department_ids": req.department_ids, "status": "started"}


@router.post("/hotspot/query")
def query_hotspot(req: HotspotQueryReq):
    validate_date(req.date)
    topics = load_topics(req.date, ("department", req.department_id))
    if not topics:
        raise HTTPException(404, f"无部门 {req.department_id} 在 {req.date} 的热点结果")
    return {"date": req.date, "department_id": req.department_id, "count": len(topics), "topics": topics}


@mock_router.post("/managers")
def mock_managers(req: RosterPageReq):
    """模拟名单接口：mock 文件为 {部门 id: [经理 id, ...]}，父部门的列表已包含子部门成员。"""
    settings = get_settings()
    if not settings.department_mock_enabled:
        raise HTTPException(404, "Not Found")
    path = Path(settings.department_mock_file)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HTTPException(500, f"mock 名单文件不存在：{path}")
    ids = [str(i) for i in data.get(req.department_id, [])]
    start = (req.page - 1) * req.page_size
    return {"department_id": req.department_id, "page": req.page, "page_size": req.page_size,
            "total": len(ids), "manager_ids": ids[start:start + req.page_size]}
