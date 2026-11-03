"""部门 → 客户经理名单：每次统计实时调用名单接口，不落库。

接口约定（DEPARTMENT_ROSTER_URL）：
  请求  POST JSON {"department_id": "D1003", "page": 1, "page_size": 500}，page 从 1 开始；
  响应  {"total": 48, "manager_ids": ["13999315", ...]}。
返回该部门及其全部子部门的客户经理，本服务不展开部门树。对接真实接口时如字段不同，改 _parse_page。
"""
from __future__ import annotations

import logging

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.config import get_settings

log = logging.getLogger(__name__)


class RosterError(RuntimeError):
    pass


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return isinstance(exc, httpx.TransportError)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8),
       retry=retry_if_exception(_retryable), reraise=True)
def _fetch_page(client: httpx.Client, url: str, department_id: str, page: int, page_size: int) -> dict:
    resp = client.post(url, json={"department_id": department_id, "page": page, "page_size": page_size})
    resp.raise_for_status()
    return resp.json()


def _parse_page(data: dict) -> tuple[int, list[str]]:
    total, ids = data.get("total"), data.get("manager_ids")
    if not isinstance(total, int) or not isinstance(ids, list):
        raise RosterError("名单接口返回格式不符，应为 {total: int, manager_ids: list}")
    return total, [str(i).strip() for i in ids if str(i).strip()]


def fetch_managers(department_id: str) -> list[str]:
    """分页取完整名单（去重、保持顺序）。任一页失败即整体失败，不使用不完整的名单。"""
    settings = get_settings()
    url = settings.department_roster_url
    if not url:
        raise RosterError("未配置 DEPARTMENT_ROSTER_URL")
    page_size = settings.department_roster_page_size
    headers = ({"Authorization": f"Bearer {settings.department_roster_token}"}
               if settings.department_roster_token else {})
    managers: dict[str, None] = {}
    with httpx.Client(timeout=settings.department_roster_timeout, headers=headers) as client:
        for page in range(1, settings.department_roster_max_pages + 1):
            try:
                total, ids = _parse_page(_fetch_page(client, url, department_id, page, page_size))
            except (httpx.HTTPError, ValueError) as exc:
                raise RosterError(f"部门 {department_id} 名单第 {page} 页查询失败：{exc}") from exc
            managers.update(dict.fromkeys(ids))
            if not ids or page * page_size >= total:
                break
        else:
            raise RosterError(f"部门 {department_id} 名单超过 {settings.department_roster_max_pages} 页")
    if not managers:
        log.warning("部门 %s 名单为空，请确认部门 id", department_id)
    log.info("部门 %s 名单 %d 人（%d 页）", department_id, len(managers), page)
    return list(managers)
