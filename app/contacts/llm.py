"""接触总结的大模型调用：限流预占 → 调用 → 修正实际 token → 记录 llm_calls。

- 429：按 Retry-After（缺省 10 秒）设置全局冷却后重试。
- 5xx / 超时 / 连接错误 / JSON 解析失败：指数退避重试，最多 3 次。
- 输出被截断（finish_reason=length）：max_tokens 翻倍后重试。
- 其他 4xx（鉴权、参数错误等）：立即失败，不重试；401/402/403 之后同一实例的调用直接失败。
每次逻辑调用只写一行 llm_calls，attempts 记录实际尝试次数；不保存 prompt 原文。
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass

from app.config import get_settings
from app.contacts.limiter import SlidingWindowLimiter
from app.contacts.transcript import estimate_tokens
from app.db import get_session
from app.models import LlmCall

log = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_MESSAGE_OVERHEAD = 20  # 对话模板固定开销的估算
_FATAL_STATUS = (401, 402, 403)  # key 无效、余额不足、无权限：后续调用不可能成功
_MAX_OUTPUT_TOKENS = 8000


class LlmCallError(RuntimeError):
    pass


@dataclass
class LlmResult:
    data: dict
    call_id: str
    prompt_tokens: int


def parse_json(content: str) -> dict:
    """容错解析：模型偶尔会包一层 ```json 代码块。"""
    text = (content or "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    data = json.loads(match.group(0) if match else text)
    if not isinstance(data, dict):
        raise ValueError("模型输出不是 JSON 对象")
    return data


def _retry_after(exc) -> float:
    try:
        return max(1.0, float(exc.response.headers.get("retry-after")))
    except (AttributeError, TypeError, ValueError):
        return 10.0


def _cached_tokens(usage) -> int:
    # DeepSeek 返回 prompt_cache_hit_tokens；OpenAI 兼容实现放在 prompt_tokens_details。
    hit = getattr(usage, "prompt_cache_hit_tokens", None)
    if hit is None:
        details = getattr(usage, "prompt_tokens_details", None)
        hit = getattr(details, "cached_tokens", 0) if details else 0
    return int(hit or 0)


class ContactLlm:
    def __init__(self, limiter: SlidingWindowLimiter | None = None, client=None, stop=None):
        self.settings = get_settings()
        self.limiter = limiter or SlidingWindowLimiter()
        self._client = client
        self.stop = stop
        # 出现鉴权/余额类错误后，同一实例的后续调用直接失败，不再占用限流配额逐个请求。
        self.fatal_error: str | None = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.settings.llm_base_url,
                                  api_key=self.settings.llm_api_key,
                                  timeout=self.settings.contact_llm_timeout, max_retries=0)
        return self._client

    def chat_json(self, *, purpose: str, system: str, user: str, prompt_version: str,
                  max_tokens: int, priority: str = "low", manager_id: str = "",
                  customer_id: str = "") -> LlmResult:
        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

        if self.fatal_error:
            raise LlmCallError(self.fatal_error)
        est = estimate_tokens(system) + estimate_tokens(user) + _MESSAGE_OVERHEAD
        if est > self.settings.contact_max_call_tokens:
            raise LlmCallError(f"单次输入估算 {est} token 超过上限 {self.settings.contact_max_call_tokens}")
        call_id = uuid.uuid4().hex
        record = dict(call_id=call_id, purpose=purpose, manager_id=manager_id,
                      customer_id=customer_id, est_input_tokens=est,
                      prompt_version=prompt_version, model=self.settings.llm_model)
        error = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            # 每次尝试单独登记限流配额，重试也计入窗口。
            slot = self.limiter.acquire(est, priority, call_id=f"{call_id}:{attempt}", stop=self.stop)
            if self.fatal_error:  # 排队期间其他线程已遇到鉴权/余额错误
                self.limiter.reconcile(slot, 0)
                raise LlmCallError(self.fatal_error)
            t0 = time.monotonic()
            try:
                resp = self.client.chat.completions.create(
                    model=self.settings.llm_model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=0.2,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    **self._extra_body(),
                )
            except RateLimitError as exc:
                wait = _retry_after(exc)
                self.limiter.cooldown(wait)
                error = f"429，冷却 {wait:.0f}s"
                record.update(status="rate_limited", http_status=429)
                log.warning("LLM 限流 call_id=%s attempt=%d 冷却 %.0fs", call_id, attempt, wait)
                continue
            except APIStatusError as exc:
                record.update(status="failed", http_status=exc.status_code)
                error = f"HTTP {exc.status_code}: {exc}"[:500]
                if exc.status_code in _FATAL_STATUS:
                    self.fatal_error = error
                    log.error("LLM 鉴权/余额错误，本批次后续调用直接失败：%s", error)
                if exc.status_code < 500:
                    break
                time.sleep(2 ** attempt)
                continue
            except (APITimeoutError, APIConnectionError) as exc:
                record["status"] = "failed"
                error = f"{type(exc).__name__}: {exc}"[:500]
                time.sleep(2 ** attempt)
                continue

            usage = resp.usage
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or est)
            self.limiter.reconcile(slot, prompt_tokens)
            record.update(http_status=200, prompt_tokens=record.get("prompt_tokens", 0) + prompt_tokens,
                          cached_tokens=record.get("cached_tokens", 0) + _cached_tokens(usage),
                          completion_tokens=record.get("completion_tokens", 0)
                          + int(getattr(usage, "completion_tokens", 0) or 0),
                          latency_ms=record.get("latency_ms", 0) + int((time.monotonic() - t0) * 1000))
            if getattr(resp.choices[0], "finish_reason", None) == "length":
                # 输出（含推理 token）达到上限被截断，加大上限重试。
                error = f"输出达到 max_tokens={max_tokens} 上限被截断"
                record["status"] = "invalid"
                max_tokens = min(max_tokens * 2, _MAX_OUTPUT_TOKENS)
                continue
            try:
                data = parse_json(resp.choices[0].message.content)
            except (ValueError, IndexError, AttributeError) as exc:
                error = f"JSON 解析失败: {exc}"[:500]
                record["status"] = "invalid"
                continue
            self._save(record, status="ok", attempts=attempt, error=None)
            return LlmResult(data=data, call_id=call_id, prompt_tokens=record["prompt_tokens"])

        self._save(record, status=record.get("status") or "failed", attempts=attempt, error=error)
        raise LlmCallError(error or "LLM 调用失败")

    def _extra_body(self) -> dict:
        # DeepSeek 新模型默认先推理，推理 token 计入 max_tokens 且增加耗时与费用；
        # 结构化抽取不需要推理，默认关闭。其他厂商不支持该参数时设置 CONTACT_LLM_DISABLE_THINKING=false。
        if self.settings.contact_llm_disable_thinking:
            return {"extra_body": {"thinking": {"type": "disabled"}}}
        return {}

    @staticmethod
    def _save(record: dict, *, status: str, attempts: int, error: str | None) -> None:
        session = get_session()
        try:
            session.add(LlmCall(**{**record, "status": status, "attempts": attempts, "error": error}))
            session.commit()
        except Exception:
            # 调用日志用于校准与审计，写入失败不影响业务结果。
            log.exception("llm_calls 写入失败 call_id=%s", record.get("call_id"))
        finally:
            session.close()
