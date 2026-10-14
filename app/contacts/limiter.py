"""Redis 滑动窗口限流：同时约束请求数与输入 token，跨进程 / 跨 pod 共享。

令牌桶在任意窗口内最多放行“容量 + 速率 × 窗口”，会超过厂商“N 次/10 秒”的口径；
这里用 ZSET 记录窗口内每次调用，精确保证任意窗口内不超限。

- acquire：按估算 token 预占，请求数与 token 同时满足才登记（Lua 原子执行）。
- reconcile：拿到响应后用实际 prompt_tokens 修正，估算误差不会累积。
- cooldown：收到 429 后所有 worker 一起暂停。
- 优先级：high（按需总结）等待期间，low（夜间笔记）不发起新调用。

时间取调用方时钟（与现有 TokenBucket 一致），多 pod 部署需开启 NTP 同步。
"""
from __future__ import annotations

import time
import uuid

from app.config import get_settings
from app.redis_client import get_redis

# KEYS: 1=请求 ZSET 2=token ZSET 3=token 数 HASH 4=冷却键 5=高优先级等待标记
# ARGV: 1=now 2=请求窗口 3=最大请求数 4=token 窗口 5=最大 token 6=本次 token 7=call_id 8=优先级
# 返回需等待的秒数字符串，"0" 表示已登记成功
_ACQUIRE = """
local now = tonumber(ARGV[1])
local req_window, max_req = tonumber(ARGV[2]), tonumber(ARGV[3])
local tok_window, max_tok = tonumber(ARGV[4]), tonumber(ARGV[5])
local need, id, priority = tonumber(ARGV[6]), ARGV[7], ARGV[8]

local cooldown = redis.call('PTTL', KEYS[4])
if cooldown > 0 then return tostring(cooldown / 1000.0) end
if priority == 'high' then
  redis.call('SET', KEYS[5], '1', 'PX', 3000)
elseif redis.call('EXISTS', KEYS[5]) == 1 then
  return '0.5'
end

redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now - req_window)
local expired = redis.call('ZRANGEBYSCORE', KEYS[2], '-inf', now - tok_window)
if #expired > 0 then
  redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now - tok_window)
  redis.call('HDEL', KEYS[3], unpack(expired))
end

local wait = 0
local nreq = redis.call('ZCARD', KEYS[1])
if nreq >= max_req then
  local idx = nreq - max_req
  local oldest = redis.call('ZRANGE', KEYS[1], idx, idx, 'WITHSCORES')
  wait = math.max(wait, tonumber(oldest[2]) + req_window - now)
end

local members = redis.call('ZRANGE', KEYS[2], 0, -1, 'WITHSCORES')
local used, items = 0, {}
for i = 1, #members, 2 do
  local amount = tonumber(redis.call('HGET', KEYS[3], members[i]) or '0')
  used = used + amount
  items[#items + 1] = {tonumber(members[i + 1]), amount}
end
if used + need > max_tok then
  local excess, freed = used + need - max_tok, 0
  for _, item in ipairs(items) do
    freed = freed + item[2]
    if freed >= excess then
      wait = math.max(wait, item[1] + tok_window - now)
      break
    end
  end
end
if wait > 0 then return tostring(wait) end

redis.call('ZADD', KEYS[1], now, id)
redis.call('ZADD', KEYS[2], now, id)
redis.call('HSET', KEYS[3], id, need)
local ttl = math.ceil(math.max(req_window, tok_window)) + 60
for i = 1, 3 do redis.call('EXPIRE', KEYS[i], ttl) end
return '0'
"""

_RECONCILE = """
if redis.call('HEXISTS', KEYS[1], ARGV[1]) == 1 then
  redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
end
return 0
"""

_COOLDOWN = """
if redis.call('PTTL', KEYS[1]) < tonumber(ARGV[1]) then
  redis.call('SET', KEYS[1], '1', 'PX', ARGV[1])
end
return 0
"""


class RateLimitTooLarge(ValueError):
    """单次调用的估算 token 超过窗口上限，永远无法获得配额。"""


class SlidingWindowLimiter:
    def __init__(self, name: str = "contact_llm", redis=None, *, max_requests: int | None = None,
                 request_window: float | None = None, max_tokens: int | None = None,
                 token_window: float = 60.0):
        settings = get_settings()
        self.redis = redis or get_redis()
        self.max_requests = max_requests or settings.contact_llm_max_requests
        self.request_window = request_window or settings.contact_llm_request_window_sec
        self.max_tokens = max_tokens or int(
            settings.contact_llm_tokens_per_min * settings.contact_llm_token_safety)
        self.token_window = token_window
        prefix = f"ratelimit:{name}"
        self.keys = [f"{prefix}:req", f"{prefix}:tok", f"{prefix}:tokamt",
                     f"{prefix}:cooldown", f"{prefix}:high"]
        self._acquire = self.redis.register_script(_ACQUIRE)
        self._reconcile = self.redis.register_script(_RECONCILE)
        self._cooldown = self.redis.register_script(_COOLDOWN)

    def try_acquire(self, tokens: int, call_id: str, priority: str = "low") -> float:
        """尝试登记一次调用，返回需等待的秒数（0 表示成功）。"""
        if tokens > self.max_tokens:
            raise RateLimitTooLarge(f"单次估算 {tokens} token 超过窗口上限 {self.max_tokens}")
        wait = self._acquire(keys=self.keys, args=[
            time.time(), self.request_window, self.max_requests, self.token_window,
            self.max_tokens, tokens, call_id, priority,
        ])
        return float(wait)

    def acquire(self, tokens: int, priority: str = "low", call_id: str | None = None,
                stop=None) -> str:
        """阻塞直到获得配额，返回 call_id。stop 为 threading.Event 时可被中断。"""
        call_id = call_id or uuid.uuid4().hex
        while True:
            wait = self.try_acquire(tokens, call_id, priority)
            if wait <= 0:
                return call_id
            # 高优先级每轮都会刷新 3 秒的等待标记，睡眠不超过 1 秒保证标记不中断。
            delay = min(wait, 1.0)
            if stop is not None:
                if stop.wait(delay):
                    raise InterruptedError("限流等待被中断")
            else:
                time.sleep(delay)

    def reconcile(self, call_id: str, actual_tokens: int) -> None:
        self._reconcile(keys=[self.keys[2]], args=[call_id, int(actual_tokens)])

    def cooldown(self, seconds: float) -> None:
        self._cooldown(keys=[self.keys[3]], args=[int(seconds * 1000)])
