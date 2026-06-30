"""Redis 全局令牌桶限流（§12）。

跨进程 / 跨 worker 统一管控 embedding（每秒 m 次）与 DeepSeek（每秒 n 次）调用。
用 Lua 脚本保证「取令牌」原子性，避免多 worker 竞态超限。
"""
from __future__ import annotations

import time

from app.redis_client import get_redis

# KEYS[1]=bucket key
# ARGV[1]=rate(每秒补充), ARGV[2]=capacity(桶容量), ARGV[3]=now(秒,浮点), ARGV[4]=需要的令牌数
# 返回：>=0 表示成功并给出剩余；<0 表示需等待的秒数（取负）
_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local need = tonumber(ARGV[4])

local data = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(data[1])
local ts = tonumber(data[2])
if tokens == nil then
  tokens = capacity
  ts = now
end

local delta = math.max(0, now - ts)
tokens = math.min(capacity, tokens + delta * rate)

if tokens >= need then
  tokens = tokens - need
  redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
  redis.call('EXPIRE', key, 3600)
  return '0'
else
  local deficit = need - tokens
  redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
  redis.call('EXPIRE', key, 3600)
  return tostring(deficit / rate)
end
"""


class TokenBucket:
    def __init__(self, name: str, rate_per_sec: float, capacity: float | None = None):
        self.key = f"ratelimit:{name}"
        self.rate = float(rate_per_sec)
        # 容量默认 = 1 秒的量（至少 1），允许小幅突发
        self.capacity = float(capacity) if capacity else max(1.0, self.rate)
        self._redis = get_redis()
        self._script = self._redis.register_script(_LUA)

    def acquire(self, n: int = 1) -> None:
        """阻塞直到取到 n 个令牌。"""
        while True:
            wait = float(
                self._script(keys=[self.key], args=[self.rate, self.capacity, time.time(), n])
            )
            if wait <= 0:
                return
            time.sleep(min(wait, 1.0))
