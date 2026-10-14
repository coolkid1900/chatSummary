"""滑动窗口限流的 Lua 脚本测试，需要真实 Redis（不可达时跳过）。

本地运行：REDIS_HOST=127.0.0.1 python -m unittest tests.test_contacts_limiter
"""
import os
import time
import unittest
import uuid

import redis

from app.contacts.limiter import RateLimitTooLarge, SlidingWindowLimiter


def _client():
    client = redis.Redis(host=os.environ.get("REDIS_HOST", "127.0.0.1"),
                         port=int(os.environ.get("REDIS_PORT", "6379")), decode_responses=True,
                         protocol=2, socket_connect_timeout=1)
    try:
        client.ping()
    except redis.RedisError:
        return None
    return client


class LimiterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.redis = _client()
        if cls.redis is None:
            raise unittest.SkipTest("Redis 不可达")

    def make(self, **kwargs):
        name = f"test_{uuid.uuid4().hex[:8]}"
        limiter = SlidingWindowLimiter(name, self.redis, **{
            "max_requests": 3, "request_window": 1.0, "max_tokens": 1000, "token_window": 2.0, **kwargs})
        self.addCleanup(lambda: self.redis.delete(*limiter.keys))
        return limiter

    def test_request_count_within_window(self):
        limiter = self.make()
        waits = [limiter.try_acquire(10, f"c{i}") for i in range(4)]
        self.assertEqual(waits[:3], [0, 0, 0])
        self.assertGreater(waits[3], 0.5)
        self.assertLessEqual(waits[3], 1.0)

    def test_requests_release_after_window(self):
        limiter = self.make(request_window=0.3)
        for i in range(3):
            limiter.try_acquire(10, f"c{i}")
        self.assertGreater(limiter.try_acquire(10, "c3"), 0)
        time.sleep(0.35)
        self.assertEqual(limiter.try_acquire(10, "c4"), 0)

    def test_token_budget_and_wait_until_enough_expires(self):
        limiter = self.make(max_requests=100)
        self.assertEqual(limiter.try_acquire(600, "a"), 0)
        self.assertEqual(limiter.try_acquire(300, "b"), 0)
        wait = limiter.try_acquire(200, "c")  # 600+300+200 > 1000，需等 a 过期
        self.assertGreater(wait, 1.5)
        self.assertLessEqual(wait, 2.0)
        self.assertEqual(limiter.try_acquire(100, "d"), 0)

    def test_reconcile_with_actual_usage(self):
        limiter = self.make(max_requests=100)
        limiter.try_acquire(500, "a")
        limiter.reconcile("a", 900)
        self.assertGreater(limiter.try_acquire(200, "b"), 0)
        limiter.reconcile("a", 100)
        self.assertEqual(limiter.try_acquire(200, "b"), 0)

    def test_cooldown_blocks_all_callers(self):
        limiter = self.make()
        limiter.cooldown(0.5)
        self.assertGreater(limiter.try_acquire(10, "a"), 0.3)
        limiter.cooldown(0.1)  # 更短的冷却不缩短已有冷却
        self.assertGreater(limiter.try_acquire(10, "a"), 0.3)

    def test_low_priority_yields_to_waiting_high(self):
        limiter = self.make(max_requests=1)
        self.assertEqual(limiter.try_acquire(10, "first", "low"), 0)
        self.assertGreater(limiter.try_acquire(10, "high", "high"), 0)  # 等待中，登记高优先级标记
        self.assertEqual(limiter.try_acquire(10, "low", "low"), 0.5)

    def test_oversized_call_rejected(self):
        with self.assertRaises(RateLimitTooLarge):
            self.make().try_acquire(5000, "big")

    def test_blocking_acquire_respects_limit(self):
        limiter = self.make(max_requests=2, request_window=0.4)
        t0 = time.monotonic()
        for i in range(3):
            limiter.acquire(10)
        self.assertGreaterEqual(time.monotonic() - t0, 0.35)


if __name__ == "__main__":
    unittest.main()
