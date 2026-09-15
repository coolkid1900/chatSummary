"""步骤 3：bge-m3 文本嵌入（§5.3 / §8）—— 最慢步，重点减量。

减量手段：
  - 文本去重：同一文本只嵌一次（本批内去重 + Redis 跨天缓存复用，§8.1）。
  - 批量打满：单请求塞 batch_size 条，等效吞吐 = m × batch（§8.3）。
  - 全局限流：调用前向 Redis 令牌桶取令牌，多 worker 共享同一桶（§12）。
  - 单 pod 内并发：受令牌桶约束的并发请求，打满 m/s 限额。
"""
from __future__ import annotations

import base64
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.pipeline.embedding_input import EmbeddingInput
from app.pipeline.ratelimit import TokenBucket
from app.pipeline.preprocess import SessionUnit
from app.redis_client import get_redis


class Embedder:
    def __init__(self):
        self.settings = get_settings()
        self.dim = self.settings.embedding_dim
        self.redis = get_redis()
        self.bucket = TokenBucket("embedding", self.settings.embedding_rate_per_sec)
        self._client = None
        self._lock = threading.Lock()  # 保护并发下的计数
        self.input = EmbeddingInput(
            self.settings.embedding_max_input_chars
        )
        # 统计
        self.api_calls = 0
        self.cache_hits = 0

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.settings.embedding_base_url,
                api_key=self.settings.embedding_api_key,
            )
        return self._client

    def _cache_key(self, text_hash: str) -> str:
        return (f"emb:{self.settings.embedding_model}:chars-truncate-v1:"
                f"{self.settings.embedding_max_input_chars}:{text_hash}")

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(min=1, max=20))
    def _call_api(self, texts: list[str]) -> list[list[float]]:
        # 取令牌（全局令牌桶）：即使多线程并发，实际请求速率仍被限在 m/s，不会超供应商限速
        self.bucket.acquire(1)
        resp = self.client.embeddings.create(
            model=self.settings.embedding_model, input=texts
        )
        with self._lock:
            self.api_calls += 1
        embs = [d.embedding for d in resp.data]
        if len(embs) != len(texts):
            raise ValueError(f"embedding 返回数量异常: got={len(embs)} expected={len(texts)}")
        for emb in embs:
            self._validate_dim(emb)
        return embs

    def _validate_dim(self, vec: list[float]) -> None:
        if len(vec) != self.dim:
            raise ValueError(f"embedding 维度异常: got={len(vec)} expected={self.dim}")

    @staticmethod
    def _encode(vec: list[float]) -> str:
        return base64.b64encode(np.asarray(vec, dtype=np.float32).tobytes()).decode("ascii")

    @staticmethod
    def _decode(blob: str) -> list[float]:
        return np.frombuffer(base64.b64decode(blob), dtype=np.float32).tolist()

    def _cacheable(self, text: str) -> bool:
        # 只缓存短文本（高频话术），符合 §9「Redis 只放小热数据」；长尾会话靠 parquet 复用
        return len(text) <= self.settings.embed_cache_max_len

    def embed_units(self, units: Iterable[SessionUnit]) -> dict[str, list[float]]:
        """对会话单元批量嵌入，返回 session_id -> vector。"""
        units = list(units)
        # 1) 按 text_hash 去重：唯一文本集合
        hash_to_text: dict[str, str] = {}
        for u in units:
            hash_to_text.setdefault(u.text_hash, u.text)

        # 2) 查缓存（仅短文本）
        vectors: dict[str, list[float]] = {}
        misses: list[str] = []
        for h, text in hash_to_text.items():
            if self._cacheable(text):
                cached = self.redis.get(self._cache_key(h))
                if cached is not None:
                    vec = self._decode(cached)
                    if len(vec) == self.dim:
                        vectors[h] = vec
                        self.cache_hits += 1
                        continue
                    self.redis.delete(self._cache_key(h))
            misses.append(h)

        # 按字符数截断请求文本；保留原会话和原文 hash。
        prepared = {h: self.input.truncate(hash_to_text[h]) for h in misses}

        # 3) 未命中的批量调用：并发发请求（令牌桶把实际速率限在 m/s，打满限额）
        batch_size = self.settings.embedding_batch_size
        batches = [misses[i : i + batch_size] for i in range(0, len(misses), batch_size)]

        def _work(batch_hashes: list[str]) -> tuple[list[str], list[list[float]]]:
            batch_texts = [prepared[h] for h in batch_hashes]
            return batch_hashes, self._call_api(batch_texts)

        workers = max(1, self.settings.embedding_concurrency)
        if batches and workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                results = list(ex.map(_work, batches))
        else:
            results = [_work(b) for b in batches]

        # 缓存写回在主线程串行做，避免 Redis 并发争用
        for batch_hashes, embs in results:
            for h, emb in zip(batch_hashes, embs):
                vectors[h] = emb
                if self._cacheable(hash_to_text[h]):  # 仅短文本（高频话术）写缓存
                    self.redis.set(
                        self._cache_key(h), self._encode(emb), ex=self.settings.dedup_cache_ttl
                    )

        # 4) 映射回每个 session_id
        return {u.session_id: vectors[u.text_hash] for u in units}
