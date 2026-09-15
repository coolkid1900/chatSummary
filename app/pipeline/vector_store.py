"""步骤 3 落盘：向量分片存储（§9）—— 可插拔后端 + 容错复用。

每行存一个会话单元：(session_id, text, tokens, msg_count, customer, vector)。
  - text   ：脱敏后的原文，供 representative_docs / 送 LLM；
  - tokens ：jieba 预分词的空格词串，供聚类 vectorizer（避免 fit 时重复中文分词）；
  - vector ：bge-m3 向量，显式 **float32**（§容量测算 4 字节/维，避免 float64 翻倍）。

分片即「当天最贵的 embedding 结果」的持久化，重跑可直接复用（断点续跑，技术优化5）。
按日期分目录，文件名带 worker_id 防并发写冲突；local / s3 两后端可切换（§9）。
"""
from __future__ import annotations

import io
import json
import os
import shutil
from datetime import datetime, timedelta
from typing import Iterator, TypedDict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from app.config import get_settings

# 嵌入全部分片写完后才写入的「完成标记」，用于区分「写完」与「写一半崩了」（技术优化1）
MANIFEST = "_manifest.json"
PIPELINE_VERSION = "hotspot-v2"

# parquet 显式 schema：vector 用定长 float32 list
SCHEMA = pa.schema(
    [
        ("session_id", pa.string()),
        ("text", pa.string()),
        ("tokens", pa.string()),
        ("msg_count", pa.int32()),
        ("customer", pa.string()),
        ("vector", pa.list_(pa.float32())),
    ]
)


class Record(TypedDict):
    session_id: str
    text: str
    tokens: str
    msg_count: int
    customer: str
    vector: list[float]


class Batch(TypedDict):
    session_id: list[str]
    text: list[str]
    tokens: list[str]
    msg_count: list[int]
    customer: list[str]
    vectors: np.ndarray  # (n, dim) float32


def get_vector_store():
    settings = get_settings()
    if settings.vector_store_backend == "s3":
        return S3VectorStore()
    return LocalVectorStore()


def _embed_config() -> dict:
    """决定 vector 能否复用：模型/维度/预处理参数变了，向量必须重算。"""
    from app.lexicon import load_words
    import hashlib

    settings = get_settings()
    chitchat = "\n".join(sorted(load_words("chitchat")))
    chitchat_fp = hashlib.md5(chitchat.encode("utf-8")).hexdigest()[:16]
    return {
        "pipeline_version": PIPELINE_VERSION,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "embedding_input_version": "chars-truncate-v1",
        "embedding_max_input_chars": settings.embedding_max_input_chars,
        "min_text_len": settings.min_text_len,
        "session_gap_minutes": settings.session_gap_minutes,
        "chitchat_fingerprint": chitchat_fp,
        "vector_schema": "session-text-tokens-count-customer-float32-vector-v1",
    }


def _token_config() -> dict:
    """决定 tokens 能否复用：分词逻辑或词库变了，只需重分词（不必重嵌）。"""
    from app.lexicon import lexicon_fingerprint
    from app.pipeline.tokenizer import TOKENIZER_VERSION

    return {"tokenizer_version": TOKENIZER_VERSION, "lexicon": lexicon_fingerprint()}


def _embed_matches(m: dict | None) -> bool:
    return bool(m) and m.get("embed_config") == _embed_config()


def _token_matches(m: dict | None) -> bool:
    return bool(m) and m.get("token_config") == _token_config()


def _to_table(records: list[Record]) -> pa.Table:
    cols = {
        "session_id": [r["session_id"] for r in records],
        "text": [r["text"] for r in records],
        "tokens": [r["tokens"] for r in records],
        "msg_count": pa.array([r["msg_count"] for r in records], type=pa.int32()),
        "customer": [r["customer"] for r in records],
        "vector": pa.array(
            [np.asarray(r["vector"], dtype=np.float32).tolist() for r in records],
            type=pa.list_(pa.float32()),
        ),
    }
    return pa.table(cols, schema=SCHEMA)


def _table_to_batches(table: pa.Table, batch_size: int) -> Iterator[Batch]:
    df = table.to_pandas()
    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i : i + batch_size]
        yield Batch(
            session_id=chunk["session_id"].tolist(),
            text=chunk["text"].tolist(),
            tokens=chunk["tokens"].tolist(),
            msg_count=chunk["msg_count"].tolist(),
            customer=chunk["customer"].tolist(),
            vectors=np.asarray(list(chunk["vector"]), dtype=np.float32),
        )


class LocalVectorStore:
    def __init__(self):
        self.settings = get_settings()
        self.base = self.settings.vector_store_dir

    def _dir(self, date_str: str) -> str:
        return os.path.join(self.base, date_str)

    def _parquet_count(self, date_str: str) -> int:
        d = self._dir(date_str)
        if not os.path.isdir(d):
            return 0
        return sum(1 for f in os.listdir(d) if f.endswith(".parquet"))

    def clear(self, date_str: str) -> None:
        """删除该日期全部分片与标记（重嵌前清场，避免残留/seq 冲突）。"""
        shutil.rmtree(self._dir(date_str), ignore_errors=True)

    def write_manifest(self, date_str: str, n_shards: int, n_records: int) -> None:
        d = self._dir(date_str)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, MANIFEST), "w", encoding="utf-8") as f:
            json.dump(
                {"n_shards": n_shards, "n_records": n_records,
                 "worker_id": self.settings.worker_id,
                 "finished_at": datetime.utcnow().isoformat(),
                 "embed_config": _embed_config(),
                 "token_config": _token_config()},
                f,
                ensure_ascii=False,
            )

    def read_manifest(self, date_str: str) -> dict | None:
        path = os.path.join(self._dir(date_str), MANIFEST)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def is_complete(self, date_str: str) -> bool:
        """完全可复用 = 向量与 tokens 配置都匹配 且 分片数一致 → 整步跳过。"""
        m = self.read_manifest(date_str)
        return (
            _embed_matches(m) and _token_matches(m)
            and self._parquet_count(date_str) == m.get("n_shards")
        )

    def needs_retokenize(self, date_str: str) -> bool:
        """向量可复用但词库/分词变了 → 只需重分词，不必重嵌。"""
        m = self.read_manifest(date_str)
        return (
            _embed_matches(m) and not _token_matches(m)
            and self._parquet_count(date_str) == m.get("n_shards")
            and self._parquet_count(date_str) > 0
        )

    def rewrite_tokens(self, date_str: str, retok_fn) -> int:
        """复用已落盘向量，仅用 retok_fn 重算 tokens 列并原子写回，刷新 manifest。"""
        d = self._dir(date_str)
        files = sorted(f for f in os.listdir(d) if f.endswith(".parquet")) if os.path.isdir(d) else []
        total = 0
        for f in files:
            path = os.path.join(d, f)
            df = pq.read_table(path).to_pandas()
            new_tokens = retok_fn(df["text"].tolist())
            records = [
                Record(session_id=r.session_id, text=r.text, tokens=tok,
                       msg_count=int(r.msg_count), customer=r.customer, vector=list(r.vector))
                for r, tok in zip(df.itertuples(index=False), new_tokens)
            ]
            tmp = path + ".tmp"
            pq.write_table(_to_table(records), tmp)
            os.replace(tmp, path)
            total += len(records)
        m = self.read_manifest(date_str) or {}
        self.write_manifest(date_str, n_shards=m.get("n_shards", len(files)),
                            n_records=m.get("n_records", total))
        return total

    def write_shard(self, date_str: str, records: list[Record], seq: int) -> str:
        d = self._dir(date_str)
        os.makedirs(d, exist_ok=True)
        fname = f"shard-{self.settings.worker_id}-{seq:05d}.parquet"
        path = os.path.join(d, fname)
        tmp = path + ".tmp"
        pq.write_table(_to_table(records), tmp)
        os.replace(tmp, path)
        return path

    def iter_batches(self, date_str: str, batch_size: int | None = None) -> Iterator[Batch]:
        bs = batch_size or self.settings.cluster_batch_size
        d = self._dir(date_str)
        if not os.path.isdir(d):
            return
        for f in sorted(x for x in os.listdir(d) if x.endswith(".parquet")):
            yield from _table_to_batches(pq.read_table(os.path.join(d, f)), bs)

    def cleanup_expired(self) -> None:
        if not os.path.isdir(self.base):
            return
        cutoff = datetime.now() - timedelta(days=self.settings.vector_ttl_days)
        for name in os.listdir(self.base):
            try:
                d = datetime.strptime(name, "%Y-%m-%d")
            except ValueError:
                continue
            if d < cutoff:
                shutil.rmtree(os.path.join(self.base, name), ignore_errors=True)


class S3VectorStore:
    """S3 兼容对象存储后端。过期清理交给 bucket lifecycle 规则（见 README）。"""

    def __init__(self):
        import boto3

        self.settings = get_settings()
        self.bucket = self.settings.s3_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=self.settings.s3_endpoint or None,
            aws_access_key_id=self.settings.s3_access_key or None,
            aws_secret_access_key=self.settings.s3_secret_key or None,
            region_name=self.settings.s3_region,
        )

    def _prefix(self, date_str: str) -> str:
        return f"embeddings/{date_str}/"

    def _parquet_count(self, date_str: str) -> int:
        n = 0
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._prefix(date_str)):
            n += sum(1 for o in page.get("Contents", []) if o["Key"].endswith(".parquet"))
        return n

    def clear(self, date_str: str) -> None:
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._prefix(date_str)):
            keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
            if keys:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": keys})

    def write_manifest(self, date_str: str, n_shards: int, n_records: int) -> None:
        body = json.dumps(
            {"n_shards": n_shards, "n_records": n_records,
             "worker_id": self.settings.worker_id,
             "finished_at": datetime.utcnow().isoformat(),
             "embed_config": _embed_config(),
             "token_config": _token_config()},
            ensure_ascii=False,
        ).encode("utf-8")
        self.client.put_object(Bucket=self.bucket, Key=self._prefix(date_str) + MANIFEST, Body=body)

    def read_manifest(self, date_str: str) -> dict | None:
        try:
            body = self.client.get_object(
                Bucket=self.bucket, Key=self._prefix(date_str) + MANIFEST
            )["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None
        return json.loads(body)

    def is_complete(self, date_str: str) -> bool:
        m = self.read_manifest(date_str)
        return (
            _embed_matches(m) and _token_matches(m)
            and self._parquet_count(date_str) == m.get("n_shards")
        )

    def needs_retokenize(self, date_str: str) -> bool:
        m = self.read_manifest(date_str)
        return (
            _embed_matches(m) and not _token_matches(m)
            and self._parquet_count(date_str) == m.get("n_shards")
            and self._parquet_count(date_str) > 0
        )

    def _shard_keys(self, date_str: str) -> list[str]:
        keys: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._prefix(date_str)):
            keys += [o["Key"] for o in page.get("Contents", []) if o["Key"].endswith(".parquet")]
        return sorted(keys)

    def rewrite_tokens(self, date_str: str, retok_fn) -> int:
        total = 0
        keys = self._shard_keys(date_str)
        for key in keys:
            body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
            df = pq.read_table(io.BytesIO(body)).to_pandas()
            new_tokens = retok_fn(df["text"].tolist())
            records = [
                Record(session_id=r.session_id, text=r.text, tokens=tok,
                       msg_count=int(r.msg_count), customer=r.customer, vector=list(r.vector))
                for r, tok in zip(df.itertuples(index=False), new_tokens)
            ]
            buf = io.BytesIO()
            pq.write_table(_to_table(records), buf)
            buf.seek(0)
            self.client.put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())
            total += len(records)
        m = self.read_manifest(date_str) or {}
        self.write_manifest(date_str, n_shards=m.get("n_shards", len(keys)),
                            n_records=m.get("n_records", total))
        return total

    def write_shard(self, date_str: str, records: list[Record], seq: int) -> str:
        key = f"{self._prefix(date_str)}shard-{self.settings.worker_id}-{seq:05d}.parquet"
        buf = io.BytesIO()
        pq.write_table(_to_table(records), buf)
        buf.seek(0)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=buf.getvalue())
        return f"s3://{self.bucket}/{key}"

    def iter_batches(self, date_str: str, batch_size: int | None = None) -> Iterator[Batch]:
        bs = batch_size or self.settings.cluster_batch_size
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._prefix(date_str)):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith(".parquet"):
                    continue
                body = self.client.get_object(Bucket=self.bucket, Key=obj["Key"])["Body"].read()
                yield from _table_to_batches(pq.read_table(io.BytesIO(body)), bs)

    def cleanup_expired(self) -> None:
        # 交给 bucket lifecycle；此处不主动删，避免误删并发写入。
        return
