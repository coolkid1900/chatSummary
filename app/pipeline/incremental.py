"""固定 PCA → 缓存降维向量 → 多轮 KMeans → 按最终标签汇总。

缓存只在本次运行有效，退出（包括异常）时删除；不加载整个原始向量矩阵。
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

try:
    import resource
except ImportError:  # Windows 没有该 Unix 模块；监控缺失不应阻止聚类。
    resource = None

import numpy as np
from sklearn.cluster import MiniBatchKMeans, kmeans_plusplus

from app.pipeline.reduction import Float32IncrementalPCA

log = logging.getLogger(__name__)


def _process_peak_rss_mib() -> float | None:
    """进程启动以来的内存高水位；平台不支持时返回 None，不用当前 RSS 冒充峰值。"""
    if resource is None:
        return None
    try:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except OSError:
        return None
    return round(peak / (1024 ** 2 if sys.platform == "darwin" else 1024), 1)


@contextmanager
def _stage(result, name):
    started = time.perf_counter()
    try:
        yield
    finally:
        stats = {"seconds": round(time.perf_counter() - started, 3),
                 "process_peak_rss_mib": _process_peak_rss_mib()}
        result.metrics.setdefault("stages", {})[name] = stats
        log.info("incremental stage=%s %s", name, json.dumps(stats))


def _validated(vectors, dimension=None):
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError("向量必须是非零维的二维矩阵")
    if dimension is not None and vectors.shape[1] != dimension:
        raise ValueError("分片间向量维度不一致")
    if not np.isfinite(vectors).all():
        raise ValueError("向量含 NaN 或 Infinity")
    return vectors


def _digest(vectors):
    return hashlib.sha256(memoryview(vectors).cast("B")).hexdigest()


def _fit_pca(store, date_str, settings):
    requested = settings.incremental_n_components or settings.n_components
    reducer, pending, dimension = None, None, None
    total = 0
    fingerprint = hashlib.sha256()
    for vectors in store.iter_vector_batches(date_str):
        vectors = _validated(vectors, dimension)
        if not len(vectors):
            continue
        if reducer is None:
            dimension = vectors.shape[1]
            reducer = Float32IncrementalPCA(n_components=min(requested, dimension))
        total += len(vectors)
        fingerprint.update(memoryview(vectors).cast("B"))
        pending = vectors.copy() if pending is None else np.concatenate((pending, vectors))
        minimum = reducer.n_components
        # 按训练批大小累积，文件分片边界不触发拟合。只有下一批已足够时才
        # 释放完整训练批；小数据只拟合一次，不强切成 (N-components)+components。
        target = max(settings.cluster_batch_size, 2 * minimum)
        while len(pending) >= target + minimum:
            with np.errstate(divide="ignore", invalid="ignore"):
                reducer.partial_fit(pending[:target])
            pending = pending[target:].copy()
    if reducer is not None:
        if total < reducer.n_components:
            reducer.n_components = total
        with np.errstate(divide="ignore", invalid="ignore"):
            reducer.partial_fit(pending)
    return reducer, total, fingerprint.hexdigest()


@dataclass
class CachedBatch:
    path: Path
    rows: int
    digest: str


class _PrioritySample:
    """为每行分配独立随机优先级，保留最小 k 个，等概率覆盖全部分片。"""
    def __init__(self, capacity, seed):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.values = None
        self.keys = np.empty(0)

    def add(self, vectors):
        values = vectors if self.values is None else np.concatenate((self.values, vectors))
        keys = np.concatenate((self.keys, self.rng.random(len(vectors))))
        keep = np.argsort(keys, kind="stable")[:self.capacity]
        self.values = values[keep].copy()
        self.keys = keys[keep]


def _cache_reduced(store, date_str, settings, reducer, directory, expected_total, fingerprint):
    batches = []
    sample = _PrioritySample(max(settings.incremental_init_sample_size, settings.n_clusters),
                             settings.cluster_random_seed)
    total = 0
    current = hashlib.sha256()
    for vectors in store.iter_vector_batches(date_str):
        vectors = _validated(vectors, reducer.n_features_in_)
        if not len(vectors):
            continue
        current.update(memoryview(vectors).cast("B"))
        reduced = reducer.transform(vectors)
        path = Path(directory) / f"batch-{len(batches):06d}.npy"
        np.save(path, reduced, allow_pickle=False)
        batches.append(CachedBatch(path, len(vectors), _digest(vectors)))
        sample.add(reduced)
        total += len(vectors)
    if total != expected_total or current.hexdigest() != fingerprint:
        raise ValueError("聚类期间向量分片发生变化，请使用完整且固定的分片重跑")
    return batches, sample.values


def _train_kmeans(batches, sample, settings, result):
    n_clusters = min(settings.n_clusters, len(sample))
    centers, _ = kmeans_plusplus(sample, n_clusters=n_clusters, random_state=settings.cluster_random_seed)
    clusterer = MiniBatchKMeans(n_clusters=n_clusters, init=centers, n_init=1,
                               batch_size=max(settings.cluster_batch_size, n_clusters),
                               random_state=settings.cluster_random_seed, compute_labels=False)
    # 先让所有中心在全局样本上得到计数，避免首个同质分片把未命中中心重新分配。
    clusterer.partial_fit(sample)
    rng = np.random.default_rng(settings.cluster_random_seed)
    inertia = []
    for epoch in range(settings.incremental_epochs):
        pending = None
        for index in rng.permutation(len(batches)):
            vectors = np.load(batches[index].path, allow_pickle=False)
            vectors = vectors[rng.permutation(len(vectors))]
            pending = vectors if pending is None else np.concatenate((pending, vectors))
            # 缓存文件边界不决定更新步长；尾批不足 K 时并入前批。
            target = max(settings.cluster_batch_size, n_clusters)
            while len(pending) >= target + n_clusters:
                clusterer.partial_fit(pending[:target])
                pending = pending[target:].copy()
        clusterer.partial_fit(pending)
        distances = clusterer.transform(sample)
        mean_squared_distance = float(np.square(distances.min(axis=1)).mean())
        inertia.append(mean_squared_distance)
        log.info("incremental epoch=%d/%d sample_mean_squared_distance=%.6f",
                 epoch + 1, settings.incremental_epochs, mean_squared_distance)
    result.metrics["effective_n_clusters"] = n_clusters
    result.metrics["epoch_sample_mean_squared_distance"] = inertia
    return clusterer


def _predict(clusterer, reduced, max_distance=0.0, min_margin=0.0):
    distances = clusterer.transform(reduced)
    labels = distances.argmin(axis=1).astype(np.int64)
    nearest = distances[np.arange(len(labels)), labels]
    reject = np.zeros(len(labels), dtype=bool)
    if max_distance > 0:
        reject |= nearest > max_distance
    if min_margin > 0 and distances.shape[1] > 1:
        second = np.partition(distances, 1, axis=1)[:, 1]
        margin = (second - nearest) / np.maximum(second, 1e-12)
        reject |= margin < min_margin
    labels[reject] = -1
    return labels, nearest


def run_incremental(store, date_str, settings, result):
    from app.pipeline.cluster import _accumulate, merge_similar_topics

    with _stage(result, "pca"):
        reducer, total, fingerprint = _fit_pca(store, date_str, settings)
    result.metrics["n_sessions"] = total
    if total == 0:
        result.metrics.update(assigned_sessions=0, rejected_sessions=0, coverage=0.0,
                              topics_before_merge=0, topics_after_merge=0)
        return result
    result.metrics["effective_n_components"] = reducer.n_components
    result.metrics["pca_explained_variance_ratio"] = float(
        np.nan_to_num(reducer.explained_variance_ratio_).sum()
    )
    if settings.incremental_cache_dir:
        Path(settings.incremental_cache_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="chat-cluster-", dir=settings.incremental_cache_dir) as directory:
        with _stage(result, "reduce_cache"):
            batches, sample = _cache_reduced(store, date_str, settings, reducer, directory, total, fingerprint)
        result.metrics["reduced_cache_bytes"] = sum(b.path.stat().st_size for b in batches)
        with _stage(result, "kmeans"):
            clusterer = _train_kmeans(batches, sample, settings, result)
        del sample
        assigned = rejected = rejected_messages = messages = 0
        squared_distance = 0.0
        index = 0
        with _stage(result, "assign"):
            for batch in store.iter_batches(date_str):
                vectors = _validated(batch["vectors"], reducer.n_features_in_)
                if not len(vectors):
                    continue
                if (index >= len(batches) or len(vectors) != batches[index].rows
                        or _digest(vectors) != batches[index].digest):
                    raise ValueError("最终归类时分片顺序或向量内容变化，拒绝使用错位的缓存标签")
                cached = np.load(batches[index].path, allow_pickle=False)
                labels, distances = _predict(clusterer, cached, settings.incremental_max_distance,
                                             settings.incremental_min_margin)
                _accumulate(result, batch, labels.tolist(), settings.repr_pool_size)
                excluded = labels == -1
                counts = np.asarray(batch["msg_count"], dtype=np.int64)
                assigned += int((~excluded).sum())
                rejected += int(excluded.sum())
                messages += int(counts.sum())
                rejected_messages += int(counts[excluded].sum())
                squared_distance += float(np.square(distances).sum())
                index += 1
            if index != len(batches):
                raise ValueError("最终归类时分片缺失，拒绝输出不完整结果")
        with _stage(result, "merge_words"):
            merge_similar_topics(result, settings.topic_merge_sim)
    result.metrics.update(assigned_sessions=assigned, rejected_sessions=rejected,
                          n_messages=messages, rejected_messages=rejected_messages,
                          coverage=assigned / total, mean_squared_distance=squared_distance / total,
                          topic_session_counts=dict(sorted(result._seen.items())))
    log.info("incremental summary=%s", json.dumps(result.metrics, ensure_ascii=False, sort_keys=True))
    return result
