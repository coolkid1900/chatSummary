"""降维、聚类和最终主题汇总。

incremental 先固定 PCA，再训练 KMeans，最终标签统一驱动热度、词频和代表池。
向量工作集按批次读取；客户集合与主题词表仍随数据规模增长。
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bertopic import BERTopic

from app.config import get_settings
from app.pipeline.tokenizer import build_vectorizer
from app.pipeline.vector_store import Batch, get_vector_store


def build_models():
    settings = get_settings()
    if settings.cluster_backend == "umap_hdbscan":
        if settings.use_gpu:
            from cuml.cluster import HDBSCAN
            from cuml.manifold import UMAP
        else:
            from hdbscan import HDBSCAN
            from umap import UMAP
        return (UMAP(n_components=settings.n_components, random_state=42),
                HDBSCAN(min_cluster_size=settings.min_cluster_size, prediction_data=True))
    if settings.cluster_backend == "incremental":
        from sklearn.cluster import MiniBatchKMeans
        from app.pipeline.reduction import Float32IncrementalPCA

        return (Float32IncrementalPCA(n_components=settings.incremental_n_components or settings.n_components),
                MiniBatchKMeans(n_clusters=settings.n_clusters, random_state=settings.cluster_random_seed))
    raise ValueError(f"未知 CLUSTER_BACKEND: {settings.cluster_backend}")


def build_topic_model(vectorizer_model, representation_model=None) -> BERTopic:
    from bertopic import BERTopic

    dim_model, cluster_model = build_models()
    return BERTopic(
        umap_model=dim_model, hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model, representation_model=representation_model,
        embedding_model=None, language="multilingual",
        calculate_probabilities=False, verbose=False,
    )


@dataclass
class ClusterResult:
    topic_model: BERTopic | None
    heat_by_topic: dict[int, int]
    repr_pool: dict[int, list[tuple[str, np.ndarray]]]
    customers_by_topic: dict[int, set[str]] = field(default_factory=dict)
    members_by_topic: dict[int, list[int]] = field(default_factory=dict)
    hot_words_by_topic: dict[int, list[tuple[str, float]]] = field(default_factory=dict)
    vector_sums: dict[int, np.ndarray] = field(default_factory=dict)
    term_counts: dict[int, Counter] = field(default_factory=dict)
    metrics: dict[str, object] = field(default_factory=dict)
    _rng: random.Random = field(default_factory=lambda: random.Random(42), repr=False)
    n_topics: int = 0
    _seen: dict[int, int] = field(default_factory=dict)

    def _add_repr(self, topic: int, text: str, vec: np.ndarray, cap: int) -> None:
        seen = self._seen.get(topic, 0)
        pool = self.repr_pool.setdefault(topic, [])
        if len(pool) < cap:
            pool.append((text, vec.copy()))
        else:
            j = self._rng.randint(0, seen)
            if j < cap:
                pool[j] = (text, vec.copy())
        self._seen[topic] = seen + 1


def _topic_words(counts: dict[int, Counter], top_n: int = 10) -> dict[int, list[tuple[str, float]]]:
    """c-TF-IDF = L1 词频 × log(1 + 平均主题词数 / 全局词频)。"""
    from scipy.sparse import csr_matrix
    from sklearn.preprocessing import normalize

    topics = sorted(counts)
    vocabulary = sorted({word for counter in counts.values() for word in counter})
    if not vocabulary:
        return {topic: [] for topic in topics}
    index = {word: i for i, word in enumerate(vocabulary)}
    rows, cols, values = [], [], []
    for row, topic in enumerate(topics):
        for word, count in counts[topic].items():
            rows.append(row)
            cols.append(index[word])
            values.append(count)
    matrix = csr_matrix((values, (rows, cols)), shape=(len(topics), len(vocabulary)), dtype=np.float64)
    average = float(np.asarray(matrix.sum(axis=1)).mean())
    idf = np.log1p(average / np.asarray(matrix.sum(axis=0)).ravel())
    scores = normalize(matrix, norm="l1", copy=True).multiply(idf).tocsr()
    words = {}
    for row, topic in enumerate(topics):
        vector = scores.getrow(row)
        ranked = sorted(zip(vector.indices, vector.data), key=lambda pair: (-pair[1], vocabulary[pair[0]]))
        words[topic] = [(vocabulary[i], float(score)) for i, score in ranked[:top_n]]
    return words


def _merged_pool(result: ClusterResult, members: list[int], cap: int):
    """按会话数分配配额；容量允许时每个子主题至少保留一条，再独立抽样。"""
    eligible = sorted((m for m in members if result.repr_pool.get(m)),
                      key=lambda m: (-result._seen[m], m))
    quotas = {m: 0 for m in eligible}
    for m in eligible[:cap]:
        quotas[m] = 1
    remaining = cap - sum(quotas.values())
    while remaining > 0:
        available = [m for m in eligible if quotas[m] < len(result.repr_pool[m])]
        if not available:
            break
        selected = max(available, key=lambda m: (result._seen[m] / (quotas[m] + 1), -m))
        quotas[selected] += 1
        remaining -= 1
    return [item for m in eligible for item in result._rng.sample(result.repr_pool[m], quotas[m])]


def group_topic_centers(centers: np.ndarray, threshold: float, method: str) -> np.ndarray:
    """以原始向量的主题质心归并；average 对初始簇等权，不按消息热度加权。

    single 等价于阈值图的连通分量；complete 要求组内任意两质心达标。
    零质心独立保留。输出是局部分组标签，不是最终 topic_id。
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if method not in {"single", "average", "complete"}:
        raise ValueError(f"未知主题合并方式: {method}")
    centers = np.asarray(centers, dtype=np.float64)
    labels = np.arange(len(centers), dtype=np.int64)
    if threshold <= 0 or len(centers) < 2:
        return labels
    norms = np.linalg.norm(centers, axis=1)
    valid = np.flatnonzero(norms > 1e-12)
    if len(valid) > 1:
        normalized = centers[valid] / norms[valid, None]
        distances = np.clip(1 - normalized @ normalized.T, 0, 2)
        distances = (distances + distances.T) / 2
        np.fill_diagonal(distances, 0)
        labels[valid] = len(centers) + fcluster(
            linkage(squareform(distances), method=method),
            t=1 - threshold, criterion="distance",
        )
    return labels


def merge_similar_topics(result: ClusterResult, threshold: float, method: str | None = None) -> None:
    """全体会话质心 + 可配置链接规则；重新汇总最终分组的词频、热度和代表池。"""
    settings = get_settings()
    method = method or settings.topic_merge_linkage
    topics = sorted(result.heat_by_topic)
    groups = [[topic] for topic in topics]
    if threshold > 0 and len(topics) > 1:
        centers = np.asarray([result.vector_sums[t] for t in topics])
        labels = group_topic_centers(centers, threshold, method)
        grouped: dict[int, list[int]] = {}
        for topic, label in zip(topics, labels):
            grouped.setdefault(int(label), []).append(topic)
        groups = list(grouped.values())

    cap = settings.repr_pool_size
    heat, pools, customers, members_map, counts, sums, seen = {}, {}, {}, {}, {}, {}, {}
    for members in groups:
        canon = min(members, key=lambda m: (-result.heat_by_topic[m], m))
        heat[canon] = sum(result.heat_by_topic[m] for m in members)
        pools[canon] = (result.repr_pool[members[0]] if len(members) == 1
                       else _merged_pool(result, members, cap))
        customers[canon] = set().union(*(result.customers_by_topic.get(m, set()) for m in members))
        members_map[canon] = sorted(members)
        counter = Counter()
        for m in members:
            counter.update(result.term_counts.get(m, {}))
        counts[canon] = counter
        sums[canon] = np.sum([result.vector_sums[m] for m in members], axis=0)
        seen[canon] = sum(result._seen[m] for m in members)
    result.metrics["topics_before_merge"] = len(topics)
    result.metrics["topics_after_merge"] = len(groups)
    result.metrics["topic_merge_linkage"] = method
    result.metrics["topic_merge_sim"] = threshold
    result.heat_by_topic, result.repr_pool = heat, pools
    result.customers_by_topic, result.members_by_topic = customers, members_map
    result.term_counts, result.vector_sums, result._seen = counts, sums, seen
    result.hot_words_by_topic = _topic_words(counts)
    result.n_topics = len(groups)


def _accumulate(result: ClusterResult, batch: Batch, topics: list[int], cap: int) -> None:
    has_customer = bool(batch["customer"])
    for i, topic in enumerate(topics):
        if topic == -1:
            continue
        result.heat_by_topic[topic] = result.heat_by_topic.get(topic, 0) + int(batch["msg_count"][i])
        vector = batch["vectors"][i]
        if topic not in result.vector_sums:
            result.vector_sums[topic] = np.zeros(vector.shape, dtype=np.float64)
        result.vector_sums[topic] += vector
        result.term_counts.setdefault(topic, Counter()).update(batch["tokens"][i].lower().split())
        result._add_repr(topic, batch["text"][i], vector, cap)
        if has_customer:
            result.customers_by_topic.setdefault(topic, set()).add(batch["customer"][i])


def run_clustering(date_str: str) -> ClusterResult:
    settings = get_settings()
    store = get_vector_store()
    cap = settings.repr_pool_size
    result = ClusterResult(topic_model=None, heat_by_topic={}, repr_pool={})
    result._rng.seed(settings.cluster_random_seed)
    if settings.cluster_backend == "incremental":
        from app.pipeline.incremental import run_incremental

        return run_incremental(store, date_str, settings, result)
    if settings.cluster_backend != "umap_hdbscan":
        raise ValueError(f"未知 CLUSTER_BACKEND: {settings.cluster_backend}")

    tokens, texts, vecs, counts, custs = [], [], [], [], []
    for b in store.iter_batches(date_str):
        tokens.extend(b["tokens"])
        texts.extend(b["text"])
        counts.extend(b["msg_count"])
        custs.extend(b["customer"])
        vecs.extend(list(b["vectors"]))
    if not tokens:
        return result
    embeddings = np.asarray(vecs, dtype=np.float32)
    del vecs
    topic_model = build_topic_model(build_vectorizer(online=False))
    topics, _ = topic_model.fit_transform(tokens, embeddings=embeddings)
    pseudo = Batch(session_id=[], text=texts, tokens=tokens, msg_count=counts,
                   customer=custs, vectors=embeddings)
    _accumulate(result, pseudo, [int(t) for t in topics], cap)
    result.topic_model = topic_model
    threshold = (settings.umap_topic_merge_sim if settings.umap_topic_merge_sim is not None
                 else settings.topic_merge_sim)
    merge_similar_topics(result, threshold, settings.umap_topic_merge_linkage)
    return result
