"""步骤 4~6：降维 + 聚类 + 归类（§6.0）。

降维 + 聚类必须可通过环境变量在两套实现间无改码切换（强制要求 §6.0）：
  - CLUSTER_BACKEND=incremental（默认）：IncrementalPCA + MiniBatchKMeans，partial_fit
    流式，内存只与单 batch 相关，2G pod 可跑。
  - CLUSTER_BACKEND=umap_hdbscan：UMAP + HDBSCAN（USE_GPU=true 用 cuML）。

技术优化2（fit / transform 分离）：incremental 路径**先**分批 partial_fit 建模，
**再**分批 transform 显式打标签，不再依赖 BERTopic 内部 topics_ 的追加顺序，更稳。

技术优化1（不全量驻内存）：聚类只从 parquet 分批读取；归类阶段用「按主题累计热度 +
蓄水池采样代表文档」替代「保留全部 docs」，内存与主题数×池大小相关，与总会话数无关。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from bertopic import BERTopic

from app.config import get_settings
from app.pipeline.tokenizer import build_vectorizer
from app.pipeline.vector_store import Batch, get_vector_store


def build_models():
    """§6.0：依据集中配置返回 (dim_model, cluster_model)。"""
    settings = get_settings()
    backend = settings.cluster_backend
    n_components = settings.n_components

    if backend == "umap_hdbscan":
        if settings.use_gpu:
            from cuml.cluster import HDBSCAN  # GPU (RAPIDS)
            from cuml.manifold import UMAP
        else:
            from hdbscan import HDBSCAN  # CPU
            from umap import UMAP
        dim_model = UMAP(n_components=n_components, random_state=42)
        cluster_model = HDBSCAN(
            min_cluster_size=settings.min_cluster_size,
            prediction_data=True,  # 支持 approximate_predict 批量推断
        )
    elif backend == "incremental":
        from sklearn.cluster import MiniBatchKMeans
        from app.pipeline.reduction import Float32IncrementalPCA

        dim_model = Float32IncrementalPCA(n_components=n_components)
        cluster_model = MiniBatchKMeans(
            n_clusters=settings.n_clusters, random_state=42
        )
    else:
        raise ValueError(f"未知 CLUSTER_BACKEND: {backend}")

    return dim_model, cluster_model


def build_topic_model(vectorizer_model, representation_model=None) -> BERTopic:
    dim_model, cluster_model = build_models()
    return BERTopic(
        umap_model=dim_model,          # 接受任何 fit/transform 模型
        hdbscan_model=cluster_model,   # 接受任何 fit/predict 模型
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        embedding_model=None,          # 始终传入预算向量，不本地建模
        # 默认 language="english" 会用正则剥除所有非 ASCII 字符（中文会被清空），
        # 必须设为 multilingual 跳过该清洗，保留中文供 c-TF-IDF。
        language="multilingual",
        calculate_probabilities=False,
        verbose=False,
    )


@dataclass
class ClusterResult:
    topic_model: BERTopic
    heat_by_topic: dict[int, int]                          # topic -> 累计客户消息数（热度）
    repr_pool: dict[int, list[tuple[str, np.ndarray]]]     # topic -> [(脱敏原文, 向量)]，蓄水池采样
    customers_by_topic: dict[int, set[str]] = field(default_factory=dict)  # topic -> 去重客户集合
    members_by_topic: dict[int, list[int]] = field(default_factory=dict)   # 合并后主题 -> 原始 topic 列表
    hot_words_by_topic: dict[int, list[tuple[str, float]]] = field(default_factory=dict)
    n_topics: int = 0
    _seen: dict[int, int] = field(default_factory=dict)    # 蓄水池内部计数

    def _add_repr(self, topic: int, text: str, vec: np.ndarray, cap: int) -> None:
        seen = self._seen.get(topic, 0)
        pool = self.repr_pool.setdefault(topic, [])
        if len(pool) < cap:
            pool.append((text, vec))
        else:  # 标准蓄水池替换，保证均匀采样
            j = random.randint(0, seen)
            if j < cap:
                pool[j] = (text, vec)
        self._seen[topic] = seen + 1


def _merge_topic_words(result: ClusterResult, members: list[int]) -> list[tuple[str, float]]:
    """按成员主题热度加权融合 BERTopic c-TF-IDF 热词。"""
    scores: dict[str, float] = {}
    total_heat = sum(result.heat_by_topic.get(m, 0) for m in members) or len(members)
    for m in members:
        weight = result.heat_by_topic.get(m, 0) / total_heat if total_heat else 0.0
        for word, score in result.topic_model.get_topic(m) or []:
            if word:
                scores[word] = scores.get(word, 0.0) + float(score) * weight
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def merge_similar_topics(result: ClusterResult, threshold: float) -> None:
    """合并近重复主题（技术优化2）：按主题质心向量余弦 ≥ threshold 用并查集合并。

    KMeans 预设簇数易把同一真实话题切成多个簇，导致 TOP 榜出现重复热点。
    每组保留热度最高的原 topic_id 作 canonical，组内热度/客户/代表池累加，
    c-TF-IDF 热词按成员主题热度加权融合。threshold<=0 关闭。
    """
    result.members_by_topic = {t: [t] for t in result.heat_by_topic}
    result.hot_words_by_topic = {
        t: _merge_topic_words(result, [t]) for t in result.heat_by_topic
    }
    if threshold <= 0:
        return
    topics = [t for t in result.heat_by_topic if result.repr_pool.get(t)]
    if len(topics) < 2:
        return

    # 归一化质心
    cent: dict[int, np.ndarray] = {}
    for t in topics:
        v = np.mean([vec for _, vec in result.repr_pool[t]], axis=0)
        cent[t] = v / (np.linalg.norm(v) + 1e-9)

    parent = {t: t for t in topics}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            a, b = topics[i], topics[j]
            if float(cent[a] @ cent[b]) >= threshold:
                parent[find(a)] = find(b)

    groups: dict[int, list[int]] = {}
    for t in topics:
        groups.setdefault(find(t), []).append(t)

    cap = get_settings().repr_pool_size
    new_heat: dict[int, int] = {}
    new_pool: dict[int, list[tuple[str, np.ndarray]]] = {}
    new_cust: dict[int, set[str]] = {}
    new_members: dict[int, list[int]] = {}
    new_words: dict[int, list[tuple[str, float]]] = {}
    for members in groups.values():
        canon = max(members, key=lambda m: result.heat_by_topic[m])  # 代表簇=热度最高
        new_heat[canon] = sum(result.heat_by_topic[m] for m in members)
        pool: list[tuple[str, np.ndarray]] = []
        custs: set[str] = set()
        for m in members:
            pool.extend(result.repr_pool.get(m, []))
            custs |= result.customers_by_topic.get(m, set())
        new_pool[canon] = pool[:cap]
        new_cust[canon] = custs
        new_members[canon] = sorted(members)
        new_words[canon] = _merge_topic_words(result, members)

    # 未参与合并（无质心/空池）的主题原样保留
    for t in result.heat_by_topic:
        if t not in cent and t not in new_heat:
            new_heat[t] = result.heat_by_topic[t]
            new_pool[t] = result.repr_pool.get(t, [])
            new_cust[t] = result.customers_by_topic.get(t, set())
            new_members[t] = [t]
            new_words[t] = _merge_topic_words(result, [t])

    result.heat_by_topic = new_heat
    result.repr_pool = new_pool
    result.customers_by_topic = new_cust
    result.members_by_topic = new_members
    result.hot_words_by_topic = new_words
    result.n_topics = len(new_heat)


def _accumulate(result: ClusterResult, batch: Batch, topics: list[int], cap: int) -> None:
    has_customer = bool(batch["customer"])
    for i, topic in enumerate(topics):
        if topic == -1:  # HDBSCAN 噪声不计入热点
            continue
        result.heat_by_topic[topic] = (
            result.heat_by_topic.get(topic, 0) + int(batch["msg_count"][i])
        )
        result._add_repr(topic, batch["text"][i], batch["vectors"][i], cap)
        if has_customer:  # 去重客户数（广度）
            result.customers_by_topic.setdefault(topic, set()).add(batch["customer"][i])


def run_clustering(date_str: str) -> ClusterResult:
    """从 parquet 分批读取 + 按 CLUSTER_BACKEND 选 fit 路径，流式累计热度与代表池。"""
    settings = get_settings()
    backend = settings.cluster_backend
    store = get_vector_store()
    cap = settings.repr_pool_size
    result = ClusterResult(topic_model=None, heat_by_topic={}, repr_pool={})  # type: ignore

    if backend == "incremental":
        topic_model = build_topic_model(build_vectorizer(online=True))
        # PASS 1：分批 partial_fit 建模（内存只占一个 batch，§6 方案 A）
        for b in store.iter_batches(date_str):
            if b["tokens"]:
                topic_model.partial_fit(b["tokens"], embeddings=b["vectors"])
        # PASS 2：分批 transform 显式打标签（技术优化2：不依赖内部顺序）
        for b in store.iter_batches(date_str):
            if not b["tokens"]:
                continue
            topics, _ = topic_model.transform(b["tokens"], embeddings=b["vectors"])
            _accumulate(result, b, [int(t) for t in topics], cap)
        result.topic_model = topic_model
        result.n_topics = len(result.heat_by_topic)
        merge_similar_topics(result, settings.topic_merge_sim)  # 合并近重复簇
        return result

    # umap_hdbscan：UMAP/HDBSCAN 是全量驻内存算法，本地 MVP 一次性 fit（§6 方案 C；
    # 大规模见方案 B：采样 fit + approximate_predict 分批推断）。
    tokens: list[str] = []
    texts: list[str] = []
    vecs: list[np.ndarray] = []
    counts: list[int] = []
    custs: list[str] = []
    for b in store.iter_batches(date_str):
        tokens.extend(b["tokens"])
        texts.extend(b["text"])
        counts.extend(b["msg_count"])
        custs.extend(b["customer"])
        vecs.extend(list(b["vectors"]))
    if not tokens:
        return result
    embeddings = np.asarray(vecs, dtype=np.float32)
    topic_model = build_topic_model(build_vectorizer(online=False))
    topics, _ = topic_model.fit_transform(tokens, embeddings=embeddings)
    pseudo = Batch(
        session_id=[], text=texts, tokens=tokens, msg_count=counts,
        customer=custs, vectors=embeddings,
    )
    _accumulate(result, pseudo, [int(t) for t in topics], cap)
    result.topic_model = topic_model
    result.n_topics = len(result.heat_by_topic)
    merge_similar_topics(result, settings.topic_merge_sim)  # 合并近重复簇
    return result
