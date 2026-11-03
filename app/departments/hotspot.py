"""部门热点：全公司聚类一次，按部门经理名单切片统计。

依赖当天全公司运行留下的：
  - 向量目录下的聚类产物（会话→主题、主题质心，见 vector_store._TopicArtifacts）；
  - 主题目录 daily_topic_catalog（全部主题的热词、业务词、意图）；
  - 向量分片（文本、分词、消息数、客户、向量）。
不重读 MySQL、不重新嵌入、不重新全局聚类。

步骤：
  1. 每个部门实时调用名单接口取经理 id（含子部门）；名单失败的部门跳过，其余照常。
  2. 扫描一遍分片，按 (部门, 主题) 累计会话数、热度、客户、词频、向量和、代表原文。
  3. 被吞没检测：部门在某主题内的会话质心与全公司主题质心的余弦相似度低于
     DEPARTMENT_DRIFT_SIM 时，再扫描一遍取出这些会话，在部门内细分；子主题取代原主题参与排名。
  4. 按部门内各主题的词频重算 c-TF-IDF 热词；排序分同全公司，过门槛后取 TOP N。
  5. 意图用本部门代表原文重新概括；失败或关闭时沿用主题目录里的全公司意图。
  6. 与本部门昨日榜单比较环比，覆盖写入 daily_hot_topics(scope=department)。
运行情况只写日志，不建任务表。
"""
from __future__ import annotations

import json
import logging
import math
import random
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select

from app.config import get_settings
from app.db import get_session
from app.departments.roster import RosterError, fetch_managers
from app.models import DailyTopicCatalog
from app.pipeline.cluster import _topic_words
from app.pipeline.hotwords import TopicHot, rank_score
from app.pipeline.intent import IntentSummarizer
from app.pipeline.persist import save_results
from app.pipeline.trend import enrich_with_trend
from app.pipeline.vector_store import get_vector_store, session_staff

log = logging.getLogger(__name__)


class DepartmentDataMissing(RuntimeError):
    """当天没有可用的全公司聚类结果。"""


@dataclass
class _Acc:
    """某部门某主题的累计量；代表原文用蓄水池抽样，与全公司口径一致。"""
    sessions: int = 0
    heat: int = 0
    customers: set[str] = field(default_factory=set)
    terms: Counter = field(default_factory=Counter)
    vector_sum: np.ndarray | None = None
    pool: list[tuple[str, np.ndarray]] = field(default_factory=list)
    parent: int | None = None  # 细分子主题所属的全公司主题

    def add(self, text: str, tokens: str, msg_count: int, customer: str, vector: np.ndarray,
            cap: int, rng: random.Random) -> None:
        if len(self.pool) < cap:
            self.pool.append((text, vector.copy()))
        else:
            j = rng.randint(0, self.sessions)
            if j < cap:
                self.pool[j] = (text, vector.copy())
        self.sessions += 1
        self.heat += int(msg_count)
        if customer:
            self.customers.add(customer)
        self.terms.update(tokens.lower().split())
        if self.vector_sum is None:
            self.vector_sum = np.zeros(vector.shape, dtype=np.float64)
        self.vector_sum += vector


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float(a @ b) / (na * nb)


def _load_catalog(date_str: str) -> dict[int, dict]:
    session = get_session()
    try:
        rows = session.execute(
            select(DailyTopicCatalog).where(DailyTopicCatalog.stat_date == date_str)
        ).scalars().all()
        return {
            r.topic_id: {"run_id": r.run_id, "customer_intent": r.customer_intent,
                         "business_words": json.loads(r.business_words)}
            for r in rows
        }
    finally:
        session.close()


def check_ready(date_str: str) -> dict[int, dict]:
    """确认聚类产物与主题目录来自同一次全公司运行，返回主题目录。"""
    manifest = get_vector_store().read_topic_manifest(date_str)
    catalog = _load_catalog(date_str)
    if not manifest or not catalog:
        raise DepartmentDataMissing(
            f"{date_str} 没有可用的全公司聚类结果，请先运行全公司热点统计"
            f"（聚类产物随向量保留 {get_settings().vector_ttl_days} 天）")
    if {c["run_id"] for c in catalog.values()} != {manifest.get("run_id")}:
        raise DepartmentDataMissing(f"{date_str} 的聚类产物与主题目录不是同一次运行，请重跑全公司热点统计")
    return catalog


def _scan(store, date_str: str, assignments: dict[str, int], owners: dict[str, list[str]],
          visit) -> None:
    """遍历分片中属于名单经理的会话，visit(部门, 主题, 批, 行号)。"""
    for batch in store.iter_batches(date_str):
        for i, sid in enumerate(batch["session_id"]):
            topic = assignments.get(sid)
            if topic is None:
                continue
            for dept in owners.get(session_staff(sid), ()):
                visit(dept, topic, batch, i)


def _split(vectors: np.ndarray, settings) -> list[np.ndarray] | None:
    """部门内细分：k=2..max_k 中轮廓系数最高且每组都够大的划分；划不开返回 None。"""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(vectors)
    x = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    min_part = max(settings.department_min_topic_sessions, math.ceil(n * 0.1))
    best = None
    for k in range(2, settings.department_split_max_k + 1):
        if n < k * min_part:
            break
        labels = KMeans(n_clusters=k, n_init=4, random_state=settings.cluster_random_seed).fit_predict(x)
        if np.bincount(labels, minlength=k).min() < min_part:
            continue
        score = float(silhouette_score(x, labels, sample_size=min(n, 2000),
                                       random_state=settings.cluster_random_seed))
        if score > 0 and (best is None or score > best[0]):
            best = (score, labels, k)
    if best is None:
        return None
    return [np.flatnonzero(best[1] == j) for j in range(best[2])]


def _subclusters(store, date_str, assignments, owners, accs, targets, next_ids, settings, rng) -> int:
    """取出被吞没的 (部门, 主题) 会话，细分成功则用子主题替换原主题。返回细分出的子主题数。"""
    rows: dict[tuple[str, int], list[tuple]] = {key: [] for key in targets}

    def collect(dept, topic, batch, i):
        key = (dept, topic)
        if key in rows:
            rows[key].append((batch["text"][i], batch["tokens"][i], batch["msg_count"][i],
                              batch["customer"][i] if batch["customer"] else "", batch["vectors"][i].copy()))

    _scan(store, date_str, assignments, owners, collect)
    created = 0
    for (dept, topic), items in rows.items():
        parts = _split(np.asarray([item[4] for item in items]), settings)
        if not parts:
            log.info("部门 %s 主题 %d 相似度偏低但划不开，保留原主题", dept, topic)
            continue
        del accs[dept][topic]
        for part in parts:
            sub = _Acc(parent=topic)
            for index in part:
                sub.add(*items[index], settings.repr_pool_size, rng)
            accs[dept][next_ids[dept]] = sub
            next_ids[dept] += 1
            created += 1
        log.info("部门 %s 主题 %d 细分为 %d 个子主题（%s 会话）", dept, topic, len(parts),
                 "/".join(str(len(p)) for p in parts))
    return created


def _describe(summarizer: IntentSummarizer, topic: TopicHot, use_llm: bool):
    if use_llm:
        return summarizer.describe(topic)
    return "", [], summarizer._pick_docs(topic)[: summarizer.settings.nr_repr_docs], False


def _build_results(date_str, dept, topics: dict[int, _Acc], catalog, summarizer, settings) -> list[dict]:
    total = sum(acc.sessions for acc in topics.values())
    floor = max(settings.department_min_topic_sessions,
                math.ceil(total * settings.department_min_topic_share))
    words = _topic_words({t: acc.terms for t, acc in topics.items()})
    candidates = [
        TopicHot(topic_id=t, heat=acc.heat, customer_count=len(acc.customers),
                 hot_words=words.get(t, []), repr_pool=acc.pool, session_count=acc.sessions)
        for t, acc in topics.items() if acc.sessions >= floor
    ]
    candidates.sort(key=lambda th: rank_score(th.heat, th.customer_count), reverse=True)
    ranked = candidates[: settings.top_n]
    use_llm = settings.department_local_intent
    workers = max(1, min(settings.llm_concurrency, len(ranked)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        described = list(ex.map(lambda th: _describe(summarizer, th, use_llm), ranked))

    results = []
    for rank, (th, (intent, business_words, repr_docs, ok)) in enumerate(zip(ranked, described), start=1):
        parent = topics[th.topic_id].parent
        source = "department"
        if not ok:
            # 子主题沿用所属全公司主题的意图；目录缺失时保留回退文本
            entry = catalog.get(parent if parent is not None else th.topic_id)
            if entry:
                intent, business_words, source = entry["customer_intent"], entry["business_words"], "global"
            elif not intent:
                intent = f"客户关注{'、'.join(w for w, _ in th.hot_words[:3]) or '相关业务'}"
        results.append({
            "date": date_str,
            "scope_type": "department",
            "scope_id": dept,
            "rank": rank,
            "topic_id": int(th.topic_id),
            "parent_topic_id": parent,
            "heat": int(th.heat),
            "customer_count": int(th.customer_count),
            "hot_words": [[w, round(s, 4)] for w, s in th.hot_words],
            "business_words": business_words,
            "customer_intent": intent,
            "intent_source": source,
            "representative_docs": repr_docs,
        })
    return results


def run_departments(date_str: str, department_ids: list[str], run_id: str | None = None) -> dict:
    """统计一批部门的当天热点，返回各部门结果摘要（同时写日志）。调用方负责同日期互斥。"""
    settings = get_settings()
    run_id = run_id or f"dh_{uuid.uuid4().hex[:10]}"
    t0 = time.time()
    catalog = check_ready(date_str)
    report: dict[str, dict] = {}

    rosters: dict[str, set[str]] = {}
    for dept in dict.fromkeys(department_ids):
        try:
            rosters[dept] = set(fetch_managers(dept))
        except RosterError as exc:
            report[dept] = {"status": "failed", "error": str(exc)}
            log.error("部门热点 run_id=%s 部门 %s 名单获取失败，跳过：%s", run_id, dept, exc)
    owners: dict[str, list[str]] = {}
    for dept, staff in rosters.items():
        for manager in staff:
            owners.setdefault(manager, []).append(dept)

    store = get_vector_store()
    assignments = store.read_topic_assignments(date_str, set(owners))
    centroids = store.read_topic_centroids(date_str)
    accs: dict[str, dict[int, _Acc]] = {dept: {} for dept in rosters}
    rng = random.Random(settings.cluster_random_seed)
    cap = settings.repr_pool_size

    def accumulate(dept, topic, batch, i):
        acc = accs[dept].get(topic)
        if acc is None:
            acc = accs[dept][topic] = _Acc()
        acc.add(batch["text"][i], batch["tokens"][i], batch["msg_count"][i],
                batch["customer"][i] if batch["customer"] else "", batch["vectors"][i], cap, rng)

    if assignments:
        _scan(store, date_str, assignments, owners, accumulate)

    # 被吞没检测：只看会话数足够的主题，相似度分布写日志供校准阈值
    targets: set[tuple[str, int]] = set()
    for dept, topics in accs.items():
        sims = []
        for topic, acc in topics.items():
            if acc.sessions < settings.department_split_min_sessions or topic not in centroids:
                continue
            sim = _cosine(acc.vector_sum, centroids[topic])
            sims.append(sim)
            if settings.department_drift_sim > 0 and sim < settings.department_drift_sim:
                targets.add((dept, topic))
        if sims:
            log.info("部门 %s 主题质心相似度 n=%d min=%.3f p50=%.3f（阈值 %.2f）", dept, len(sims),
                     min(sims), float(np.median(sims)), settings.department_drift_sim)
    base = max([*catalog, *centroids], default=-1) + 1
    next_ids = {dept: base for dept in accs}
    splits = (_subclusters(store, date_str, assignments, owners, accs, targets, next_ids, settings, rng)
              if targets else 0)

    summarizer = IntentSummarizer()
    for dept, topics in accs.items():
        scope = ("department", dept)
        try:
            failures = summarizer.failures
            results = _build_results(date_str, dept, topics, catalog, summarizer, settings)
            enrich_with_trend(date_str, results, scope)
            save_results(date_str, results, scope=scope, run_id=run_id)
            sessions = sum(acc.sessions for acc in topics.values())
            report[dept] = {
                "status": "success" if results else "empty",
                "managers": len(rosters[dept]),
                "sessions": sessions,
                "topics": len(results),
                "split_topics": sum(1 for r in results if r["parent_topic_id"] is not None),
                "llm_failures": summarizer.failures - failures,
            }
        except Exception as exc:  # 单个部门失败不影响其他部门
            report[dept] = {"status": "failed", "error": repr(exc)[:500]}
            log.exception("部门热点 run_id=%s 部门 %s 统计失败", run_id, dept)

    summary = {"run_id": run_id, "date": date_str, "subclusters": splits,
               "duration_sec": round(time.time() - t0, 2), "departments": report}
    log.info("部门热点完成 %s", json.dumps(summary, ensure_ascii=False))
    return summary
