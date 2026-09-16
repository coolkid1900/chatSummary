"""端到端批处理流水线入口（§4 / §5）。

每日批量主题挖掘（incremental 分批训练；客户集合和词表仍随数据增长）：
  MySQL 流式取数 → 预处理/会话聚合/PII脱敏 → bge-m3 嵌入(限流+缓存)
  → jieba 多进程分词 + 分片 parquet 落盘 → 降维+聚类(fit/transform分离)
  → c-TF-IDF 热词 → 代表文档 MMR 采样 → DeepSeek 意图概括+业务词
  → 结果写回 MySQL + Redis 缓存。

用法：
    python -m app.pipeline.run_pipeline --date 2026-06-27           # 跑指定日期
    python -m app.pipeline.run_pipeline --date 2026-06-27 --force   # 忽略已存在分片，强制重嵌入
"""
from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from app.config import get_settings
from app.obs import RunState, setup_logging
from app.pipeline.cluster import run_clustering
from app.pipeline.embedding import Embedder
from app.pipeline.hotwords import rank_topics
from app.pipeline.ingest import iter_customer_messages
from app.pipeline.intent import IntentSummarizer
from app.pipeline.persist import save_results
from app.pipeline.preprocess import SessionUnit, stream_sessions
from app.pipeline.tokenizer import tokenize_batch
from app.pipeline.trend import enrich_with_trend
from app.pipeline.vector_store import Record, get_vector_store

log = logging.getLogger("pipeline")


def _embed_and_shard(date_str: str, state: RunState) -> None:
    """流式：聚合会话 → 累积一个分片窗口 → 嵌入+分词 → 写 parquet 分片。"""
    settings = get_settings()
    store = get_vector_store()
    store.clear(date_str)  # 清场：去掉上次写一半的残留分片，保证干净起点
    embedder = Embedder()
    shard_size = settings.shard_size

    buffer: list[SessionUnit] = []
    seq = 0
    n_sessions = 0
    n_messages = 0

    def flush(units: list[SessionUnit], seq: int) -> None:
        vectors = embedder.embed_units(units)               # 限流 + 去重缓存
        tokens = tokenize_batch([u.text for u in units])    # jieba 多进程预分词
        records: list[Record] = [
            Record(
                session_id=u.session_id,
                text=u.text,
                tokens=tok,
                msg_count=u.msg_count,
                customer=u.customer,
                vector=vectors[u.session_id],
            )
            for u, tok in zip(units, tokens)
        ]
        store.write_shard(date_str, records, seq)

    for unit in stream_sessions(iter_customer_messages(date_str)):
        buffer.append(unit)
        n_sessions += 1
        n_messages += unit.msg_count
        if len(buffer) >= shard_size:
            flush(buffer, seq)
            seq += 1
            buffer = []
    if buffer:
        flush(buffer, seq)
        seq += 1

    # 所有分片写完后才写完成标记（is_complete 的唯一凭据）
    store.write_manifest(date_str, n_shards=seq, n_records=n_sessions)

    state.set(
        n_messages=n_messages,
        n_sessions=n_sessions,
        embed_api_calls=embedder.api_calls,
        embed_cache_hits=embedder.cache_hits,
    )
    log.info(
        "嵌入+分片完成: %d 会话(%d 客户消息) → %d 分片 (API %d 次, 缓存命中 %d)",
        n_sessions, n_messages, seq, embedder.api_calls, embedder.cache_hits,
    )


def run(date_str: str, force: bool, run_id: str | None = None) -> None:
    settings = get_settings()
    state = RunState(date_str, settings.cluster_backend, run_id=run_id)
    state.start()
    log.info(
        "pipeline start run_id=%s date=%s backend=%s vector_store=%s",
        state.run_id, date_str, settings.cluster_backend,
        settings.vector_store_backend,
    )
    try:
        store = get_vector_store()
        store.cleanup_expired()

        # 复用策略（技术优化1）：
        #  - 完整（向量+词库都没变）→ 跳过嵌入；
        #  - 仅词库/分词变了（向量可复用）→ 只重分词，不重嵌（省最贵的一步）；
        #  - 否则（首次/部分失败/换模型）→ 完整嵌入+分词。
        if store.is_complete(date_str) and not force:
            m = store.read_manifest(date_str) or {}
            state.set(n_sessions=m.get("n_records", 0))
            log.info("检测到完整分片(%s 会话)，跳过嵌入直接复用（--force 可强制重嵌）",
                     m.get("n_records", "?"))
        elif store.needs_retokenize(date_str) and not force:
            n = store.rewrite_tokens(date_str, tokenize_batch)
            state.set(n_sessions=n)
            log.info("词库/分词已变更，复用向量仅重分词(0 次 embedding): %d 会话", n)
        else:
            _embed_and_shard(date_str, state)

        # 降维 + 聚类 + 归类（fit/transform 分离，流式累计热度与代表池）
        result = run_clustering(date_str)
        n_assigned = sum(result._seen.values())
        n_sessions = int(result.metrics.get("n_sessions", n_assigned))
        if state.metrics["n_sessions"] == 0:
            state.set(n_sessions=n_sessions)
        if state.metrics["n_messages"] == 0 and "n_messages" in result.metrics:
            state.set(n_messages=int(result.metrics["n_messages"]))
        state.set(n_topics=result.n_topics)
        log.info("聚类: %d 主题 (%d/%d 会话已归类，%d 待归类)",
                 result.n_topics, n_assigned, n_sessions,
                 result.metrics.get("rejected_sessions", 0))
        if result.n_topics == 0:
            log.warning("无主题（可能无数据），结束。")
            state.success()
            return

        # 热词 + 热度排序
        hot_topics = rank_topics(result)
        log.info("热词+排序: TOP %d", len(hot_topics))

        # 意图概括 + 业务词筛选（代表文档 MMR 采样）。按主题并发调用 DeepSeek，
        # 并发度由 LLM 令牌桶约束在 n/s（主题数几十~上百，串行是 wall-clock 瓶颈）。
        summarizer = IntentSummarizer()
        workers = max(1, settings.llm_concurrency)
        if len(hot_topics) > 1 and workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                summaries = list(ex.map(summarizer.summarize, hot_topics))
        else:
            summaries = [summarizer.summarize(th) for th in hot_topics]

        results: list[dict] = []
        for rank, (th, (intent, business_words, repr_docs)) in enumerate(
            zip(hot_topics, summaries), start=1
        ):
            results.append({
                "date": date_str,
                "rank": rank,
                "topic_id": int(th.topic_id),
                "heat": int(th.heat),
                "customer_count": int(th.customer_count),
                "hot_words": [[w, round(s, 4)] for w, s in th.hot_words],
                "business_words": business_words,
                "customer_intent": intent,
                "representative_docs": repr_docs,
            })
        state.set(llm_failures=summarizer.failures)
        log.info("意图概括完成 (失败回退 %d 个主题)", summarizer.failures)

        # 环比 / 突增：与昨日结果对比
        surge_count = enrich_with_trend(date_str, results)
        log.info("环比/突增完成 (突增 %d 个主题)", surge_count)

        # 落库 + 缓存
        save_results(date_str, results)
        state.success()
        log.info("落库 %d 条热点, 耗时 %.1fs", len(results), time.time() - state.t0)
        for r in results[:5]:
            words = "、".join(r["business_words"][:3])
            pct = "新" if r["is_new"] else f"{r['heat_change_pct']:+.0%}"
            surge = " ⚡突增" if r["is_surge"] else ""
            log.info("  #%d heat=%d 客户%d 环比%s%s [%s] %s",
                     r["rank"], r["heat"], r["customer_count"], pct, surge, words, r["customer_intent"])
    except Exception as e:
        state.failed(repr(e))
        log.exception("pipeline failed: %s", e)
        raise


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    ap.add_argument("--force", action="store_true", help="忽略已存在分片，强制重新嵌入")
    ap.add_argument("--run-id", default=None, help="外部指定 run_id（HTTP 触发时由 API 传入）")
    args = ap.parse_args()
    run(args.date, args.force, run_id=args.run_id)


if __name__ == "__main__":
    main()
