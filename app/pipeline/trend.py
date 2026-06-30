"""环比 / 突增（趋势）—— 与昨日结果对比。

KMeans/合并后主题 id 不跨天稳定，故用**热词 Jaccard** 把今日主题匹配到昨日主题：
  - 匹配上：算环比涨幅 heat_change_pct = (今日 - 昨日)/昨日；
  - 匹配不上：标记 is_new（新出现）。
突增 is_surge：新出现且热度达门槛，或环比涨幅 ≥ SURGE_PCT。
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from sqlalchemy import select

from app.config import get_settings
from app.db import get_session
from app.models import DailyHotTopic

_TOP_K = 10  # 用于匹配的热词数


def _top_words(hot_words: list) -> set[str]:
    return {w for w, _ in hot_words[:_TOP_K]}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _load_prev(date_str: str) -> list[dict]:
    prev_date = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    session = get_session()
    try:
        rows = session.execute(
            select(DailyHotTopic.topic_id, DailyHotTopic.hot_words, DailyHotTopic.heat).where(
                DailyHotTopic.stat_date == prev_date
            )
        ).all()
        return [
            {"topic_id": topic_id, "words": _top_words(json.loads(hw)), "heat": heat}
            for topic_id, hw, heat in rows
        ]
    finally:
        session.close()


def enrich_with_trend(date_str: str, results: list[dict]) -> int:
    """给每条结果补 prev_heat / heat_change_pct / is_new / is_surge，返回突增条数。"""
    settings = get_settings()
    prev = _load_prev(date_str)
    used_prev: set[int] = set()
    surge_count = 0

    for r in results:
        words = _top_words(r["hot_words"])
        best, best_sim = None, 0.0
        best_idx: int | None = None
        for idx, p in enumerate(prev):
            if idx in used_prev:
                continue
            s = _jaccard(words, p["words"])
            if s > best_sim:
                best_sim, best, best_idx = s, p, idx

        if best and best_idx is not None and best_sim >= settings.trend_match_sim and best["heat"] > 0:
            used_prev.add(best_idx)
            prev_heat = int(best["heat"])
            pct = round((r["heat"] - prev_heat) / prev_heat, 4)
            r["prev_heat"] = prev_heat
            r["heat_change_pct"] = pct
            r["is_new"] = False
            r["is_surge"] = bool(r["heat"] >= settings.surge_min_heat and pct >= settings.surge_pct)
        else:
            # 昨日无匹配 = 新出现的热点；达到门槛即视为突增（最强的突增信号）
            r["prev_heat"] = 0
            r["heat_change_pct"] = None
            r["is_new"] = True
            r["is_surge"] = bool(r["heat"] >= settings.surge_min_heat)

        if r["is_surge"]:
            surge_count += 1

    return surge_count
