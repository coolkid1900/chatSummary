"""步骤 7~8：热词产出 + 热度排序（§10.2）。

c-TF-IDF 按最终分组及合并后的词频统一计算。热度 = 该主题客户消息数（§13），归类阶段已流式累计。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.config import get_settings
from app.pipeline.cluster import ClusterResult


@dataclass
class TopicHot:
    topic_id: int
    heat: int
    customer_count: int                      # 涉及去重客户数（广度）
    hot_words: list[tuple[str, float]]
    repr_pool: list[tuple[str, np.ndarray]]  # 该主题代表文档池 (脱敏原文, 向量)
    session_count: int = 0


def rank_score(heat: int, customer_count: int) -> float:
    # 排序分纳入客户广度：heat × customer_count^w，抑制单个话痨客户把冷门主题刷上榜。
    # heat 字段本身不变（仍是客户消息数），仅影响排名。
    return heat * (max(1, customer_count) ** get_settings().rank_breadth_weight)


def all_topics(result: ClusterResult, top_words: int = 10) -> list[TopicHot]:
    """全部主题按排序分降序（含每主题热词与代表文档池），用于主题目录。"""
    topics: list[TopicHot] = []
    for topic_id, heat in result.heat_by_topic.items():
        words = (
            result.hot_words_by_topic.get(topic_id)
            or (result.topic_model.get_topic(topic_id) if result.topic_model is not None else [])
            or []
        )
        hot_words = [(w, float(s)) for w, s in words[:top_words] if w]
        topics.append(
            TopicHot(
                topic_id=topic_id,
                heat=heat,
                customer_count=len(result.customers_by_topic.get(topic_id, set())),
                hot_words=hot_words,
                repr_pool=result.repr_pool.get(topic_id, []),
                session_count=result._seen.get(topic_id, 0),
            )
        )
    topics.sort(key=lambda t: rank_score(t.heat, t.customer_count), reverse=True)
    return topics


def rank_topics(result: ClusterResult, top_words: int = 10) -> list[TopicHot]:
    """按热度排序，返回 TOP N 候选（含每主题热词与代表文档池）。"""
    return all_topics(result, top_words)[: get_settings().top_n]
