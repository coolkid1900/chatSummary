"""步骤 9：DeepSeek 意图概括 + 业务热词筛选（§10.4）。

DeepSeek 兼容 OpenAI 协议。按主题逐个调用（调用次数 = 主题数，几十~几百，
天然绕开限流），仍统一走 Redis 令牌桶（§12）。

送给大模型的内容：
  - 候选业务热词（来自 c-TF-IDF 的 hot_words）；
  - nr_docs 条代表原文（§10.4 建议 8~10，截断防超长）。
要求大模型严格返回 JSON：
  - intent          ：一句话概括客户核心意图/诉求；
  - business_words  ：从候选热词中挑出真正有银行业务意义的词（剔除「能帮/看看/时候」等无意义词）。

单主题调用失败时回退为「客户关注<热词>」，不阻断整体。
"""
from __future__ import annotations

import json
import re
import threading

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.pipeline.hotwords import TopicHot
from app.pipeline.ratelimit import TokenBucket


def _mmr_select(pool: list[tuple[str, np.ndarray]], k: int, lambda_: float = 0.6) -> list[str]:
    """最大边际相关 (MMR) 选 k 条多样代表文档（§10.4 开 diversity）。

    以主题中心为相关性基准，迭代选「与中心相关、且与已选差异大」的文档，
    避免送 LLM 的样本高度雷同。pool 为 (脱敏原文, 向量)。
    """
    if not pool:
        return []
    texts = [t for t, _ in pool]
    vecs = np.asarray([v for _, v in pool], dtype=np.float32)
    # 归一化便于用点积当余弦
    norm = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    vecs = vecs / norm
    centroid = vecs.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    relevance = vecs @ centroid  # 与主题中心的相似度

    selected: list[int] = []
    candidates = set(range(len(pool)))
    k = min(k, len(pool))
    while len(selected) < k:
        best_i, best_score = None, -1e9
        for i in candidates:
            if selected:
                diversity = max(float(vecs[i] @ vecs[j]) for j in selected)
            else:
                diversity = 0.0
            score = lambda_ * float(relevance[i]) - (1 - lambda_) * diversity
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)
        candidates.discard(best_i)
    return [texts[i] for i in selected]

_PROMPT_TMPL = """你是某银行的对话分析助手。下面给出同一聊天热点主题的「候选业务热词」与若干条「客户消息原文」。
请完成两件事，并严格只返回一个 JSON 对象，不要输出多余文字：
1. intent：用一句话（40 字以内）概括这些客户的核心意图/诉求，聚焦客户关注的业务点，不要复述客服回复。
2. business_words：从「候选业务热词」中挑出真正具有银行业务意义的词（如「提前还款」「理财赎回」「信用卡分期」「公积金贷款」），剔除无业务意义的词（如「能帮」「看看」「时候」「哪个」），保持原词不改写，按重要性排序。

候选业务热词：{words}

客户消息原文：
{docs}

只返回 JSON，格式如：{{"intent": "客户咨询……", "business_words": ["提前还款", "违约金"]}}"""


class IntentSummarizer:
    def __init__(self):
        self.settings = get_settings()
        self.bucket = TokenBucket("llm", self.settings.llm_rate_per_sec)
        self._client = None
        self._lock = threading.Lock()  # 并发概括时保护 client 初始化与计数
        self.failures = 0  # LLM 概括失败的主题数（可观测性）

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            with self._lock:
                if self._client is None:
                    self._client = OpenAI(
                        base_url=self.settings.llm_base_url, api_key=self.settings.llm_api_key
                    )
        return self._client

    def _pick_docs(self, topic: TopicHot) -> list[str]:
        """从该主题代表池用 MMR 选 nr_docs 条多样代表原文（§10.4 开 diversity）。"""
        docs = _mmr_select(topic.repr_pool, self.settings.nr_docs)
        # 截断过长会话，避免 prompt 超长（§10.4 注意中文截断）
        return [d[:200] for d in docs]

    @staticmethod
    def _parse_json(content: str) -> dict:
        """容错解析大模型返回的 JSON（可能带 ```json 代码块或多余文字）。"""
        text = content.strip()
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
        return json.loads(text)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=15))
    def _call_llm(self, docs: list[str], candidate_words: list[str]) -> tuple[str, list[str]]:
        self.bucket.acquire(1)
        prompt = _PROMPT_TMPL.format(
            words="、".join(candidate_words) or "（无）",
            docs="\n".join(f"- {d}" for d in docs),
        )
        resp = self.client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
            response_format={"type": "json_object"},  # 引导 DeepSeek 输出 JSON
        )
        data = self._parse_json(resp.choices[0].message.content)
        intent = str(data.get("intent", "")).strip()
        bw = data.get("business_words", []) or []
        # 只保留确实出现在候选热词里的词，防止模型臆造
        business_words = [w for w in bw if w in candidate_words]
        return intent, business_words

    def summarize(self, topic: TopicHot) -> tuple[str, list[str], list[str]]:
        """返回 (customer_intent, business_words, representative_docs)。"""
        docs = self._pick_docs(topic)
        repr_docs = docs[: self.settings.nr_repr_docs]
        candidate_words = [w for w, _ in topic.hot_words]

        try:
            intent, business_words = self._call_llm(docs, candidate_words)
            if not intent:  # 极端情况下模型没给 intent
                intent = f"客户关注{('、'.join(candidate_words[:3])) or '相关业务'}"
            if not business_words:  # 模型没挑出词则回退取前 3 个候选
                business_words = candidate_words[:3]
        except Exception:  # 单主题失败不阻断整体
            with self._lock:
                self.failures += 1
            intent = f"（概括失败，回退）客户关注{('、'.join(candidate_words[:3])) or '相关业务'}"
            business_words = candidate_words[:3]
        return intent, business_words, repr_docs
