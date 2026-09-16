"""步骤 7 支撑：中文分词（§10.1 / §5 步骤7 jieba 多进程）。

默认 CountVectorizer 不切中文，必须基于 jieba 分词，并加载银行业务词典 + 停用词，
否则热词无业务意义。

性能（§5 步骤7：jieba 是隐藏 CPU 大户）：百万级分词用 `tokenize_batch` 多进程并行
预分词，把每条会话切成「空格分隔的词串」一次性算好，落进 parquet 的 tokens 列。
之后喂给 BERTopic 的 vectorizer 只做**空格切分**（轻量），既并行化了最重的 jieba，
又让 c-TF-IDF 的 documents_per_topic 复用同一套词，无需在 fit 时重复中文分词。
"""
from __future__ import annotations

from functools import lru_cache

import jieba
from sklearn.feature_extraction.text import CountVectorizer

from app.config import get_settings
from app.lexicon import load_words

# 分词逻辑版本：改变 jieba_tokenizer 的过滤/切分规则时手动 bump，
# 使已落盘的 tokens 列失效并触发重分词（配合 manifest 的 token 指纹）。
TOKENIZER_VERSION = "jieba-v1"


@lru_cache
def _load_stopwords() -> frozenset[str]:
    """从数据库加载停用词（首次自动从种子文件播种）。"""
    return frozenset(load_words("stopword"))


@lru_cache
def _ensure_user_dict() -> bool:
    """从数据库加载银行业务词典到 jieba（只需一次）。"""
    for w in load_words("term"):
        jieba.add_word(w)
    return True


def jieba_tokenizer(text: str) -> list[str]:
    _ensure_user_dict()
    stop = _load_stopwords()
    tokens = []
    for w in jieba.cut(text):
        w = w.strip()
        # 过滤停用词、单字、纯数字/标点
        if len(w) < 2 or w in stop:
            continue
        if not any("一" <= ch <= "鿿" for ch in w):
            continue
        tokens.append(w)
    return tokens


def _tok_to_str(text: str) -> str:
    return " ".join(jieba_tokenizer(text))


def tokenize_batch(texts: list[str]) -> list[str]:
    """多进程 jieba 预分词，返回每条文本的「空格分隔词串」。

    数据量小时退化为单进程，避免进程池开销。
    """
    if not texts:
        return []
    # 在父进程先把词库装进 jieba / 停用词缓存，fork 出的子进程直接继承，
    # 避免每个 worker 各查一次数据库。
    _ensure_user_dict()
    _load_stopwords()
    workers = max(1, get_settings().jieba_workers)
    if workers == 1 or len(texts) < 256:
        return [_tok_to_str(t) for t in texts]
    # 用 multiprocessing 进程池并行（jieba 是 CPU 密集，GIL 下需多进程）
    from multiprocessing import Pool

    with Pool(processes=workers) as pool:
        return pool.map(_tok_to_str, texts, chunksize=64)


def build_vectorizer(online: bool = False):
    """返回供 BERTopic 用的 vectorizer。

    文档已是空格分隔的词串（tokenize_batch 预分词），故只需空格切分。
    online=True 保留 OnlineCountVectorizer 兼容入口；当前 incremental 直接统计最终标签词频。
    """
    common = dict(tokenizer=str.split, token_pattern=None, min_df=1)
    if online:
        from bertopic.vectorizers import OnlineCountVectorizer

        return OnlineCountVectorizer(**common)
    return CountVectorizer(**common)
