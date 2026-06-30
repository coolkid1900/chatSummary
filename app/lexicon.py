"""词库管理：业务词典(term) 与 停用词(stopword) 落库 + 维护（§10.1 / §15）。

- 数据存 lexicon 表，供 jieba 分词使用；通过 API 维护，改动在**下次流水线**生效。
- 首次使用自动从 data/dict/*.txt 播种（保留既有种子词），之后以数据库为准。
"""
from __future__ import annotations

import hashlib
import os

from sqlalchemy import select

from app.db import get_engine, get_session
from app.models import Base, Lexicon

KINDS = ("term", "stopword", "chitchat")
_DICT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "dict")
_SEED_FILE = {"term": "bank_terms.txt", "stopword": "stopwords.txt", "chitchat": "chitchat.txt"}


def _read_seed_file(kind: str) -> list[str]:
    path = os.path.join(_DICT_DIR, _SEED_FILE[kind])
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _ensure_table() -> None:
    Base.metadata.create_all(get_engine())


def seed_if_empty() -> None:
    """某 kind 在表中无数据时，用种子文件播种（幂等）。"""
    _ensure_table()
    session = get_session()
    try:
        for kind in KINDS:
            count = session.query(Lexicon.id).filter(Lexicon.kind == kind).count()
            if count == 0:
                words = _read_seed_file(kind)
                session.add_all(
                    Lexicon(kind=kind, word=w, enabled=True) for w in dict.fromkeys(words)
                )
        session.commit()
    finally:
        session.close()


def load_words(kind: str) -> list[str]:
    """加载某 kind 的启用词（供分词器使用）；空表则先播种。"""
    seed_if_empty()
    session = get_session()
    try:
        rows = session.execute(
            select(Lexicon.word).where(Lexicon.kind == kind, Lexicon.enabled.is_(True))
        ).scalars().all()
        return list(rows)
    finally:
        session.close()


def lexicon_fingerprint() -> str:
    """启用词库内容指纹（term + stopword）。词库一变指纹就变，用于让 parquet 的
    tokens 列失效并触发「只重分词」（manifest token 指纹，见 vector_store）。"""
    terms = "\n".join(sorted(load_words("term")))
    stops = "\n".join(sorted(load_words("stopword")))
    return hashlib.md5(f"{terms}||{stops}".encode("utf-8")).hexdigest()[:16]


# ===== 维护接口（供 API 调用）=====

def list_words(kind: str, include_disabled: bool = False) -> list[dict]:
    seed_if_empty()
    session = get_session()
    try:
        stmt = select(Lexicon).where(Lexicon.kind == kind)
        if not include_disabled:
            stmt = stmt.where(Lexicon.enabled.is_(True))
        rows = session.execute(stmt.order_by(Lexicon.word)).scalars().all()
        return [{"id": r.id, "word": r.word, "enabled": r.enabled} for r in rows]
    finally:
        session.close()


def add_words(kind: str, words: list[str]) -> int:
    """新增词（已存在则重新启用），返回新增/更新条数。"""
    _ensure_table()
    session = get_session()
    added = 0
    try:
        existing = {
            r.word: r
            for r in session.execute(
                select(Lexicon).where(Lexicon.kind == kind)
            ).scalars().all()
        }
        for w in dict.fromkeys(x.strip() for x in words if x.strip()):
            if w in existing:
                if not existing[w].enabled:
                    existing[w].enabled = True
                    added += 1
            else:
                session.add(Lexicon(kind=kind, word=w, enabled=True))
                added += 1
        session.commit()
        return added
    finally:
        session.close()


def delete_word(kind: str, word: str) -> bool:
    """软禁用词（enabled=False），保留行以保存审计记录。"""
    session = get_session()
    try:
        row = session.execute(
            select(Lexicon).where(
                Lexicon.kind == kind, Lexicon.word == word, Lexicon.enabled.is_(True)
            )
        ).scalars().first()
        if not row:
            return False
        row.enabled = False
        session.commit()
        return True
    finally:
        session.close()
