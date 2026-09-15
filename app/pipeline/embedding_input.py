"""按字符数截断 Embedding 输入，保留文本开头。"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class EmbeddingInput:
    def __init__(self, max_chars: int):
        if max_chars <= 0:
            raise ValueError("embedding_max_input_chars 必须大于 0")
        self.max_chars = max_chars

    def truncate(self, text: str) -> str:
        if len(text) <= self.max_chars:
            return text
        log.info("Embedding 输入截断: chars=%d -> %d", len(text), self.max_chars)
        return text[:self.max_chars]
