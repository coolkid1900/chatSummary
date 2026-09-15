import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.config import Settings
from app.pipeline.embedding_input import EmbeddingInput


class EmbeddingInputTests(unittest.TestCase):
    def test_short_and_exact_limit_preserve_original(self):
        value = EmbeddingInput(4)
        for text in ["", "借款", " a中 "]:
            self.assertEqual(value.truncate(text), text)

    def test_unicode_counts_characters_not_bytes(self):
        self.assertEqual(EmbeddingInput(4).truncate("客户🙂贷款咨询"), "客户🙂贷")

    def test_whitespace_and_punctuation_count_toward_limit(self):
        self.assertEqual(EmbeddingInput(4).truncate("中 \n，文"), "中 \n，")

    def test_invalid_limit_fails(self):
        for limit in [0, -1]:
            with self.assertRaises(ValueError):
                EmbeddingInput(limit)
            with self.assertRaises(ValidationError):
                Settings(_env_file=None, embedding_max_input_chars=limit)

    def test_env_controls_truncation(self):
        with patch.dict("os.environ", {"EMBEDDING_MAX_INPUT_CHARS": "3"}):
            settings = Settings(_env_file=None)
            value = EmbeddingInput(settings.embedding_max_input_chars)
            self.assertEqual(value.truncate("客户贷款咨询"), "客户贷")


if __name__ == "__main__":
    unittest.main()
