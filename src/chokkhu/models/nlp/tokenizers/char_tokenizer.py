"""Character-level Tokenizer."""

from __future__ import annotations

from typing import Any, List
from .base import BaseTokenizer


class CharacterTokenizer(BaseTokenizer):
    """Character-level Tokenizer from First Principles."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def train(self, corpus: List[str], **kwargs: Any) -> CharacterTokenizer:
        """Build vocabulary from unique characters in corpus."""
        unique_chars = sorted(list(set("".join(corpus))))
        for char in unique_chars:
            self.add_token(char)
        return self

    def tokenize(self, text: str) -> List[str]:
        """Split text into character tokens."""
        return list(text)

    def _detokenize(self, tokens: List[str]) -> str:
        """Join characters without spaces."""
        return "".join(tokens)
