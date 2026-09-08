"""Word-level Tokenizer with Regex Splitting and Frequency Thresholds."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, List, Optional
from .base import BaseTokenizer


class WordTokenizer(BaseTokenizer):
    """Word-level Tokenizer supporting vocab limits and minimum word frequencies."""

    def __init__(
        self,
        max_vocab_size: Optional[int] = 50000,
        min_freq: int = 1,
        lowercase: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.lowercase = lowercase
        # Matches alphanumeric words or single punctuation characters
        self.pattern = re.compile(r"\w+|[^\w\s]")

    def tokenize(self, text: str) -> List[str]:
        """Split text into words and punctuation."""
        if self.lowercase:
            text = text.lower()
        return self.pattern.findall(text)

    def train(self, corpus: List[str], **kwargs: Any) -> WordTokenizer:
        """Count words in corpus and build vocabulary."""
        counter: Counter[str] = Counter()
        for text in corpus:
            tokens = self.tokenize(text)
            counter.update(tokens)

        sorted_words = [
            word
            for word, freq in counter.most_common()
            if freq >= self.min_freq and word not in self.special_tokens
        ]

        if self.max_vocab_size is not None:
            available_slots = self.max_vocab_size - len(self.special_tokens)
            sorted_words = sorted_words[: max(0, available_slots)]

        for word in sorted_words:
            self.add_token(word)

        return self


WhitespaceTokenizer = WordTokenizer
