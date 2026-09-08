"""SentencePiece Style Tokenizer with Raw Whitespace Preservation."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, List
from .base import BaseTokenizer


class SentencePieceTokenizer(BaseTokenizer):
    """SentencePiece Tokenizer from Scratch with ' ' (U+2581) whitespace substitution."""

    def __init__(
        self,
        vocab_size: int = 1000,
        space_symbol: str = " ",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.target_vocab_size = vocab_size
        self.space_symbol = space_symbol

    def train(self, corpus: List[str], **kwargs: Any) -> SentencePieceTokenizer:
        """Train SentencePiece unigram/subword vocab from raw corpus."""
        subword_counter: Counter[str] = Counter()
        for text in corpus:
            # Replace whitespace sequences with space_symbol
            normalized = self.space_symbol + re.sub(
                r"\s+", self.space_symbol, text.strip()
            )
            # Count character n-grams
            for n in range(1, 6):
                for i in range(len(normalized) - n + 1):
                    subword_counter[normalized[i : i + n]] += 1

        # Add single character units first
        single_chars = sorted([k for k in subword_counter.keys() if len(k) == 1])
        for ch in single_chars:
            self.add_token(ch)

        for sub, _ in subword_counter.most_common():
            if len(self.token2id) >= self.target_vocab_size:
                break
            self.add_token(sub)

        return self

    def tokenize(self, text: str) -> List[str]:
        """Greedy forward-match subword segmentation on raw text."""
        normalized = self.space_symbol + re.sub(r"\s+", self.space_symbol, text.strip())
        tokens: List[str] = []
        i = 0
        n = len(normalized)
        while i < n:
            matched = False
            # Try longest matches first
            for length in range(min(16, n - i), 0, -1):
                sub = normalized[i : i + length]
                if sub in self.token2id:
                    tokens.append(sub)
                    i += length
                    matched = True
                    break
            if not matched:
                tokens.append(self.unk_token)
                i += 1
        return tokens

    def _detokenize(self, tokens: List[str]) -> str:
        """Reconstruct raw text replacing space_symbol with normal spaces."""
        text = "".join(tokens)
        return text.replace(self.space_symbol, " ").strip()
