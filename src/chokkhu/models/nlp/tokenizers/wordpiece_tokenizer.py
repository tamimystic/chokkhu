"""WordPiece Subword Tokenizer from Scratch (BERT Style)."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any, List
from .base import BaseTokenizer


class WordPieceTokenizer(BaseTokenizer):
    """WordPiece Subword Tokenizer with greedy prefix matching and '##' subword continuation."""

    def __init__(
        self,
        vocab_size: int = 1000,
        subword_prefix: str = "##",
        max_input_chars_per_word: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.target_vocab_size = vocab_size
        self.subword_prefix = subword_prefix
        self.max_input_chars_per_word = max_input_chars_per_word
        self.pattern = re.compile(r"\w+|[^\w\s]")

    def train(self, corpus: List[str], **kwargs: Any) -> WordPieceTokenizer:
        """Build WordPiece vocabulary from corpus."""
        word_counts: Counter[str] = Counter()
        char_counts: Counter[str] = Counter()

        for text in corpus:
            words = self.pattern.findall(text.lower())
            word_counts.update(words)
            for w in words:
                for i, ch in enumerate(w):
                    if i == 0:
                        char_counts[ch] += 1
                    else:
                        char_counts[self.subword_prefix + ch] += 1

        for token in sorted(char_counts.keys()):
            self.add_token(token)

        # Add most frequent n-grams/words up to target vocab size
        candidate_subwords: Counter[str] = Counter()
        for word, count in word_counts.items():
            for length in range(2, len(word) + 1):
                for start in range(len(word) - length + 1):
                    sub = word[start : start + length]
                    if start > 0:
                        sub = self.subword_prefix + sub
                    candidate_subwords[sub] += count

        for sub, _ in candidate_subwords.most_common():
            if len(self.token2id) >= self.target_vocab_size:
                break
            self.add_token(sub)

        return self

    def _tokenize_word(self, word: str) -> List[str]:
        """Greedy longest-match subword segmentation."""
        if len(word) > self.max_input_chars_per_word:
            return [self.unk_token]

        is_bad = False
        start = 0
        sub_tokens: List[str] = []

        while start < len(word):
            end = len(word)
            cur_substr = None
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = self.subword_prefix + substr
                if substr in self.token2id:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                is_bad = True
                break
            sub_tokens.append(cur_substr)
            start = end

        if is_bad:
            return [self.unk_token]
        return sub_tokens

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into WordPiece subwords."""
        words = self.pattern.findall(text.lower())
        tokens: List[str] = []
        for w in words:
            tokens.extend(self._tokenize_word(w))
        return tokens

    def _detokenize(self, tokens: List[str]) -> str:
        """Reconstruct string from WordPiece tokens."""
        out = []
        for tok in tokens:
            if tok.startswith(self.subword_prefix):
                out.append(tok[len(self.subword_prefix) :])
            else:
                if out:
                    out.append(" ")
                out.append(tok)
        return "".join(out)
