"""Byte-Pair Encoding (BPE) Tokenizer from Scratch."""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any, Dict, List, Set, Tuple
from .base import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    """Byte-Pair Encoding (BPE) Subword Tokenizer from First Principles (Sennrich et al., 2016)."""

    def __init__(
        self,
        vocab_size: int = 1000,
        end_of_word: str = "</w>",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.target_vocab_size = vocab_size
        self.end_of_word = end_of_word
        self.merges: Dict[Tuple[str, str], int] = {}
        self.pattern = re.compile(r"\w+|[^\w\s]")

    def _get_stats(
        self, vocab: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent symbol pairs."""
        pairs: Dict[Tuple[str, str], int] = defaultdict(int)
        for word_tuple, freq in vocab.items():
            for i in range(len(word_tuple) - 1):
                pair = (word_tuple[i], word_tuple[i + 1])
                pairs[pair] += freq
        return pairs

    def _merge_vocab(
        self,
        pair: Tuple[str, str],
        vocab: Dict[Tuple[str, ...], int],
    ) -> Dict[Tuple[str, ...], int]:
        """Merge all occurrences of pair in vocabulary words."""
        new_vocab = {}
        p0, p1 = pair
        for word_tuple, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word_tuple):
                if (
                    i < len(word_tuple) - 1
                    and word_tuple[i] == p0
                    and word_tuple[i + 1] == p1
                ):
                    new_word.append(p0 + p1)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    def train(self, corpus: List[str], **kwargs: Any) -> BPETokenizer:
        """Train BPE merge rules on the provided corpus."""
        # Step 1: Count word occurrences and initialize word representations as character tuples
        word_counts: Counter[str] = Counter()
        for text in corpus:
            words = self.pattern.findall(text.lower())
            word_counts.update(words)

        vocab: Dict[Tuple[str, ...], int] = {}
        all_chars: Set[str] = set()
        for word, count in word_counts.items():
            chars = tuple(list(word) + [self.end_of_word])
            vocab[chars] = count
            for ch in chars:
                all_chars.add(ch)

        # Add initial base characters to token vocabulary
        for ch in sorted(list(all_chars)):
            self.add_token(ch)

        # Step 2: Iteratively merge the most frequent adjacent pairs
        num_merges = max(0, self.target_vocab_size - len(self.token2id))
        self.merges = {}

        for i in range(num_merges):
            stats = self._get_stats(vocab)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            vocab = self._merge_vocab(best_pair, vocab)
            self.merges[best_pair] = i
            merged_token = best_pair[0] + best_pair[1]
            self.add_token(merged_token)

        return self

    def _tokenize_word(self, word: str) -> List[str]:
        """Apply learned BPE merge rules to segment a single word."""
        symbols = list(word) + [self.end_of_word]
        if not self.merges:
            return symbols

        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            # Find candidate pairs present in merge dictionary
            valid_pairs = [p for p in pairs if p in self.merges]
            if not valid_pairs:
                break
            best_pair = min(valid_pairs, key=lambda p: self.merges[p])
            p0, p1 = best_pair

            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == p0 and symbols[i + 1] == p1:
                    new_symbols.append(p0 + p1)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into BPE subwords."""
        words = self.pattern.findall(text.lower())
        tokens: List[str] = []
        for w in words:
            subwords = self._tokenize_word(w)
            tokens.extend(subwords)
        return tokens

    def _detokenize(self, tokens: List[str]) -> str:
        """Reconstruct string from BPE tokens."""
        text = "".join(tokens)
        return text.replace(self.end_of_word, " ").strip()

    def _extra_save_meta(self) -> Dict[str, Any]:
        return {
            "merges": [list(p) for p in self.merges.keys()],
            "end_of_word": self.end_of_word,
        }

    def _extra_load_meta(self, meta: Dict[str, Any]) -> None:
        self.end_of_word = meta.get("end_of_word", "</w>")
        merges_list = meta.get("merges", [])
        self.merges = {(p[0], p[1]): idx for idx, p in enumerate(merges_list)}
