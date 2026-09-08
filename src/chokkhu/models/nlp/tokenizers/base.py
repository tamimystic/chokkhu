"""Base Tokenizer Module for Chokkhu NLP."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class BaseTokenizer(ABC):
    """Abstract Base Class for all tokenizers in Chokkhu."""

    def __init__(
        self,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        mask_token: str = "<mask>",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
    ) -> None:
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token
        self.cls_token = cls_token
        self.sep_token = sep_token

        self.special_tokens = [
            pad_token,
            unk_token,
            bos_token,
            eos_token,
            mask_token,
            cls_token,
            sep_token,
        ]

        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self._init_special_tokens()

    def _init_special_tokens(self) -> None:
        for idx, token in enumerate(self.special_tokens):
            self.token2id[token] = idx
            self.id2token[idx] = token

    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size."""
        return len(self.token2id)

    @property
    def pad_token_id(self) -> int:
        return self.token2id[self.pad_token]

    @property
    def unk_token_id(self) -> int:
        return self.token2id[self.unk_token]

    @property
    def bos_token_id(self) -> int:
        return self.token2id[self.bos_token]

    @property
    def eos_token_id(self) -> int:
        return self.token2id[self.eos_token]

    @property
    def mask_token_id(self) -> int:
        return self.token2id[self.mask_token]

    @property
    def cls_token_id(self) -> int:
        return self.token2id[self.cls_token]

    @property
    def sep_token_id(self) -> int:
        return self.token2id[self.sep_token]

    def add_token(self, token: str) -> int:
        """Add a new token to the vocabulary if not present."""
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
            return idx
        return self.token2id[token]

    @abstractmethod
    def train(self, corpus: List[str], **kwargs: Any) -> BaseTokenizer:
        """Train tokenizer on a corpus of text strings."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Convert a text string into a list of token strings."""
        pass

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        bos: bool = False,
        eos: bool = False,
    ) -> List[int]:
        """Convert a text string into token IDs."""
        tokens = self.tokenize(text)
        ids = [self.token2id.get(tok, self.unk_token_id) for tok in tokens]
        if add_special_tokens or bos:
            ids = [self.bos_token_id] + ids
        if add_special_tokens or eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Convert token IDs back to a readable string."""
        tokens = []
        for tid in token_ids:
            tok = self.id2token.get(tid, self.unk_token)
            if skip_special_tokens and tok in self.special_tokens:
                continue
            tokens.append(tok)
        return self._detokenize(tokens)

    def _detokenize(self, tokens: List[str]) -> str:
        """Default join method for tokens."""
        return " ".join(tokens)

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        add_special_tokens: bool = False,
    ) -> np.ndarray:
        """Encode a batch of texts into a 2D NumPy integer array."""
        encoded_list = [
            self.encode(text, add_special_tokens=add_special_tokens) for text in texts
        ]
        if max_length is None:
            max_length = max(len(ids) for ids in encoded_list) if encoded_list else 0

        batch_ids = []
        for ids in encoded_list:
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            if padding and len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
            batch_ids.append(ids)

        return np.asarray(batch_ids, dtype=np.int64)

    def save(self, filepath: str) -> None:
        """Save tokenizer configuration and vocabulary to JSON file."""
        data = {
            "type": self.__class__.__name__,
            "token2id": self.token2id,
            "special_tokens": self.special_tokens,
            "meta": getattr(self, "_extra_save_meta", lambda: {})(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath: str) -> BaseTokenizer:
        """Load tokenizer vocabulary from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.token2id = data["token2id"]
        self.id2token = {int(v): k for k, v in self.token2id.items()}
        if "meta" in data and hasattr(self, "_extra_load_meta"):
            self._extra_load_meta(data["meta"])
        return self
