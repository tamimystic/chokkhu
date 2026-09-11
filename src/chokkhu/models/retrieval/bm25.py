from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union
import math
import re
import numpy as np


def _tokenize(text: str) -> List[str]:
    """Simple alphanumeric lowercase tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Plus:
    """Okapi BM25+ Sparse Keyword Retrieval Model implemented in pure NumPy.

    BM25+ addresses the document length penalty problem in standard BM25
    by adding a lower-bound tuning parameter delta (typically 1.0) so long
    documents containing query terms are not penalized excessively.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.delta = float(delta)

        self.corpus_size: int = 0
        self.avgdl: float = 0.0
        self.doc_len: np.ndarray = np.array([])
        self.doc_freqs: List[Dict[str, int]] = []
        self.nd: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def fit(self, corpus: Sequence[Union[str, Sequence[str]]]) -> BM25Plus:
        """Indexes a corpus of documents."""
        self.corpus_size = len(corpus)
        if self.corpus_size == 0:
            raise ValueError("Corpus cannot be empty.")

        doc_lens = []
        self.doc_freqs = []
        self.nd = {}

        for doc in corpus:
            tokens = (
                _tokenize(doc) if isinstance(doc, str) else [t.lower() for t in doc]
            )
            doc_lens.append(len(tokens))

            freqs: Dict[str, int] = {}
            for token in tokens:
                freqs[token] = freqs.get(token, 0) + 1
            self.doc_freqs.append(freqs)

            for token in freqs.keys():
                self.nd[token] = self.nd.get(token, 0) + 1

        self.doc_len = np.array(doc_lens, dtype=np.float64)
        self.avgdl = float(np.mean(self.doc_len)) if len(doc_lens) > 0 else 0.0

        # Compute Robertson-Spärck Jones IDF with smoothing
        self.idf = {}
        for token, freq in self.nd.items():
            # BM25+ IDF
            idf_val = math.log((self.corpus_size + 1.0) / (freq + 0.5)) + 1.0
            self.idf[token] = idf_val

        return self

    def get_scores(self, query: Union[str, Sequence[str]]) -> np.ndarray:
        """Calculates BM25+ relevance scores for all documents in the corpus."""
        if self.corpus_size == 0:
            raise ValueError("Model must be fitted before scoring queries.")

        q_tokens = (
            _tokenize(query) if isinstance(query, str) else [t.lower() for t in query]
        )
        scores: np.ndarray = np.zeros(self.corpus_size, dtype=np.float64)

        if len(q_tokens) == 0 or self.avgdl == 0.0:
            return scores

        denom_len = 1.0 - self.b + self.b * (self.doc_len / self.avgdl)

        for token in q_tokens:
            if token not in self.nd:
                continue

            idf_val = self.idf[token]

            for i, freqs in enumerate(self.doc_freqs):
                if token in freqs:
                    tf = freqs[token]
                    term_score = idf_val * (
                        (tf * (self.k1 + 1.0)) / (tf + self.k1 * denom_len[i])
                        + self.delta
                    )
                    scores[i] += term_score

        return scores

    def search(
        self,
        query: Union[str, Sequence[str]],
        top_k: int = 10,
    ) -> Tuple[np.ndarray, List[int]]:
        """Returns top-k highest scoring documents and their indices."""
        scores = self.get_scores(query)
        top_k = min(top_k, self.corpus_size)

        if top_k == 0:
            return np.array([], dtype=np.float64), []

        top_indices = np.argsort(-scores)[:top_k]
        top_scores = scores[top_indices]
        return top_scores, top_indices.tolist()


class OkapiBM25(BM25Plus):
    """Standard Okapi BM25 Sparse Keyword Retrieval Model (delta = 0.0)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__(k1=k1, b=b, delta=0.0)
