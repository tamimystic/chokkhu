"""Classical NLP and Information Retrieval Subsystem (TF-IDF, BM25, Porter Stemmer) from First Principles."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np


class PorterStemmer:
    """Porter Stemming Algorithm (Martin Porter, 1980) implemented from first principles."""

    def __init__(self) -> None:
        self.b = ""
        self.k = 0
        self.k0 = 0
        self.j = 0

    def _cons(self, i: int) -> bool:
        """Return True if self.b[i] is a consonant."""
        ch = self.b[i]
        if ch in "aeiou":
            return False
        if ch == "y":
            if i == self.k0:
                return True
            else:
                return not self._cons(i - 1)
        return True

    def _m(self) -> int:
        """Measure number of consonant sequences between k0 and j."""
        n = 0
        i = self.k0
        while True:
            if i > self.j:
                return n
            if not self._cons(i):
                break
            i += 1
        i += 1
        while True:
            while True:
                if i > self.j:
                    return n
                if self._cons(i):
                    break
                i += 1
            i += 1
            n += 1
            while True:
                if i > self.j:
                    return n
                if not self._cons(i):
                    break
                i += 1
            i += 1

    def _vowelinstem(self) -> bool:
        """Return True if k0...j contains a vowel."""
        for i in range(self.k0, self.j + 1):
            if not self._cons(i):
                return True
        return False

    def _doublec(self, i: int) -> bool:
        """Return True if i and i-1 are the same consonant."""
        if i < self.k0 + 1:
            return False
        if self.b[i] != self.b[i - 1]:
            return False
        return self._cons(i)

    def _cvc(self, i: int) -> bool:
        """Return True if i-2, i-1, i has the form consonant-vowel-consonant and not ending in w, x, or y."""
        if (
            i < self.k0 + 2
            or not self._cons(i)
            or self._cons(i - 1)
            or not self._cons(i - 2)
        ):
            return False
        ch = self.b[i]
        if ch in "wxy":
            return False
        return True

    def _setto(self, s: str) -> None:
        """Set stem string."""
        self.b = self.b[: self.j + 1] + s + self.b[self.j + 1 + len(s) :]
        self.k = self.j + len(s)

    def _r(self, s: str) -> None:
        """Replace if m > 0."""
        if self._m() > 0:
            self._setto(s)

    def _step1ab(self) -> None:
        if self.b[self.k] == "s":
            if self.b.endswith("sses", self.k0, self.k + 1):
                self.k -= 2
            elif self.b.endswith("ies", self.k0, self.k + 1):
                self.k -= 2
            elif not self.b.endswith("ss", self.k0, self.k + 1) and self.b.endswith(
                "s", self.k0, self.k + 1
            ):
                self.k -= 1

        if self.b.endswith("eed", self.k0, self.k + 1):
            if self.k - 3 >= self.k0:
                self.j = self.k - 3
                if self._m() > 0:
                    self.k -= 1
        elif (
            self.b.endswith("ed", self.k0, self.k + 1) and self._vowelinstem_helper(2)
        ) or (
            self.b.endswith("ing", self.k0, self.k + 1) and self._vowelinstem_helper(3)
        ):
            if self.b.endswith("ed", self.k0, self.k + 1):
                self.k -= 2
            else:
                self.k -= 3
            if (
                self.b.endswith("at", self.k0, self.k + 1)
                or self.b.endswith("bl", self.k0, self.k + 1)
                or self.b.endswith("iz", self.k0, self.k + 1)
            ):
                self.k += 1
                self.b = self.b[: self.k] + "e" + self.b[self.k + 1 :]
            elif self._doublec(self.k):
                self.k -= 1
                ch = self.b[self.k]
                if ch in "lsz":
                    self.k += 1
            elif self._m() == 1 and self._cvc(self.k):
                self.k += 1
                self.b = self.b[: self.k] + "e" + self.b[self.k + 1 :]

    def _vowelinstem_helper(self, suffix_len: int) -> bool:
        self.j = self.k - suffix_len
        return self._vowelinstem()

    def _step1c(self) -> None:
        if self.b.endswith("y", self.k0, self.k + 1) and self._vowelinstem_helper(1):
            self.b = self.b[: self.k] + "i" + self.b[self.k + 1 :]

    def _step2(self) -> None:
        suffixes = {
            "ational": "ate",
            "tional": "tion",
            "enci": "ence",
            "anci": "ance",
            "izer": "ize",
            "bli": "ble",
            "alli": "al",
            "entli": "ent",
            "eli": "e",
            "ousli": "ous",
            "ization": "ize",
            "ation": "ate",
            "ator": "ate",
            "alism": "al",
            "iveness": "ive",
            "fulness": "ful",
            "ousness": "ous",
            "aliti": "al",
            "iviti": "ive",
            "biliti": "ble",
        }
        for suf, repl in suffixes.items():
            if self.b.endswith(suf, self.k0, self.k + 1):
                self.j = self.k - len(suf)
                if self._m() > 0:
                    self.b = self.b[: self.j + 1] + repl
                    self.k = self.j + len(repl)
                break

    def _step3(self) -> None:
        suffixes = {
            "icate": "ic",
            "ative": "",
            "alize": "al",
            "iciti": "ic",
            "ical": "ic",
            "ful": "",
            "ness": "",
        }
        for suf, repl in suffixes.items():
            if self.b.endswith(suf, self.k0, self.k + 1):
                self.j = self.k - len(suf)
                if self._m() > 0:
                    self.b = self.b[: self.j + 1] + repl
                    self.k = self.j + len(repl)
                break

    def _step4(self) -> None:
        suffixes = [
            "al",
            "ance",
            "ence",
            "er",
            "ic",
            "able",
            "ible",
            "ant",
            "ement",
            "ment",
            "ent",
            "ou",
            "ism",
            "ate",
            "iti",
            "ous",
            "ive",
            "ize",
        ]
        for suf in suffixes:
            if self.b.endswith(suf, self.k0, self.k + 1):
                self.j = self.k - len(suf)
                if self._m() > 1:
                    self.k = self.j
                break
        if self.b.endswith("sion", self.k0, self.k + 1) or self.b.endswith(
            "tion", self.k0, self.k + 1
        ):
            self.j = self.k - 3
            if self._m() > 1:
                self.k = self.j

    def _step5(self) -> None:
        self.j = self.k
        if self.b[self.k] == "e":
            a = self._m()
            if a > 1 or (a == 1 and not self._cvc(self.k - 1)):
                self.k -= 1
        if self.b[self.k] == "l" and self._doublec(self.k) and self._m() > 1:
            self.k -= 1

    def stem(self, word: str) -> str:
        """Stem a lowercase word."""
        word = word.lower().strip()
        if len(word) <= 2:
            return word
        self.b = word
        self.k = len(word) - 1
        self.k0 = 0
        self._step1ab()
        self._step1c()
        self._step2()
        self._step3()
        self._step4()
        self._step5()
        return self.b[: self.k + 1]


class TfidfVectorizer:
    """Term Frequency-Inverse Document Frequency (TF-IDF) Vectorizer from First Principles."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 1),
        max_features: Optional[int] = None,
        sublinear_tf: bool = False,
        smooth_idf: bool = True,
        norm: Optional[str] = "l2",
        lowercase: bool = True,
        token_pattern: str = r"(?u)\b\w+\b",
        stop_words: Optional[Union[str, Sequence[str], Set[str]]] = None,
    ) -> None:
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.smooth_idf = smooth_idf
        self.norm = norm
        self.lowercase = lowercase
        self.token_pattern = re.compile(token_pattern)

        if stop_words == "english":
            self.stop_words: Set[str] = {
                "a",
                "about",
                "above",
                "after",
                "again",
                "against",
                "all",
                "am",
                "an",
                "and",
                "any",
                "are",
                "as",
                "at",
                "be",
                "because",
                "been",
                "before",
                "being",
                "below",
                "between",
                "both",
                "but",
                "by",
                "can",
                "did",
                "do",
                "does",
                "doing",
                "don",
                "down",
                "during",
                "each",
                "few",
                "for",
                "from",
                "further",
                "had",
                "has",
                "have",
                "having",
                "he",
                "her",
                "here",
                "hers",
                "herself",
                "him",
                "himself",
                "his",
                "how",
                "i",
                "if",
                "in",
                "into",
                "is",
                "it",
                "its",
                "itself",
                "just",
                "me",
                "more",
                "most",
                "my",
                "myself",
                "no",
                "nor",
                "not",
                "now",
                "of",
                "off",
                "on",
                "once",
                "only",
                "or",
                "other",
                "our",
                "ours",
                "ourselves",
                "out",
                "over",
                "own",
                "s",
                "same",
                "she",
                "should",
                "so",
                "some",
                "such",
                "t",
                "than",
                "that",
                "the",
                "their",
                "theirs",
                "them",
                "themselves",
                "then",
                "there",
                "these",
                "they",
                "this",
                "those",
                "through",
                "to",
                "too",
                "under",
                "until",
                "up",
                "very",
                "was",
                "we",
                "were",
                "what",
                "when",
                "where",
                "which",
                "while",
                "who",
                "whom",
                "why",
                "will",
                "with",
                "you",
                "your",
                "yours",
                "yourself",
                "yourselves",
            }
        elif stop_words is not None:
            self.stop_words = set(stop_words)
        else:
            self.stop_words = set()

        self.vocabulary_: Dict[str, int] = {}
        self.feature_names_: List[str] = []
        self.idf_: np.ndarray = np.array([])

    def _tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        tokens = self.token_pattern.findall(text)
        if self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words]

        # Extract n-grams
        min_n, max_n = self.ngram_range
        if min_n == 1 and max_n == 1:
            return tokens

        ngrams = []
        n_tokens = len(tokens)
        for n in range(min_n, max_n + 1):
            for i in range(n_tokens - n + 1):
                ngrams.append(" ".join(tokens[i : i + n]))
        return ngrams

    def fit(self, raw_documents: Sequence[str]) -> TfidfVectorizer:
        """Fit TF-IDF vocabulary and IDF weights."""
        doc_freq: Dict[str, int] = {}
        n_docs = len(raw_documents)

        for doc in raw_documents:
            tokens = set(self._tokenize(doc))
            for tok in tokens:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1

        # Sort features alphabetically or by frequency
        sorted_terms = sorted(doc_freq.keys())
        if self.max_features is not None and len(sorted_terms) > self.max_features:
            sorted_terms = sorted(
                sorted_terms, key=lambda t: doc_freq[t], reverse=True
            )[: self.max_features]
            sorted_terms.sort()

        self.vocabulary_ = {term: idx for idx, term in enumerate(sorted_terms)}
        self.feature_names_ = sorted_terms

        # Compute IDF
        idf = np.zeros(len(self.feature_names_), dtype=np.float64)
        for idx, term in enumerate(self.feature_names_):
            df = doc_freq[term]
            if self.smooth_idf:
                idf[idx] = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
            else:
                idf[idx] = np.log(float(n_docs) / df) + 1.0

        self.idf_ = idf
        return self

    def transform(self, raw_documents: Sequence[str]) -> np.ndarray:
        """Transform documents to TF-IDF matrix (n_docs, n_features)."""
        n_docs = len(raw_documents)
        n_features = len(self.feature_names_)
        matrix = np.zeros((n_docs, n_features), dtype=np.float64)

        for doc_idx, doc in enumerate(raw_documents):
            tokens = self._tokenize(doc)
            for tok in tokens:
                if tok in self.vocabulary_:
                    feat_idx = self.vocabulary_[tok]
                    matrix[doc_idx, feat_idx] += 1.0

        # Sublinear TF
        if self.sublinear_tf:
            nonzero_mask = matrix > 0
            matrix[nonzero_mask] = 1.0 + np.log(matrix[nonzero_mask])

        # Multiply by IDF
        matrix = matrix * self.idf_[np.newaxis, :]

        # Normalize
        if self.norm == "l2":
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
        elif self.norm == "l1":
            norms = np.sum(np.abs(matrix), axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms

        return matrix

    def fit_transform(self, raw_documents: Sequence[str]) -> np.ndarray:
        """Fit and transform in a single pass."""
        return self.fit(raw_documents).transform(raw_documents)

    def get_feature_names_out(self) -> List[str]:
        """Return list of feature names."""
        return list(self.feature_names_)


class BM25Retriever:
    """Best Matching 25 (BM25Okapi / BM25+) Ranking and Retrieval Engine (Robertson & Zaragoza, 2009)."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 0.0,
        lowercase: bool = True,
        token_pattern: str = r"(?u)\b\w+\b",
    ) -> None:
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.lowercase = lowercase
        self.token_pattern = re.compile(token_pattern)

        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_lengths: np.ndarray = np.array([])
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        return self.token_pattern.findall(text)

    def fit(self, corpus: Sequence[str]) -> BM25Retriever:
        """Index the text corpus and compute document statistics."""
        self.corpus_size = len(corpus)
        lengths = []
        self.doc_freqs = []
        df: Dict[str, int] = {}

        for doc in corpus:
            tokens = self._tokenize(doc)
            lengths.append(len(tokens))
            counts: Dict[str, int] = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            self.doc_freqs.append(counts)
            for t in counts:
                df[t] = df.get(t, 0) + 1

        self.doc_lengths = np.array(lengths, dtype=np.float64)
        self.avgdl = float(np.mean(self.doc_lengths)) if self.corpus_size > 0 else 0.0

        # Lucene / Robertson IDF formulation
        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = np.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )

        return self

    def get_scores(self, query: str) -> np.ndarray:
        """Compute BM25 scores for a query across all indexed documents."""
        q_tokens = self._tokenize(query)
        scores = np.zeros(self.corpus_size, dtype=np.float64)

        if self.avgdl == 0.0:
            return scores

        doc_len_norm = 1.0 - self.b + self.b * (self.doc_lengths / self.avgdl)

        for token in q_tokens:
            if token not in self.idf:
                continue
            token_idf = self.idf[token]

            tf = np.zeros(self.corpus_size, dtype=np.float64)
            for i in range(self.corpus_size):
                if token in self.doc_freqs[i]:
                    tf[i] = self.doc_freqs[i][token]

            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * doc_len_norm
            term_score = token_idf * ((numerator / (denominator + 1e-12)) + self.delta)
            scores += term_score

        return scores

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Retrieve top-K most relevant document indices and their matching scores."""
        scores = self.get_scores(query_text)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0
        ]
