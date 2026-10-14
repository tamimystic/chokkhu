"""Hyperdimensional Associative Memory Classifier in pure NumPy."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from chokkhu.models.base import ChokkhuModel

from .hypervector import HyperdimensionalVector


class HDCClassifier(ChokkhuModel):
    """Hyperdimensional Computing (HDC) Associative Memory Classifier in pure NumPy.

    Encodes continuous features into hyperdimensional space via Level and Item hypervectors,
    building robust class associative prototype memories with zero-gradient one-shot learning.
    """

    def __init__(
        self,
        dim: int = 5000,
        num_levels: int = 32,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self.dim = int(dim)
        self.num_levels = int(num_levels)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

        self.item_vectors: List[HyperdimensionalVector] = []
        self.level_vectors: List[HyperdimensionalVector] = []
        self.class_prototypes: Dict[int, HyperdimensionalVector] = {}
        self.feature_min_: np.ndarray = np.array([])
        self.feature_max_: np.ndarray = np.array([])

    def _init_bases(self, n_features: int) -> None:
        """Initializes orthogonal Item vectors and correlated continuous Level vectors."""
        self.item_vectors = [
            HyperdimensionalVector.random_bipolar(
                dim=self.dim, seed=int(self.rng.integers(0, 100000))
            )
            for _ in range(n_features)
        ]

        base = HyperdimensionalVector.random_bipolar(
            dim=self.dim, seed=int(self.rng.integers(0, 100000))
        ).values.copy()
        flips_per_step = self.dim // (2 * self.num_levels)

        self.level_vectors = [HyperdimensionalVector(base.copy())]
        curr_vec = base.copy()

        for _ in range(1, self.num_levels):
            flip_indices = self.rng.choice(self.dim, size=flips_per_step, replace=False)
            curr_vec[flip_indices] *= -1.0
            self.level_vectors.append(HyperdimensionalVector(curr_vec.copy()))

    def encode_sample(self, x: np.ndarray) -> HyperdimensionalVector:
        """Encodes continuous feature vector into bound-bundled hypervector."""
        x_arr = np.asarray(x, dtype=np.float32)
        n_features = len(x_arr)

        denom = np.maximum(1e-6, self.feature_max_ - self.feature_min_)
        norm_x = np.clip((x_arr - self.feature_min_) / denom, 0.0, 1.0)
        level_indices = np.clip(
            (norm_x * (self.num_levels - 1)).astype(int), 0, self.num_levels - 1
        )

        bound_pairs: List[HyperdimensionalVector] = []
        for j in range(n_features):
            item_hv = self.item_vectors[j]
            level_hv = self.level_vectors[level_indices[j]]
            bound_pairs.append(item_hv.bind(level_hv))

        return HyperdimensionalVector.bundle(bound_pairs, binarize=True)

    def fit(self, X: Any, y: Any = None) -> HDCClassifier:
        """Learns class prototype associative memory vectors from training data."""
        if y is None:
            raise ValueError("Target y must be provided for HDCClassifier.fit")
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=int).flatten()
        n_samples, n_features = X_arr.shape

        self.feature_min_ = np.min(X_arr, axis=0)
        self.feature_max_ = np.max(X_arr, axis=0)
        self._init_bases(n_features)

        classes = np.unique(y_arr)
        self.class_prototypes = {}

        for c in classes:
            class_indices = np.where(y_arr == c)[0]
            encoded_class_samples = [
                self.encode_sample(X_arr[i]) for i in class_indices
            ]
            self.class_prototypes[int(c)] = HyperdimensionalVector.bundle(
                encoded_class_samples, binarize=False
            )

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predicts class labels via maximum cosine similarity with class associative memories."""
        X_arr = np.asarray(X, dtype=np.float32)
        predictions: List[int] = []

        for x in X_arr:
            x_hv = self.encode_sample(x)
            best_class = -1
            best_sim = -float("inf")

            for c, proto in self.class_prototypes.items():
                sim = x_hv.similarity(proto)
                if sim > best_sim:
                    best_sim = sim
                    best_class = c

            predictions.append(best_class)

        return np.array(predictions, dtype=int)

    def predict_proba(self, X: Any, temperature: float = 1.0) -> np.ndarray:
        """Predict class probabilities via softmax-scaled hyperdimensional cosine similarities."""
        X_arr = np.asarray(X, dtype=np.float32)
        classes = sorted(self.class_prototypes.keys())
        n_samples = len(X_arr)
        n_classes = len(classes)
        sims: np.ndarray = np.zeros((n_samples, n_classes), dtype=np.float64)

        for i, x in enumerate(X_arr):
            x_hv = self.encode_sample(x)
            for j, c in enumerate(classes):
                sims[i, j] = x_hv.similarity(self.class_prototypes[c])

        temp = max(1e-5, float(temperature))
        scaled_sims = sims / temp
        max_s = np.max(scaled_sims, axis=1, keepdims=True)
        exp_s = np.exp(scaled_sims - max_s)
        return exp_s / np.sum(exp_s, axis=1, keepdims=True)

    def score(self, X: Any, y: Any) -> float:
        """Computes classification accuracy score."""
        preds = self.predict(X)
        return float(np.mean(preds == np.asarray(y, dtype=int).flatten()))
