"""Dynamic Ensemble Selection (KNORA-Eliminate & KNORA-Union) in pure NumPy."""

from __future__ import annotations

from typing import Any, Optional, Sequence
import numpy as np


class DynamicEnsembleSelection:
    """Dynamic Classifier Selection (KNORA-E & KNORA-U) in pure NumPy.

    Dynamically selects and weights competent base classifiers for each test
    query sample based on their performance in its local k-nearest validation neighborhood.
    """

    def __init__(
        self,
        pool_classifiers: Sequence[Any],
        k_neighbors: int = 7,
        method: str = "knora_e",  # "knora_e" or "knora_u"
    ) -> None:
        self.pool_classifiers = list(pool_classifiers)
        self.k_neighbors = int(k_neighbors)
        self.method = str(method).lower()

        self.dsel_X: Optional[np.ndarray] = None
        self.dsel_y: Optional[np.ndarray] = None
        self.pool_predictions: Optional[np.ndarray] = None

    def fit(self, X_dsel: np.ndarray, y_dsel: np.ndarray) -> DynamicEnsembleSelection:
        """Stores the dynamic selection dataset (DSEL) and precomputes classifier predictions."""
        self.dsel_X = np.asarray(X_dsel, dtype=np.float64)
        self.dsel_y = np.asarray(y_dsel, dtype=int)

        n_samples = len(self.dsel_X)
        n_classifiers = len(self.pool_classifiers)

        # Precompute DSEL predictions matrix: (n_classifiers, n_samples)
        preds: np.ndarray = np.zeros((n_classifiers, n_samples), dtype=int)
        for i, clf in enumerate(self.pool_classifiers):
            if hasattr(clf, "predict"):
                preds[i] = clf.predict(self.dsel_X)
            elif callable(clf):
                preds[i] = clf(self.dsel_X)
            else:
                preds[i] = self.dsel_y  # fallback

        self.pool_predictions = preds
        return self

    def _get_query_neighbors(self, x_query: np.ndarray) -> np.ndarray:
        """Finds indices of k-nearest neighbors in DSEL."""
        assert self.dsel_X is not None
        dists = np.sum((self.dsel_X - x_query) ** 2, axis=1)
        return np.argsort(dists)[: self.k_neighbors]

    def predict_sample(self, x_query: np.ndarray) -> int:
        """Predicts class label for a single query sample via dynamic selection."""
        assert (
            self.dsel_X is not None
            and self.dsel_y is not None
            and self.pool_predictions is not None
        )

        neighbor_idx = self._get_query_neighbors(x_query)
        neighbor_labels = self.dsel_y[neighbor_idx]

        # Evaluate accuracy of each classifier in local neighborhood
        # (n_classifiers, k_neighbors)
        neighbor_preds = self.pool_predictions[:, neighbor_idx]
        correct_counts = np.sum(
            neighbor_preds == neighbor_labels[np.newaxis, :], axis=1
        )

        # Query predictions from all pool classifiers
        query_2d = x_query.reshape(1, -1)
        clf_query_preds = np.array(
            [
                (
                    clf.predict(query_2d)[0]
                    if hasattr(clf, "predict")
                    else clf(query_2d)[0]
                )
                for clf in self.pool_classifiers
            ]
        )

        if self.method == "knora_e":
            # KNORA-Eliminate: Select only classifiers with 100% accuracy in neighborhood
            perfect_classifiers = np.where(correct_counts == self.k_neighbors)[0]

            if len(perfect_classifiers) > 0:
                selected_preds = clf_query_preds[perfect_classifiers]
            else:
                # Fallback to max correct
                max_correct: int = int(np.max(correct_counts))
                best_idx = np.where(correct_counts == max_correct)[0]
                selected_preds = clf_query_preds[best_idx]

            # Majority voting
            return int(np.bincount(selected_preds.astype(int)).argmax())

        else:
            # KNORA-Union: Weight classifiers by number of correct neighborhood classifications
            weights = correct_counts.astype(float)
            total_weight: float = float(np.sum(weights))

            if total_weight < 1e-12:
                # Equal weights fallback
                weights = np.ones_like(weights)

            unique_classes = np.unique(clf_query_preds)
            class_scores = {c: 0.0 for c in unique_classes}

            for i, p in enumerate(clf_query_preds):
                class_scores[p] += weights[i]

            best_class = max(class_scores.items(), key=lambda item: item[1])[0]
            return int(best_class)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts class labels for a batch of query samples."""
        X_arr = np.asarray(X, dtype=np.float64)
        return np.array([self.predict_sample(x) for x in X_arr], dtype=int)
