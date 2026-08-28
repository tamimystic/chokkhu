from __future__ import annotations

from typing import Any

import numpy as np

from ..base import ChokkhuModel
from .decision_tree import DecisionTree


class RandomForest(ChokkhuModel):
    def __init__(
        self,
        task: str = "classification",
        n_estimators: int = 10,
        max_depth: int = 100,
        min_samples_split: int = 2,
        n_features: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.task = task
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.random_state = random_state
        self.trees: list[DecisionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> RandomForest:
        if y is None:
            raise ValueError("y cannot be None for Random Forest")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.trees = []
        try:
            from tqdm import tqdm

            iterator = tqdm(
                range(self.n_estimators), desc="Building Trees (Random Forest)"
            )
        except ImportError:
            iterator = range(self.n_estimators)

        for _ in iterator:
            tree = DecisionTree(
                task=self.task,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features,
            )
            X_samp, y_samp = self._bootstrap_samples(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

        return self

    def _bootstrap_samples(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise ValueError("Model is not fitted yet.")

        predictions = np.array([tree.predict(X) for tree in self.trees])

        if self.task == "classification":
            tree_preds = np.swapaxes(predictions, 0, 1)
            predictions = np.array(
                [self._most_common_label(pred) for pred in tree_preds]
            )
        else:
            predictions = np.mean(predictions, axis=0)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise ValueError("Model is not fitted yet.")
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification.")

        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)

        classes = np.unique(predictions)
        if len(classes) == 1:
            classes = np.array([0, 1])

        n_classes = len(classes)
        n_samples = X.shape[0]
        proba = np.zeros((n_samples, n_classes), dtype=np.float64)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        for i, sample_preds in enumerate(tree_preds):
            for p in sample_preds:
                if p in class_to_idx:
                    proba[i, class_to_idx[p]] += 1.0
            total: float = float(np.sum(proba[i]))
            if total > 0:
                proba[i] /= total

        return proba

    def _most_common_label(self, y: np.ndarray) -> Any:
        unique_labels, counts = np.unique(y, return_counts=True)
        return unique_labels[np.argmax(counts)]
