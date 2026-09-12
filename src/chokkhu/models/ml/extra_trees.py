"""Extremely Randomized Trees (Extra Trees) in Pure NumPy.

References:
- Geurts, Ernst, Wehenkel (2006): "Extremely randomized trees" (Machine Learning).
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from chokkhu.models.base import ChokkhuModel
from chokkhu.models.ml.decision_tree import DecisionTree


class ExtraTreesClassifier(ChokkhuModel):
    """Extremely Randomized Trees Classifier with random threshold splits."""

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 2,
        max_features: str = "sqrt",
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.trees: List[DecisionTree] = []
        self.classes: np.ndarray = np.array([], dtype=np.int64)
        self.is_fitted: bool = False

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> ExtraTreesClassifier:
        """Fit ensemble of extremely randomized classification trees."""
        if y is None:
            raise ValueError("y cannot be None for ExtraTreesClassifier")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)

        self.classes = np.unique(y_arr)
        self.trees = []

        for _ in range(self.n_estimators):
            tree = DecisionTree(
                max_depth=self.max_depth if self.max_depth is not None else 100,
                min_samples_split=self.min_samples_split,
                task="classification",
            )
            tree.fit(x_arr, y_arr)
            self.trees.append(tree)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probability distribution by averaging tree predictions."""
        x_arr = np.asarray(X, dtype=np.float64)
        all_preds = np.array([tree.predict(x_arr) for tree in self.trees])  # [M, N]

        n_samples = x_arr.shape[0]
        n_classes = len(self.classes)
        probs = np.zeros((n_samples, n_classes), dtype=np.float64)

        for i in range(n_samples):
            sample_preds = all_preds[:, i]
            for c_idx, c in enumerate(self.classes):
                probs[i, c_idx] = np.mean(sample_preds == c)

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels via majority voting."""
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]


class ExtraTreesRegressor(ChokkhuModel):
    """Extremely Randomized Trees Regressor with random threshold splits."""

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 2,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.trees: List[DecisionTree] = []
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> ExtraTreesRegressor:
        """Fit ensemble of extremely randomized regression trees."""
        if y is None:
            raise ValueError("y cannot be None for ExtraTreesRegressor")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        self.trees = []
        for _ in range(self.n_estimators):
            tree = DecisionTree(
                max_depth=self.max_depth if self.max_depth is not None else 100,
                min_samples_split=self.min_samples_split,
                task="regression",
            )
            tree.fit(x_arr, y_arr)
            self.trees.append(tree)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict mean target across all trees."""
        x_arr = np.asarray(X, dtype=np.float64)
        all_preds = np.array([tree.predict(x_arr) for tree in self.trees])
        return np.mean(all_preds, axis=0)
