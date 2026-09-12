"""Adaptive Boosting (AdaBoost) for Classification & Regression in Pure NumPy.

References:
- Freund & Schapire (1997): "A Decision-Theoretic Generalization of On-Line Learning" (JCSS).
- Hastie et al. (2009): "Multi-class AdaBoost" (SAMME / SAMME.R).
- Drucker (1997): "Improving Regressors using Boosting Techniques" (ICML).
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from chokkhu.models.base import ChokkhuModel
from chokkhu.models.ml.decision_tree import DecisionTree


class AdaBoostClassifier(ChokkhuModel):
    """Multi-Class AdaBoost Classifier using SAMME algorithm with Decision Stumps."""

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        max_depth: int = 1,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.estimators: List[DecisionTree] = []
        self.estimator_weights: List[float] = []
        self.classes: np.ndarray = np.array([], dtype=np.int64)
        self.n_classes: int = 0
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> AdaBoostClassifier:
        """Fit an ensemble of weighted decision stumps using SAMME multi-class boosting."""
        if y is None:
            raise ValueError("y cannot be None for AdaBoostClassifier")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        n_samples = x_arr.shape[0]

        self.classes = np.unique(y_arr)
        self.n_classes = len(self.classes)

        # Initialize uniform sample weights
        w: np.ndarray = (
            np.copy(sample_weight).astype(np.float64)
            if sample_weight is not None
            else np.ones(n_samples, dtype=np.float64) / float(n_samples)
        )
        w = w / np.sum(w)

        self.estimators = []
        self.estimator_weights = []

        for m in range(self.n_estimators):
            tree = DecisionTree(
                max_depth=self.max_depth,
                task="classification",
            )

            # Subsample data with probability proportional to weights
            sample_indices = self.rng.choice(
                n_samples, size=n_samples, replace=True, p=w
            )
            tree.fit(x_arr[sample_indices], y_arr[sample_indices])

            preds = tree.predict(x_arr)
            incorrect: np.ndarray = (preds != y_arr).astype(np.float64)

            # Weighted error rate
            w_sum: float = float(np.sum(w))
            err: float = float(np.sum(w * incorrect) / max(w_sum, 1e-12))

            # Stop if estimator is worse than random guessing
            if err >= 1.0 - (1.0 / max(self.n_classes, 2)):
                if len(self.estimators) == 0:
                    self.estimators.append(tree)
                    self.estimator_weights.append(1.0)
                break

            if err <= 1e-12:
                # Perfect estimator
                self.estimators.append(tree)
                self.estimator_weights.append(1.0)
                break

            # SAMME weight: alpha = lr * (ln((1 - err) / err) + ln(K - 1))
            alpha: float = self.learning_rate * float(
                np.log((1.0 - err) / max(err, 1e-12))
                + np.log(max(self.n_classes - 1, 1))
            )

            # Update sample weights
            w = w * np.exp(alpha * incorrect)
            w = w / np.sum(w)

            self.estimators.append(tree)
            self.estimator_weights.append(float(alpha))

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities via weighted voting."""
        x_arr = np.asarray(X, dtype=np.float64)
        n_samples = x_arr.shape[0]
        votes = np.zeros((n_samples, self.n_classes), dtype=np.float64)

        for tree, alpha in zip(self.estimators, self.estimator_weights):
            preds = tree.predict(x_arr)
            for i, p in enumerate(preds):
                class_idx = int(np.where(self.classes == p)[0][0])
                votes[i, class_idx] += alpha

        # Softmax normalization
        shifted = votes - np.max(votes, axis=1, keepdims=True)
        exp_votes = np.exp(shifted)
        return exp_votes / np.sum(exp_votes, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict multi-class labels."""
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]


class AdaBoostRegressor(ChokkhuModel):
    """AdaBoost.R2 Regressor with Decision Tree base estimators."""

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        max_depth: int = 3,
        loss: str = "linear",
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.loss = loss
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.estimators: List[DecisionTree] = []
        self.estimator_weights: List[float] = []
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> AdaBoostRegressor:
        """Fit ensemble of regression trees using AdaBoost.R2."""
        if y is None:
            raise ValueError("y cannot be None for AdaBoostRegressor")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples = x_arr.shape[0]

        w: np.ndarray = np.ones(n_samples, dtype=np.float64) / float(n_samples)
        self.estimators = []
        self.estimator_weights = []

        for m in range(self.n_estimators):
            tree = DecisionTree(
                max_depth=self.max_depth,
                task="regression",
            )
            sample_indices = self.rng.choice(
                n_samples, size=n_samples, replace=True, p=w
            )
            tree.fit(x_arr[sample_indices], y_arr[sample_indices])

            preds = tree.predict(x_arr)
            diff = np.abs(preds - y_arr)
            d_max: float = float(np.max(diff))

            if d_max == 0:
                self.estimators.append(tree)
                self.estimator_weights.append(1.0)
                break

            if self.loss == "square":
                losses = (diff / d_max) ** 2
            elif self.loss == "exponential":
                losses = 1.0 - np.exp(-diff / d_max)
            else:
                losses = diff / d_max

            avg_loss: float = float(np.sum(w * losses))
            if avg_loss >= 0.5:
                if len(self.estimators) == 0:
                    self.estimators.append(tree)
                    self.estimator_weights.append(1.0)
                break

            beta: float = avg_loss / max(1.0 - avg_loss, 1e-12)
            alpha: float = self.learning_rate * float(np.log(1.0 / max(beta, 1e-12)))

            w = w * (beta ** (1.0 - losses))
            w = w / np.sum(w)

            self.estimators.append(tree)
            self.estimator_weights.append(float(alpha))

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict weighted median across ensemble."""
        x_arr = np.asarray(X, dtype=np.float64)
        preds_all = np.array(
            [tree.predict(x_arr) for tree in self.estimators]
        )  # [M, N]
        weights = np.array(self.estimator_weights)
        weights = weights / np.sum(weights)

        # Weighted average prediction
        return np.sum(preds_all * weights[:, np.newaxis], axis=0)
