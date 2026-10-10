from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from chokkhu.models.base import ChokkhuModel
from chokkhu.models.ml.decision_tree import DecisionTree


class AdaBoostClassifier(ChokkhuModel):
    def __init__(
        self,
        estimator: Any = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        algorithm: str = "SAMME",
        random_state: Optional[int] = 42,
        base_estimator: Any = None,
        max_depth: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.estimator = estimator or base_estimator
        self.base_estimator = self.estimator
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.algorithm = algorithm
        self.random_state = random_state
        self.max_depth = int(max_depth)
        self.rng = np.random.RandomState(random_state)
        self.estimators: List[Any] = []
        self.estimator_weights: List[float] = []
        self.classes: np.ndarray = np.array([], dtype=np.int64)
        self.n_classes: int = 0
        self.is_fitted: bool = False

    def _make_estimator(self) -> Any:
        if self.estimator is not None:
            if hasattr(self.estimator, "clone"):
                return self.estimator.clone()
            cls = self.estimator.__class__
            params = getattr(self.estimator, "__dict__", {})
            try:
                return cls(**{k: v for k, v in params.items() if not k.startswith("_")})
            except Exception:
                pass
        return DecisionTree(max_depth=self.max_depth, task="classification")

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> AdaBoostClassifier:
        if y is None:
            raise ValueError("y cannot be None for AdaBoostClassifier")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        n_samples = x_arr.shape[0]
        self.classes = np.unique(y_arr)
        self.n_classes = len(self.classes)

        w: np.ndarray = (
            np.copy(sample_weight).astype(np.float64)
            if sample_weight is not None
            else np.ones(n_samples, dtype=np.float64) / float(n_samples)
        )
        w = w / np.sum(w)
        self.estimators, self.estimator_weights = [], []

        for _ in range(self.n_estimators):
            tree = self._make_estimator()
            sample_indices = self.rng.choice(
                n_samples, size=n_samples, replace=True, p=w
            )
            tree.fit(x_arr[sample_indices], y_arr[sample_indices])
            preds = np.asarray(tree.predict(x_arr), dtype=np.int64)
            incorrect = (preds != y_arr).astype(np.float64)
            err = float(np.sum(w * incorrect))

            if err <= 1e-10:
                self.estimators.append(tree)
                self.estimator_weights.append(1.0)
                break
            if err >= 1.0 - (1.0 / max(self.n_classes, 2)):
                if len(self.estimators) == 0:
                    self.estimators.append(tree)
                    self.estimator_weights.append(1.0)
                break

            alpha = self.learning_rate * (
                float(np.log((1.0 - err) / max(err, 1e-12)))
                + (float(np.log(self.n_classes - 1)) if self.n_classes > 2 else 0.0)
            )
            w = w * np.exp(alpha * incorrect)
            w = w / np.sum(w)
            self.estimators.append(tree)
            self.estimator_weights.append(alpha)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or not self.estimators:
            raise ValueError("AdaBoostClassifier is not fitted yet.")
        x_arr = np.asarray(X, dtype=np.float64)
        n_samples = x_arr.shape[0]
        class_scores = np.zeros((n_samples, self.n_classes), dtype=np.float64)

        for tree, alpha in zip(self.estimators, self.estimator_weights):
            preds = np.asarray(tree.predict(x_arr), dtype=np.int64)
            for c_idx, c in enumerate(self.classes):
                class_scores[:, c_idx] += alpha * (preds == c)

        exps = np.exp(class_scores - np.max(class_scores, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]


class AdaBoostRegressor(ChokkhuModel):
    def __init__(
        self,
        estimator: Any = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        max_depth: int = 3,
        loss: str = "linear",
        random_state: Optional[int] = 42,
        base_estimator: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.estimator = estimator or base_estimator
        self.base_estimator = self.estimator
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.loss = loss
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.estimators: List[Any] = []
        self.estimator_weights: List[float] = []
        self.is_fitted: bool = False

    def _make_estimator(self) -> Any:
        if self.estimator is not None:
            if hasattr(self.estimator, "clone"):
                return self.estimator.clone()
            cls = self.estimator.__class__
            params = getattr(self.estimator, "__dict__", {})
            try:
                return cls(**{k: v for k, v in params.items() if not k.startswith("_")})
            except Exception:
                pass
        return DecisionTree(max_depth=self.max_depth, task="regression")

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> AdaBoostRegressor:
        if y is None:
            raise ValueError("y cannot be None for AdaBoostRegressor")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples = x_arr.shape[0]

        w: np.ndarray = np.ones(n_samples, dtype=np.float64) / float(n_samples)
        self.estimators, self.estimator_weights = [], []

        for _ in range(self.n_estimators):
            tree = self._make_estimator()
            sample_indices = self.rng.choice(
                n_samples, size=n_samples, replace=True, p=w
            )
            tree.fit(x_arr[sample_indices], y_arr[sample_indices])
            preds = np.asarray(tree.predict(x_arr), dtype=np.float64)
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
        x_arr = np.asarray(X, dtype=np.float64)
        preds_all = np.array([tree.predict(x_arr) for tree in self.estimators])
        weights = np.array(self.estimator_weights)
        weights = weights / np.sum(weights)
        return np.sum(preds_all * weights[:, np.newaxis], axis=0)
