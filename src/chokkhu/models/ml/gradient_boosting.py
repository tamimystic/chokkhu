from __future__ import annotations

import numpy as np

from ..base import ChokkhuModel
from .decision_tree import DecisionTree


class GradientBoosting(ChokkhuModel):
    def __init__(
        self,
        task: str = "classification",
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int | None = None,
    ) -> None:
        self.task = task
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees: list[DecisionTree] = []
        self.initial_prediction: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> GradientBoosting:
        if y is None:
            raise ValueError("y cannot be None for Gradient Boosting")

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.trees = []

        try:
            from tqdm import tqdm

            iterator = tqdm(
                range(self.n_estimators), desc=f"Training GBM ({self.task})"
            )
        except ImportError:
            iterator = range(self.n_estimators)

        if self.task == "regression":
            self.initial_prediction = float(np.mean(y))
            y_pred = np.full(np.shape(y), self.initial_prediction)

            for _ in iterator:
                residuals = y - y_pred
                tree = DecisionTree(task="regression", max_depth=self.max_depth)
                tree.fit(X, residuals)

                update = tree.predict(X)
                y_pred += self.learning_rate * update
                self.trees.append(tree)

        elif self.task == "classification":
            # Optimal log-odds initialization: ln(p / (1-p))
            p_mean = np.clip(float(np.mean(y)), 1e-5, 1.0 - 1e-5)
            self.initial_prediction = float(np.log(p_mean / (1.0 - p_mean)))
            y_pred = np.full(np.shape(y), self.initial_prediction)

            for _ in iterator:
                p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -20.0, 20.0)))
                residuals = y - p

                tree = DecisionTree(task="regression", max_depth=self.max_depth)
                tree.fit(X, residuals)

                update = tree.predict(X)
                y_pred += self.learning_rate * update
                self.trees.append(tree)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise ValueError("Model is not fitted yet.")
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification.")

        y_pred = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -20.0, 20.0)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise ValueError("Model is not fitted yet.")

        y_pred = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        if self.task == "classification":
            p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -20.0, 20.0)))
            return np.where(p >= 0.5, 1, 0)

        return y_pred
