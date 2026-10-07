from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ..base import ChokkhuModel
from .decision_tree import DecisionTree


class RandomForest(ChokkhuModel):
    def __init__(
        self,
        n_estimators: int = 10,
        criterion: str = "gini",
        max_depth: Optional[int] = 100,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Optional[Union[int, float, str]] = None,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
        warm_start: bool = False,
        class_weight: Optional[Union[Dict[Any, float], str]] = None,
        ccp_alpha: float = 0.0,
        max_samples: Optional[Union[int, float]] = None,
        task: str = "classification",
        n_features: Optional[int] = None,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = n_features if n_features is not None else max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.task = task
        self.n_features = self.max_features

        self.trees: List[DecisionTree] = []
        self.estimators_: List[DecisionTree] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self.feature_importances_: np.ndarray = np.array([])
        self.oob_score_: float = 0.0
        self.oob_decision_function_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> RandomForest:
        if y is None:
            raise ValueError("y cannot be None for Random Forest")

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        n_samples, n_features = X_arr.shape
        self.n_features_in_ = n_features

        if self.task == "auto":
            self.task = (
                "regression"
                if np.issubdtype(y_arr.dtype, np.floating)
                else "classification"
            )

        if self.task == "classification":
            self.classes_ = np.unique(y_arr)
            self.n_classes_ = len(self.classes_)
            if self.criterion not in ("gini", "entropy", "log_loss"):
                self.criterion = "gini"
        else:
            if self.criterion not in (
                "squared_error",
                "variance",
                "mse",
                "absolute_error",
                "mae",
                "friedman_mse",
            ):
                self.criterion = "squared_error"

        if not self.warm_start or not self.trees:
            self.trees = []

        if self.random_state is not None:
            np.random.seed(self.random_state)
        n_samples_draw = (
            int(math.ceil(self.max_samples * n_samples))
            if isinstance(self.max_samples, float) and 0.0 < self.max_samples <= 1.0
            else int(self.max_samples) if self.max_samples is not None else n_samples
        )

        oob_counts = np.zeros(n_samples, dtype=np.int64)
        if self.task == "classification":
            oob_preds = np.zeros((n_samples, self.n_classes_), dtype=np.float64)
        else:
            oob_preds = np.zeros(n_samples, dtype=np.float64)

        for i in range(len(self.trees), self.n_estimators):
            tree = DecisionTree(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                class_weight=self.class_weight,
                ccp_alpha=self.ccp_alpha,
                task=self.task,
                classes=self.classes_,
                random_state=None,
            )

            if self.bootstrap:
                idxs = np.random.choice(n_samples, n_samples_draw, replace=True)
                oob_mask = np.ones(n_samples, dtype=bool)
                oob_mask[idxs] = False
                X_samp, y_samp = X_arr[idxs], y_arr[idxs]
            else:
                idxs = np.arange(n_samples)
                oob_mask = np.zeros(n_samples, dtype=bool)
                X_samp, y_samp = X_arr, y_arr

            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

            if self.oob_score and self.bootstrap and np.any(oob_mask):
                oob_indices = np.where(oob_mask)[0]
                oob_counts[oob_indices] += 1
                if self.task == "classification":
                    tree_probs = tree.predict_proba(X_arr[oob_indices])
                    oob_preds[oob_indices] += tree_probs
                else:
                    tree_p = tree.predict(X_arr[oob_indices])
                    oob_preds[oob_indices] += tree_p

        self.estimators_ = self.trees

        # Feature importances: Mean of tree feature importances
        importances = [
            t.feature_importances_
            for t in self.trees
            if len(t.feature_importances_) == n_features
        ]
        if importances:
            self.feature_importances_ = np.mean(importances, axis=0)
        else:
            self.feature_importances_ = np.zeros(n_features)

        # OOB Score calculation
        if self.oob_score and self.bootstrap:
            valid_oob = oob_counts > 0
            if np.any(valid_oob):
                if self.task == "classification":
                    oob_probs = oob_preds[valid_oob] / oob_counts[valid_oob, None]
                    self.oob_decision_function_ = oob_probs
                    oob_chosen = np.argmax(oob_probs, axis=1)
                    if self.classes_ is not None:
                        pred_labels = self.classes_[oob_chosen]
                        self.oob_score_ = float(
                            np.mean(pred_labels == y_arr[valid_oob])
                        )
                else:
                    oob_vals = oob_preds[valid_oob] / oob_counts[valid_oob]
                    self.oob_decision_function_ = oob_vals
                    y_true_oob = y_arr[valid_oob]
                    ss_tot: float = float(
                        np.sum((y_true_oob - np.mean(y_true_oob)) ** 2)
                    )
                    ss_res: float = float(np.sum((y_true_oob - oob_vals) ** 2))
                    self.oob_score_ = float(1.0 - (ss_res / (ss_tot + 1e-15)))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise ValueError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)

        if self.task == "classification":
            predictions = np.array([tree.predict(X_arr) for tree in self.trees])
            tree_preds = np.swapaxes(predictions, 0, 1)
            return np.array([self._most_common_label(p) for p in tree_preds])
        else:
            preds = np.array([tree.predict(X_arr) for tree in self.trees])
            return np.mean(preds, axis=0)

    def _most_common_label(self, y: np.ndarray) -> Any:
        unique_labels, counts = np.unique(y, return_counts=True)
        return unique_labels[np.argmax(counts)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise ValueError("Model is not fitted yet.")
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification.")

        X_arr = np.asarray(X, dtype=np.float64)
        n_samples = len(X_arr)
        sum_proba: np.ndarray = np.zeros((n_samples, self.n_classes_), dtype=np.float64)
        for tree in self.trees:
            t_prob = tree.predict_proba(X_arr)
            if t_prob.shape == (n_samples, self.n_classes_):
                sum_proba += t_prob
            else:
                for i in range(n_samples):
                    for c_idx, c_val in enumerate(self.classes_ or []):
                        if tree.classes_ is not None and c_val in tree.classes_:
                            t_idx = int(np.where(tree.classes_ == c_val)[0][0])
                            if t_idx < t_prob.shape[1]:
                                sum_proba[i, c_idx] += t_prob[i, t_idx]
        return sum_proba / max(1, len(self.trees))

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.log(np.clip(probs, 1e-15, 1.0))


class RandomForestClassifier(RandomForest):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("task", "classification")
        super().__init__(**kwargs)


class RandomForestRegressor(RandomForest):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("task", "regression")
        kwargs.setdefault("criterion", "squared_error")
        kwargs.setdefault("max_features", 1.0)
        super().__init__(**kwargs)
