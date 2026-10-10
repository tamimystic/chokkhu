from __future__ import annotations
from typing import Any, Optional, Union
import numpy as np
from ..base import ChokkhuModel
from .decision_tree import DecisionTree

class GradientBoosting(ChokkhuModel):
    def __init__(
        self,
        task: str = "classification",
        loss: str = "log_loss",
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 1.0,
        criterion: str = "friedman_mse",
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_depth: Optional[int] = 3,
        min_impurity_decrease: float = 0.0,
        init: Any = None,
        random_state: Optional[int] = None,
        max_features: Optional[Union[int, float, str]] = None,
        verbose: int = 0,
        max_leaf_nodes: Optional[int] = None,
        warm_start: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: Optional[int] = None,
        tol: float = 1e-4,
        ccp_alpha: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.task = task
        self.loss = loss
        self.learning_rate = float(learning_rate)
        self.n_estimators = int(n_estimators)
        self.subsample = float(subsample)
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.init = init
        self.random_state = random_state
        self.max_features = max_features
        self.verbose = verbose
        self.max_leaf_nodes = max_leaf_nodes
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha
        self.trees: list[DecisionTree] = []
        self.initial_prediction: float = 0.0
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> GradientBoosting:
        if y is None:
            raise ValueError("y cannot be None for Gradient Boosting")
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples = X_arr.shape[0]

        rng = np.random.RandomState(self.random_state)
        if not self.warm_start or not self.trees:
            self.trees = []

        val_mask = None
        if self.n_iter_no_change and 0.0 < self.validation_fraction < 1.0:
            val_size = max(1, int(n_samples * self.validation_fraction))
            perm = rng.permutation(n_samples)
            val_idx, trn_idx = perm[:val_size], perm[val_size:]
            X_val, y_val = X_arr[val_idx], y_arr[val_idx]
            X_trn, y_trn = X_arr[trn_idx], y_arr[trn_idx]
        else:
            X_trn, y_trn = X_arr, y_arr
            X_val, y_val = None, None

        n_trn = X_trn.shape[0]

        if self.task == "regression":
            self.initial_prediction = float(np.mean(y_trn))
            y_pred = np.full(n_trn, self.initial_prediction, dtype=np.float64)
            best_loss, no_improve = float("inf"), 0

            for i in range(len(self.trees), self.n_estimators):
                residuals = y_trn - y_pred
                if self.subsample < 1.0:
                    sub_n = max(1, int(n_trn * self.subsample))
                    sub_idx = rng.choice(n_trn, size=sub_n, replace=False)
                    X_sub, res_sub = X_trn[sub_idx], residuals[sub_idx]
                else:
                    X_sub, res_sub = X_trn, residuals

                tree = DecisionTree(
                    criterion="squared_error" if self.criterion == "friedman_mse" else self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                    max_features=self.max_features,
                    max_leaf_nodes=self.max_leaf_nodes,
                    min_impurity_decrease=self.min_impurity_decrease,
                    ccp_alpha=self.ccp_alpha,
                    random_state=rng.randint(0, 1000000) if self.random_state else None,
                    task="regression",
                )
                tree.fit(X_sub, res_sub)
                y_pred += self.learning_rate * tree.predict(X_trn)
                self.trees.append(tree)

                if X_val is not None and self.n_iter_no_change:
                    v_pred = self.predict(X_val)
                    v_loss = float(np.mean((y_val - v_pred) ** 2))
                    if v_loss < best_loss - self.tol:
                        best_loss, no_improve = v_loss, 0
                    else:
                        no_improve += 1
                        if no_improve >= self.n_iter_no_change:
                            break

        elif self.task == "classification":
            p_mean = np.clip(float(np.mean(y_trn)), 1e-5, 1.0 - 1e-5)
            self.initial_prediction = float(np.log(p_mean / (1.0 - p_mean)))
            y_pred = np.full(n_trn, self.initial_prediction, dtype=np.float64)
            best_loss, no_improve = float("inf"), 0

            for i in range(len(self.trees), self.n_estimators):
                p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -20.0, 20.0)))
                residuals = y_trn - p

                if self.subsample < 1.0:
                    sub_n = max(1, int(n_trn * self.subsample))
                    sub_idx = rng.choice(n_trn, size=sub_n, replace=False)
                    X_sub, res_sub = X_trn[sub_idx], residuals[sub_idx]
                else:
                    X_sub, res_sub = X_trn, residuals

                tree = DecisionTree(
                    criterion="squared_error" if self.criterion == "friedman_mse" else self.criterion,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                    max_features=self.max_features,
                    max_leaf_nodes=self.max_leaf_nodes,
                    min_impurity_decrease=self.min_impurity_decrease,
                    ccp_alpha=self.ccp_alpha,
                    random_state=rng.randint(0, 1000000) if self.random_state else None,
                    task="regression",
                )
                tree.fit(X_sub, res_sub)
                y_pred += self.learning_rate * tree.predict(X_trn)
                self.trees.append(tree)

                if X_val is not None and self.n_iter_no_change:
                    v_prob = self.predict_proba(X_val)[:, 1]
                    v_loss = -float(np.mean(y_val * np.log(np.clip(v_prob, 1e-12, 1.0)) + (1.0 - y_val) * np.log(np.clip(1.0 - v_prob, 1e-12, 1.0))))
                    if v_loss < best_loss - self.tol:
                        best_loss, no_improve = v_loss, 0
                    else:
                        no_improve += 1
                        if no_improve >= self.n_iter_no_change:
                            break

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise ValueError("Model is not fitted yet.")
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification.")
        X_arr = np.asarray(X, dtype=np.float64)
        y_pred = np.full(X_arr.shape[0], self.initial_prediction, dtype=np.float64)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X_arr)
        p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -20.0, 20.0)))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trees:
            raise ValueError("Model is not fitted yet.")
        X_arr = np.asarray(X, dtype=np.float64)
        y_pred = np.full(X_arr.shape[0], self.initial_prediction, dtype=np.float64)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X_arr)
        if self.task == "classification":
            p = 1.0 / (1.0 + np.exp(-np.clip(y_pred, -20.0, 20.0)))
            return np.where(p >= 0.5, 1, 0)
        return y_pred

class GradientBoostingClassifier(GradientBoosting):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("task", "classification")
        kwargs.setdefault("loss", "log_loss")
        super().__init__(**kwargs)

class GradientBoostingRegressor(GradientBoosting):
    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("task", "regression")
        kwargs.setdefault("loss", "squared_error")
        super().__init__(**kwargs)
