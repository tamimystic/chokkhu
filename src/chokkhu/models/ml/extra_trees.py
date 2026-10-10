from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import numpy as np
from chokkhu.models.base import ChokkhuModel
from chokkhu.models.ml.decision_tree import DecisionTree

class ExtraTreesClassifier(ChokkhuModel):
    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Optional[Union[int, float, str]] = "sqrt",
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = False,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = 42,
        verbose: int = 0,
        warm_start: bool = False,
        class_weight: Optional[Union[Dict[Any, float], str]] = None,
        ccp_alpha: float = 0.0,
        max_samples: Optional[Union[int, float]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_estimators = int(n_estimators)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
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
        self.rng = np.random.RandomState(random_state)
        self.trees: List[DecisionTree] = []
        self.classes: np.ndarray = np.array([], dtype=np.int64)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> ExtraTreesClassifier:
        if y is None:
            raise ValueError("y cannot be None for ExtraTreesClassifier")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.int64)
        n_samples = x_arr.shape[0]
        self.classes = np.unique(y_arr)
        if not self.warm_start or not self.trees:
            self.trees = []

        for _ in range(len(self.trees), self.n_estimators):
            if self.bootstrap:
                size = int(self.max_samples * n_samples) if isinstance(self.max_samples, float) else (self.max_samples or n_samples)
                idx = self.rng.choice(n_samples, size=size, replace=True)
                x_sub, y_sub = x_arr[idx], y_arr[idx]
            else:
                x_sub, y_sub = x_arr, y_arr

            tree = DecisionTree(
                criterion=self.criterion,
                splitter="random",
                max_depth=self.max_depth if self.max_depth is not None else 100,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                class_weight=self.class_weight,
                ccp_alpha=self.ccp_alpha,
                random_state=self.rng.randint(0, 1000000) if self.random_state else None,
                task="classification",
            )
            tree.fit(x_sub, y_sub)
            self.trees.append(tree)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        all_preds = np.array([tree.predict(x_arr) for tree in self.trees])
        n_samples = x_arr.shape[0]
        probs = np.zeros((n_samples, len(self.classes)), dtype=np.float64)
        for i in range(n_samples):
            sample_preds = all_preds[:, i]
            for c_idx, c in enumerate(self.classes):
                probs[i, c_idx] = np.mean(sample_preds == c)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

class ExtraTreesRegressor(ChokkhuModel):
    def __init__(
        self,
        n_estimators: int = 100,
        criterion: str = "squared_error",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Optional[Union[int, float, str]] = 1.0,
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = False,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = 42,
        verbose: int = 0,
        warm_start: bool = False,
        ccp_alpha: float = 0.0,
        max_samples: Optional[Union[int, float]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_estimators = int(n_estimators)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples
        self.rng = np.random.RandomState(random_state)
        self.trees: List[DecisionTree] = []
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> ExtraTreesRegressor:
        if y is None:
            raise ValueError("y cannot be None for ExtraTreesRegressor")
        x_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        n_samples = x_arr.shape[0]
        if not self.warm_start or not self.trees:
            self.trees = []

        for _ in range(len(self.trees), self.n_estimators):
            if self.bootstrap:
                size = int(self.max_samples * n_samples) if isinstance(self.max_samples, float) else (self.max_samples or n_samples)
                idx = self.rng.choice(n_samples, size=size, replace=True)
                x_sub, y_sub = x_arr[idx], y_arr[idx]
            else:
                x_sub, y_sub = x_arr, y_arr

            tree = DecisionTree(
                criterion=self.criterion,
                splitter="random",
                max_depth=self.max_depth if self.max_depth is not None else 100,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features,
                max_leaf_nodes=self.max_leaf_nodes,
                min_impurity_decrease=self.min_impurity_decrease,
                ccp_alpha=self.ccp_alpha,
                random_state=self.rng.randint(0, 1000000) if self.random_state else None,
                task="regression",
            )
            tree.fit(x_sub, y_sub)
            self.trees.append(tree)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(X, dtype=np.float64)
        all_preds = np.array([tree.predict(x_arr) for tree in self.trees])
        return np.mean(all_preds, axis=0)
