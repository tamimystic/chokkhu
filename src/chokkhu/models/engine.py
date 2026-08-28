from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from chokkhu.core.logger import Logger
from .base import ChokkhuModel
from .ml import (
    DBSCAN,
    KNN,
    SVM,
    DecisionTree,
    GradientBoosting,
    HierarchicalClustering,
    KMeans,
    LinearRegression,
    LogisticRegression,
    NaiveBayes,
    RandomForest,
    NeuralNetwork,
)
from .rl import QLearning


def _get_default_param_grid(model_name: str) -> Dict[str, List[Any]]:
    """Returns candidate hyperparameter search grids."""
    if model_name in ("random_forest", "rf"):
        return {"n_estimators": [10, 20], "max_depth": [3, 5, 10]}
    elif model_name in ("decision_tree", "dt"):
        return {"max_depth": [3, 5, 10], "min_samples_split": [2, 5]}
    elif model_name in ("knn",):
        return {"k": [3, 5, 7]}
    elif model_name in ("gradient_boosting", "gbm"):
        return {"n_estimators": [10, 20], "learning_rate": [0.05, 0.1]}
    elif model_name in ("neural_network", "mlp"):
        return {"learning_rate": [0.01, 0.05], "epochs": [30, 50]}
    elif model_name in ("ridge", "lasso"):
        return {"learning_rate": [0.001, 0.01, 0.1]}
    return {}


def _create_model_instance(
    model: str, task: str, random_state: Optional[int], **kwargs
) -> ChokkhuModel:
    if model == "linear_regression":
        return LinearRegression(**kwargs)
    elif model == "ridge":
        return LinearRegression(
            method="gradient_descent", regularization="ridge", **kwargs
        )
    elif model == "lasso":
        return LinearRegression(
            method="gradient_descent", regularization="lasso", **kwargs
        )
    elif model == "elastic_net":
        return LinearRegression(
            method="gradient_descent", regularization="elastic_net", **kwargs
        )
    elif model == "logistic_regression":
        return LogisticRegression(**kwargs)
    elif model == "knn":
        return KNN(task=task if task != "auto" else "classification", **kwargs)
    elif model == "naive_bayes":
        return NaiveBayes(**kwargs)
    elif model == "kmeans":
        return KMeans(random_state=random_state, **kwargs)
    elif model == "dbscan":
        return DBSCAN(**kwargs)
    elif model == "hierarchical":
        return HierarchicalClustering(**kwargs)
    elif model == "svm":
        return SVM(**kwargs)
    elif model == "decision_tree":
        return DecisionTree(task=task if task != "auto" else "classification", **kwargs)
    elif model in ("random_forest", "rf"):
        return RandomForest(
            task=task if task != "auto" else "classification",
            random_state=random_state,
            **kwargs,
        )
    elif model in ("gradient_boosting", "gbm"):
        return GradientBoosting(
            task=task if task != "auto" else "classification",
            random_state=random_state,
            **kwargs,
        )
    elif model in ("neural_network", "mlp", "dense"):
        return NeuralNetwork(
            task=task,
            random_state=random_state,
            **kwargs,
        )
    elif model == "q_learning":
        return QLearning(random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Model {model} is not supported yet.")


def train(
    model: str,
    X_train: Any = None,
    y_train: Any = None,
    task: str = "auto",
    random_state: int | None = None,
    tune: bool = False,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    cv: int = 3,
    verbose: bool = True,
    **kwargs,
) -> ChokkhuModel:
    if verbose:
        Logger.info(f"Training model: {model} (task: {task})")

    if isinstance(X_train, (pd.DataFrame, pd.Series)):
        X_train = X_train.values
    if y_train is not None and isinstance(y_train, (pd.DataFrame, pd.Series)):
        y_train = y_train.values

    if X_train is not None and not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    if y_train is not None and not isinstance(y_train, np.ndarray):
        y_train = np.array(y_train)

    best_params: Dict[str, Any] = {}
    if tune and X_train is not None and y_train is not None:
        grid = param_grid or _get_default_param_grid(model)
        if grid:
            if verbose:
                Logger.info(
                    "Hyperparameter tuning enabled. Evaluating candidate configurations..."
                )
            keys = list(grid.keys())
            combinations = [
                dict(zip(keys, v)) for v in itertools.product(*grid.values())
            ]

            n_samples = len(X_train)
            indices = np.arange(n_samples)
            if random_state is not None:
                np.random.seed(random_state)
            np.random.shuffle(indices)

            best_score = float("-inf")
            best_config = kwargs

            # Simple K-Fold CV search
            fold_size = max(1, n_samples // cv)
            for cfg in combinations:
                merged_kwargs = {**kwargs, **cfg}
                scores = []
                for f_idx in range(cv):
                    val_idx = indices[f_idx * fold_size : (f_idx + 1) * fold_size]
                    tr_idx = np.setdiff1d(indices, val_idx)

                    m = _create_model_instance(
                        model, task, random_state, **merged_kwargs
                    )
                    m.fit(X_train[tr_idx], y_train[tr_idx])
                    preds = m.predict(X_train[val_idx])

                    if task == "regression" or np.issubdtype(
                        y_train.dtype, np.floating
                    ):
                        ss_res: float = float(np.sum((y_train[val_idx] - preds) ** 2))
                        ss_tot: float = float(
                            np.sum((y_train[val_idx] - np.mean(y_train[val_idx])) ** 2)
                        )
                        score = 1.0 - (ss_res / max(1e-9, ss_tot))
                    else:
                        score = float(np.mean(y_train[val_idx] == preds))
                    scores.append(score)

                mean_score = float(np.mean(scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_config = merged_kwargs
                    best_params = cfg

            kwargs = best_config
            if verbose:
                Logger.info(
                    f"Tuning complete. Best Score: {best_score:.4f}, Best Params: {best_params}"
                )

    model_obj = _create_model_instance(model, task, random_state, **kwargs)
    model_obj.fit(X_train, y_train)

    if best_params:
        setattr(model_obj, "best_params_", best_params)

    if verbose:
        Logger.info(f"Successfully trained {model}")

    return model_obj
