from __future__ import annotations

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


def train(
    model: str,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    task: str = "auto",
    random_state: int | None = None,
    verbose: bool = True,
    **kwargs,
) -> ChokkhuModel:
    if verbose:
        Logger.info(f"Training model: {model} (task: {task})")

    model_obj: ChokkhuModel | None = None

    if model == "linear_regression":
        model_obj = LinearRegression(**kwargs)
    elif model == "ridge":
        model_obj = LinearRegression(
            method="gradient_descent", regularization="ridge", **kwargs
        )
    elif model == "lasso":
        model_obj = LinearRegression(
            method="gradient_descent", regularization="lasso", **kwargs
        )
    elif model == "elastic_net":
        model_obj = LinearRegression(
            method="gradient_descent", regularization="elastic_net", **kwargs
        )
    elif model == "logistic_regression":
        model_obj = LogisticRegression(**kwargs)
    elif model == "knn":
        model_obj = KNN(task=task if task != "auto" else "classification", **kwargs)
    elif model == "naive_bayes":
        model_obj = NaiveBayes(**kwargs)
    elif model == "kmeans":
        model_obj = KMeans(random_state=random_state, **kwargs)
    elif model == "dbscan":
        model_obj = DBSCAN(**kwargs)
    elif model == "hierarchical":
        model_obj = HierarchicalClustering(**kwargs)
    elif model == "svm":
        model_obj = SVM(**kwargs)
    elif model == "decision_tree":
        model_obj = DecisionTree(
            task=task if task != "auto" else "classification", **kwargs
        )
    elif model == "random_forest":
        model_obj = RandomForest(
            task=task if task != "auto" else "classification",
            random_state=random_state,
            **kwargs,
        )
    elif model == "gradient_boosting":
        model_obj = GradientBoosting(
            task=task if task != "auto" else "classification",
            random_state=random_state,
            **kwargs,
        )
    elif model in ("neural_network", "mlp", "dense"):
        model_obj = NeuralNetwork(
            task=task,
            random_state=random_state,
            **kwargs,
        )
    elif model == "q_learning":
        model_obj = QLearning(random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Model {model} is not supported yet.")

    if isinstance(X_train, (pd.DataFrame, pd.Series)):
        X_train = X_train.values
    if y_train is not None and isinstance(y_train, (pd.DataFrame, pd.Series)):
        y_train = y_train.values

    if X_train is not None and not isinstance(X_train, np.ndarray):
        X_train = np.array(X_train)
    if y_train is not None and not isinstance(y_train, np.ndarray):
        y_train = np.array(y_train)

    model_obj.fit(X_train, y_train)

    if verbose:
        Logger.info(f"Successfully trained {model}")

    return model_obj
