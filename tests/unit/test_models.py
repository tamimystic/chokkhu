from __future__ import annotations

import numpy as np


from chokkhu import train
from chokkhu.models.base import ChokkhuModel


def test_linear_regression():
    X = np.array([[1], [2], [3], [4]], dtype=np.float64)
    y = np.array([2, 4, 6, 8], dtype=np.float64)

    # Test OLS
    model = train("linear_regression", X, y, verbose=False)
    assert isinstance(model, ChokkhuModel)
    preds = model.predict(X)
    assert np.allclose(preds, y, atol=1e-5)

    # Test Gradient Descent
    model_gd = train(
        "linear_regression",
        X,
        y,
        method="gradient_descent",
        learning_rate=0.05,
        epochs=2000,
        verbose=False,
    )
    preds_gd = model_gd.predict(X)
    assert np.allclose(preds_gd, y, atol=1e-1)


def test_logistic_regression():
    X = np.array([[1], [2], [10], [11]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.float64)

    model = train(
        "logistic_regression", X, y, learning_rate=0.1, epochs=1000, verbose=False
    )
    preds = model.predict(X)
    assert np.array_equal(preds, y)


def test_knn():
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.float64)

    model = train("knn", X, y, n_neighbors=2, task="classification", verbose=False)
    preds = model.predict(np.array([[1, 1.5], [10, 10.5]]))
    assert np.array_equal(preds, [0, 1])


def test_naive_bayes():
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.float64)

    model = train("naive_bayes", X, y, verbose=False)
    preds = model.predict(np.array([[1, 1.5], [10, 10.5]]))
    assert np.array_equal(preds, [0, 1])


def test_kmeans():
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]], dtype=np.float64)

    model = train("kmeans", X, n_clusters=2, random_state=42, verbose=False)
    preds = model.predict(X)
    # They should be grouped 0,0 and 1,1 (or 1,1 and 0,0)
    assert preds[0] == preds[1]
    assert preds[2] == preds[3]
    assert preds[0] != preds[2]


def test_svm():
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.float64)
    model = train("svm", X, y, epochs=1000, learning_rate=0.01, verbose=False)
    preds = model.predict(X)
    assert np.array_equal(preds, y)


def test_decision_tree():
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.float64)
    model = train("decision_tree", X, y, max_depth=2, verbose=False)
    preds = model.predict(X)
    assert np.array_equal(preds, y)


def test_random_forest():
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.float64)
    model = train("random_forest", X, y, n_estimators=5, random_state=42, verbose=False)
    preds = model.predict(X)
    assert np.array_equal(preds, y)


def test_gradient_boosting():
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]], dtype=np.float64)
    y = np.array([0, 0, 1, 1], dtype=np.float64)
    model = train(
        "gradient_boosting", X, y, n_estimators=5, random_state=42, verbose=False
    )
    preds = model.predict(X)
    assert np.array_equal(preds, y)


def test_dbscan():
    X = np.array([[1, 1], [1, 1.1], [10, 10], [10, 10.1]], dtype=np.float64)
    model = train("dbscan", X, eps=1.0, min_samples=2, verbose=False)
    # the first two should be cluster 0, next two cluster 1
    assert model.labels_[0] == model.labels_[1]
    assert model.labels_[2] == model.labels_[3]
    assert model.labels_[0] != model.labels_[2]


def test_hierarchical():
    X = np.array([[1, 1], [1, 1.1], [10, 10], [10, 10.1]], dtype=np.float64)
    model = train("hierarchical", X, n_clusters=2, verbose=False)
    assert model.labels_[0] == model.labels_[1]
    assert model.labels_[2] == model.labels_[3]
    assert model.labels_[0] != model.labels_[2]
