"""Unit tests for Classical ML Ensembles & Advanced Clustering in Pure NumPy."""

from __future__ import annotations

import numpy as np
from chokkhu.models.ml.adaboost import AdaBoostClassifier, AdaBoostRegressor
from chokkhu.models.ml.extra_trees import ExtraTreesClassifier, ExtraTreesRegressor
from chokkhu.models.ml.kmedoids import KMedoids
from chokkhu.models.ml.optics import OPTICS


def test_adaboost_classifier():
    """Verify AdaBoostClassifier SAMME algorithm fits and predicts correctly."""
    rng = np.random.RandomState(42)
    c1 = rng.randn(30, 4) + np.array([2.0, 2.0, 2.0, 2.0])
    c2 = rng.randn(30, 4) + np.array([-2.0, -2.0, -2.0, -2.0])
    c3 = rng.randn(30, 4) + np.array([2.0, -2.0, 2.0, -2.0])

    X = np.vstack([c1, c2, c3])
    y = np.array([0] * 30 + [1] * 30 + [2] * 30)

    clf = AdaBoostClassifier(
        n_estimators=20, learning_rate=1.0, max_depth=2, random_state=42
    )
    clf.fit(X, y)

    preds = clf.predict(X)
    acc = np.mean(preds == y)
    assert acc > 0.85

    probs = clf.predict_proba(X)
    assert probs.shape == (90, 3)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(90), rtol=1e-5)


def test_adaboost_regressor():
    """Verify AdaBoost.R2 Regressor fits and predicts continuous targets."""
    rng = np.random.RandomState(42)
    X = np.linspace(-3, 3, 50).reshape(-1, 1)
    y = np.sin(X).ravel() + 0.1 * rng.randn(50)

    reg = AdaBoostRegressor(
        n_estimators=15, learning_rate=0.8, max_depth=3, random_state=42
    )
    reg.fit(X, y)

    preds = reg.predict(X)
    assert preds.shape == (50,)
    mse = float(np.mean((preds - y) ** 2))
    assert mse < 0.2


def test_extra_trees():
    """Verify ExtraTreesClassifier and ExtraTreesRegressor ensembles."""
    rng = np.random.RandomState(42)
    X = rng.randn(60, 4)
    y_cls = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

    clf = ExtraTreesClassifier(n_estimators=10, max_depth=4, random_state=42)
    clf.fit(X, y_cls)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    assert preds.shape == (60,)
    assert probs.shape == (60, 2)

    y_reg = X[:, 0] * 2.0 - X[:, 1] + 0.1 * rng.randn(60)
    reg = ExtraTreesRegressor(n_estimators=10, max_depth=4, random_state=42)
    reg.fit(X, y_reg)
    preds_reg = reg.predict(X)
    assert preds_reg.shape == (60,)
    mse = float(np.mean((preds_reg - y_reg) ** 2))
    assert mse < 1.0


def test_kmedoids():
    """Verify PAM K-Medoids algorithm discovers medoids and assigns clusters."""
    rng = np.random.RandomState(42)
    c1 = rng.randn(20, 2) + np.array([5.0, 5.0])
    c2 = rng.randn(20, 2) + np.array([-5.0, -5.0])
    X = np.vstack([c1, c2])

    kmed = KMedoids(n_clusters=2, metric="euclidean", random_state=42)
    kmed.fit(X)

    assert len(kmed.medoid_indices_) == 2
    assert kmed.medoids_.shape == (2, 2)
    assert len(kmed.labels_) == 40
    assert kmed.inertia_ > 0.0

    preds = kmed.predict(X)
    np.testing.assert_array_equal(preds, kmed.labels_)


def test_optics():
    """Verify OPTICS reachability ordering and cluster extraction."""
    rng = np.random.RandomState(42)
    c1 = rng.randn(25, 2) * 0.2 + np.array([0.0, 0.0])
    c2 = rng.randn(25, 2) * 0.2 + np.array([5.0, 5.0])
    X = np.vstack([c1, c2])

    optics = OPTICS(min_samples=5, metric="euclidean")
    optics.fit(X)

    assert len(optics.ordering_) == 50
    assert len(optics.reachability_) == 50
    assert len(optics.core_distances_) == 50
    assert len(optics.labels_) == 50
