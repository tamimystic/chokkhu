"""Unit tests for Phase 7 Classical ML: IsolationForest, HistGB, SVM kernels, NearestCentroid, RadiusNeighbors."""

from __future__ import annotations

import numpy as np
from chokkhu.models.ml.isolation_forest import IsolationForest
from chokkhu.models.ml.hist_gradient_boosting import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from chokkhu.models.ml.svm_kernels import SVC, SVR, OneClassSVM, LinearSVC
from chokkhu.models.ml.nearest_neighbors import (
    NearestCentroid,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)


def test_isolation_forest():
    """Verify IsolationForest detects injected anomalous outliers."""
    rng = np.random.RandomState(42)
    # Dense cluster of normal instances
    X_inliers = rng.randn(100, 2) * 0.5
    # Distant outliers
    X_outliers = np.array([[10.0, 10.0], [-10.0, -10.0], [12.0, -12.0]])
    X = np.vstack([X_inliers, X_outliers])

    iso = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
    iso.fit(X)

    scores = iso.score_samples(X)
    preds = iso.predict(X)

    assert len(scores) == 103
    assert len(preds) == 103
    # Outliers should have higher anomaly scores and receive -1 label
    assert scores[-1] > scores[0]
    assert np.all(preds[-3:] == -1)


def test_hist_gradient_boosting():
    """Verify HistGradientBoostingClassifier and HistGradientBoostingRegressor."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    y_cls = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

    hgb_cls = HistGradientBoostingClassifier(max_iter=20, max_bins=64, random_state=42)
    hgb_cls.fit(X, y_cls)
    preds_cls = hgb_cls.predict(X)
    probs_cls = hgb_cls.predict_proba(X)

    assert preds_cls.shape == (100,)
    assert probs_cls.shape == (100, 2)
    np.testing.assert_allclose(np.sum(probs_cls, axis=1), np.ones(100), rtol=1e-5)

    y_reg = X[:, 0] * 3.0 - X[:, 1] * 2.0 + 0.1 * rng.randn(100)
    hgb_reg = HistGradientBoostingRegressor(max_iter=50, max_bins=64, random_state=42)
    hgb_reg.fit(X, y_reg)
    preds_reg = hgb_reg.predict(X)

    assert preds_reg.shape == (100,)
    mse = float(np.mean((preds_reg - y_reg) ** 2))
    assert mse < 1.0


def test_svm_family():
    """Verify SVC, SVR, OneClassSVM, and LinearSVC."""
    rng = np.random.RandomState(42)
    # Linearly separable binary dataset
    c1 = rng.randn(25, 2) + np.array([2.0, 2.0])
    c2 = rng.randn(25, 2) + np.array([-2.0, -2.0])
    X = np.vstack([c1, c2])
    y = np.array([0] * 25 + [1] * 25)

    # SVC with RBF kernel
    svc = SVC(C=1.0, kernel="rbf", gamma=0.5, max_iter=100)
    svc.fit(X, y)
    preds_svc = svc.predict(X)
    acc_svc = np.mean(preds_svc == y)
    assert acc_svc > 0.90

    # LinearSVC
    lsvc = LinearSVC(C=1.0, max_iter=200)
    lsvc.fit(X, y)
    preds_lsvc = lsvc.predict(X)
    acc_lsvc = np.mean(preds_lsvc == y)
    assert acc_lsvc > 0.90

    # SVR
    y_reg = X[:, 0] * 1.5 + 0.1 * rng.randn(50)
    svr = SVR(C=1.0, epsilon=0.1, kernel="linear", max_iter=100)
    svr.fit(X, y_reg)
    preds_svr = svr.predict(X)
    assert preds_svr.shape == (50,)

    # OneClassSVM
    oc_svm = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1, max_iter=100)
    oc_svm.fit(c1)
    oc_preds = oc_svm.predict(c1)
    assert len(oc_preds) == 25


def test_nearest_centroid_and_radius_neighbors():
    """Verify NearestCentroid and RadiusNeighbors models."""
    rng = np.random.RandomState(42)
    c1 = rng.randn(20, 2) + np.array([3.0, 3.0])
    c2 = rng.randn(20, 2) + np.array([-3.0, -3.0])
    X = np.vstack([c1, c2])
    y = np.array([0] * 20 + [1] * 20)

    # NearestCentroid with soft shrinkage
    nc = NearestCentroid(metric="euclidean", shrink_threshold=0.1)
    nc.fit(X, y)
    preds_nc = nc.predict(X)
    assert np.mean(preds_nc == y) > 0.90

    # RadiusNeighborsClassifier
    rnc = RadiusNeighborsClassifier(radius=3.0, weights="distance")
    rnc.fit(X, y)
    preds_rnc = rnc.predict(X)
    assert np.mean(preds_rnc == y) > 0.90

    # RadiusNeighborsRegressor
    y_reg = X[:, 0] * 2.0
    rnr = RadiusNeighborsRegressor(radius=3.0, weights="distance")
    rnr.fit(X, y_reg)
    preds_rnr = rnr.predict(X)
    assert preds_rnr.shape == (40,)
