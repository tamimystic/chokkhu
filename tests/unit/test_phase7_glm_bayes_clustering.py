"""Unit tests for Phase 7: GLMs, Bayesian Classifiers, Clustering Universe & KNN in Pure NumPy."""

from __future__ import annotations

import numpy as np
from chokkhu.models.ml.glm import (
    RidgeRegression,
    LassoRegression,
    ElasticNet,
    HuberRegressor,
    BayesianRidge,
    ARDRegression,
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
)
from chokkhu.models.ml.naive_bayes_extended import (
    GaussianNB,
    MultinomialNB,
    BernoulliNB,
    ComplementNB,
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from chokkhu.models.ml.clustering_advanced import (
    MiniBatchKMeans,
    GaussianMixture,
    SpectralClustering,
)
from chokkhu.models.ml.knn_spatial import (
    KNNClassifier,
    KNNRegressor,
)


def test_glms_regularized_regression():
    """Verify Ridge, Lasso, and ElasticNet models with analytical and coordinate descent solvers."""
    rng = np.random.RandomState(42)
    X = rng.randn(60, 5)
    true_w = np.array([3.0, 0.0, -2.0, 0.0, 1.5])
    y = np.dot(X, true_w) + 1.0 + 0.1 * rng.randn(60)

    # Ridge
    ridge = RidgeRegression(alpha=0.5, fit_intercept=True)
    ridge.fit(X, y)
    preds_ridge = ridge.predict(X)
    assert np.mean((preds_ridge - y) ** 2) < 0.1

    # Lasso (should drive zero-weights close to 0)
    lasso = LassoRegression(alpha=0.1, fit_intercept=True)
    lasso.fit(X, y)
    preds_lasso = lasso.predict(X)
    assert np.mean((preds_lasso - y) ** 2) < 0.2
    assert abs(lasso.coef_[1]) < 0.05
    assert abs(lasso.coef_[3]) < 0.05

    # ElasticNet
    enet = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=True)
    enet.fit(X, y)
    preds_enet = enet.predict(X)
    assert np.mean((preds_enet - y) ** 2) < 0.2


def test_huber_and_bayesian_regression():
    """Verify HuberRegressor on outlier corrupted targets and BayesianRidge/ARD on probabilistic targets."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 3)
    y = X[:, 0] * 2.0 - X[:, 1] * 1.5 + 0.1 * rng.randn(50)
    # Add heavy outlier
    y[0] += 50.0

    huber = HuberRegressor(epsilon=1.35, alpha=0.01)
    huber.fit(X, y)
    preds_huber = huber.predict(X[1:])
    mse_inliers = float(np.mean((preds_huber - y[1:]) ** 2))
    assert mse_inliers < 0.2

    # BayesianRidge with uncertainty bounds
    y_clean = X[:, 0] * 2.0 - X[:, 1] * 1.5 + 0.1 * rng.randn(50)
    bridge = BayesianRidge()
    bridge.fit(X, y_clean)
    preds, stds = bridge.predict_with_std(X)
    assert preds.shape == (50,)
    assert stds.shape == (50,)
    assert np.all(stds > 0)

    # ARDRegression
    ard = ARDRegression()
    ard.fit(X, y_clean)
    preds_ard = ard.predict(X)
    assert np.mean((preds_ard - y_clean) ** 2) < 0.1


def test_passive_aggressive_streaming():
    """Verify PassiveAggressiveClassifier and PassiveAggressiveRegressor."""
    rng = np.random.RandomState(42)
    c1 = rng.randn(25, 2) + np.array([2.0, 2.0])
    c2 = rng.randn(25, 2) + np.array([-2.0, -2.0])
    X = np.vstack([c1, c2])
    y_cls = np.array([0] * 25 + [1] * 25)

    pac = PassiveAggressiveClassifier(C=1.0)
    pac.fit(X, y_cls)
    preds_pac = pac.predict(X)
    assert np.mean(preds_pac == y_cls) > 0.90

    y_reg = X[:, 0] * 1.5 + 0.1 * rng.randn(50)
    par = PassiveAggressiveRegressor(C=1.0, epsilon=0.1)
    par.fit(X, y_reg)
    preds_par = par.predict(X)
    assert preds_par.shape == (50,)


def test_bayesian_and_discriminant_classifiers():
    """Verify GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, LDA, and QDA."""
    rng = np.random.RandomState(42)
    c1 = rng.randn(30, 3) + np.array([2.0, 2.0, 2.0])
    c2 = rng.randn(30, 3) + np.array([-2.0, -2.0, -2.0])
    X_cont = np.vstack([c1, c2])
    y = np.array([0] * 30 + [1] * 30)

    # GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_cont, y)
    preds_gnb = gnb.predict(X_cont)
    assert np.mean(preds_gnb == y) > 0.90

    # MultinomialNB & ComplementNB (positive token count distributions)
    c1_counts = rng.poisson(lam=[10.0, 8.0, 1.0], size=(30, 3)).astype(np.float64)
    c2_counts = rng.poisson(lam=[1.0, 2.0, 12.0], size=(30, 3)).astype(np.float64)
    X_counts = np.vstack([c1_counts, c2_counts])

    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_counts, y)
    assert np.mean(mnb.predict(X_counts) == y) > 0.85

    cnb = ComplementNB(alpha=1.0)
    cnb.fit(X_counts, y)
    assert np.mean(cnb.predict(X_counts) == y) > 0.85

    # BernoulliNB
    X_bin = (X_cont > 0).astype(np.float64)
    bnb = BernoulliNB()
    bnb.fit(X_bin, y)
    assert np.mean(bnb.predict(X_bin) == y) > 0.85

    # LDA & QDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_cont, y)
    assert np.mean(lda.predict(X_cont) == y) > 0.90

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_cont, y)
    assert np.mean(qda.predict(X_cont) == y) > 0.90


def test_clustering_and_knn_universe():
    """Verify MiniBatchKMeans, GaussianMixture, SpectralClustering, KNNClassifier, and KNNRegressor."""
    rng = np.random.RandomState(42)
    c1 = rng.randn(30, 2) + np.array([4.0, 4.0])
    c2 = rng.randn(30, 2) + np.array([-4.0, -4.0])
    X = np.vstack([c1, c2])
    y_cls = np.array([0] * 30 + [1] * 30)

    # MiniBatchKMeans
    mbk = MiniBatchKMeans(n_clusters=2, batch_size=20, random_state=42)
    mbk.fit(X)
    assert len(mbk.labels_) == 60
    assert mbk.inertia_ > 0.0

    # GaussianMixture
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(X)
    probs_gmm = gmm.predict_proba(X)
    assert probs_gmm.shape == (60, 2)

    # SpectralClustering
    spec = SpectralClustering(n_clusters=2, gamma=0.1, random_state=42)
    spec.fit(X)
    assert len(spec.labels_) == 60

    # KNNClassifier with distance weighting
    knn_cls = KNNClassifier(n_neighbors=5, weights="distance", metric="euclidean")
    knn_cls.fit(X, y_cls)
    preds_knn = knn_cls.predict(X)
    assert np.mean(preds_knn == y_cls) > 0.90

    # KNNRegressor
    y_reg = X[:, 0] * 2.0 - X[:, 1]
    knn_reg = KNNRegressor(n_neighbors=5, weights="distance")
    knn_reg.fit(X, y_reg)
    preds_reg = knn_reg.predict(X)
    assert preds_reg.shape == (60,)
