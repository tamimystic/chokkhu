import numpy as np
import pytest

from chokkhu.models.ml.linear_regression import LinearRegression
from chokkhu.models.ml.logistic_regression import LogisticRegression
from chokkhu.models.ml.knn import KNN
from chokkhu.models.ml.kmeans import KMeans
from chokkhu.models.ml.svm import SVM


@pytest.fixture
def linear_data():
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 1.5 + 0.1 * np.random.randn(50)
    return X, y


@pytest.fixture
def classification_data():
    np.random.seed(42)
    X = np.random.randn(60, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def multiclass_data():
    np.random.seed(42)
    X = np.random.randn(90, 4)
    y = np.zeros(90, dtype=int)
    y[X[:, 0] > 0.5] = 1
    y[X[:, 0] < -0.5] = 2
    return X, y


def test_linear_regression_parameters(linear_data):
    X, y = linear_data
    # 1. Standard OLS with fit_intercept=True
    lr_ols = LinearRegression(fit_intercept=True)
    lr_ols.fit(X, y)
    assert lr_ols.score(X, y) > 0.95
    assert lr_ols.coef_.shape == (4,)
    assert isinstance(lr_ols.intercept_, float)

    # 2. Gradient descent with L2 Ridge
    lr_ridge = LinearRegression(
        method="gradient_descent", regularization="l2", alpha=0.5, epochs=500
    )
    lr_ridge.fit(X, y)
    assert lr_ridge.score(X, y) > 0.90

    # 3. Positive weights constraint
    lr_pos = LinearRegression(positive=True, epochs=500)
    lr_pos.fit(X, y)
    assert np.all(lr_pos.coef_ >= 0.0)


def test_logistic_regression_parameters(classification_data, multiclass_data):
    X, y = classification_data
    # 1. Binary with L2 penalty, class_weight='balanced'
    log_l2 = LogisticRegression(
        penalty="l2", C=0.5, class_weight="balanced", max_iter=300
    )
    log_l2.fit(X, y)
    proba = log_l2.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(np.sum(proba, axis=1), 1.0)
    assert log_l2.score(X, y) > 0.85

    # 2. Binary with L1 penalty
    log_l1 = LogisticRegression(penalty="l1", C=1.0, max_iter=300)
    log_l1.fit(X, y)
    assert log_l1.predict(X).shape == (len(X),)

    # 3. Multi-class (3 classes)
    X_mc, y_mc = multiclass_data
    log_mc = LogisticRegression(max_iter=300)
    log_mc.fit(X_mc, y_mc)
    proba_mc = log_mc.predict_proba(X_mc)
    assert proba_mc.shape == (len(X_mc), 3)
    assert np.allclose(np.sum(proba_mc, axis=1), 1.0)
    assert log_mc.predict(X_mc).shape == (len(X_mc),)


def test_knn_parameters(classification_data):
    X, y = classification_data
    for metric in ["euclidean", "manhattan", "cosine", "minkowski"]:
        for weights in ["uniform", "distance"]:
            knn = KNN(n_neighbors=3, weights=weights, metric=metric, p=3.0)
            knn.fit(X, y)
            dists, indices = knn.kneighbors(X[:5], n_neighbors=3)
            assert dists.shape == (5, 3)
            assert indices.shape == (5, 3)
            preds = knn.predict(X)
            proba = knn.predict_proba(X)
            assert preds.shape == (len(X),)
            assert proba.shape == (len(X), 2)
            assert np.allclose(np.sum(proba, axis=1), 1.0)


def test_kmeans_parameters():
    np.random.seed(42)
    X = np.vstack([np.random.randn(30, 2) + 5.0, np.random.randn(30, 2) - 5.0])
    # 1. k-means++ init
    km_pp = KMeans(n_clusters=2, init="k-means++", n_init=5, random_state=42)
    km_pp.fit(X)
    assert km_pp.cluster_centers_.shape == (2, 2)
    assert km_pp.labels_.shape == (60,)
    assert km_pp.inertia_ > 0.0

    # 2. transform and score
    dist_space = km_pp.transform(X)
    assert dist_space.shape == (60, 2)
    score_val = km_pp.score(X)
    assert score_val == -km_pp.inertia_

    # 3. random init
    km_rand = KMeans(n_clusters=2, init="random", n_init=3, random_state=42)
    labels = km_rand.fit_predict(X)
    assert len(labels) == 60


def test_svm_parameters(classification_data):
    X, y = classification_data
    # 1. Linear SVM
    svm_lin = SVM(kernel="linear", C=1.0, epochs=300)
    svm_lin.fit(X, y)
    assert svm_lin.predict(X).shape == (len(X),)
    assert len(svm_lin.support_vectors_) > 0

    # 2. RBF Kernel SVM with Platt scaling probability
    svm_rbf = SVM(kernel="rbf", gamma="scale", C=1.0, probability=True, epochs=200)
    svm_rbf.fit(X, y)
    probs = svm_rbf.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

    # 3. Polynomial Kernel SVM
    svm_poly = SVM(kernel="poly", degree=2, epochs=200)
    svm_poly.fit(X, y)
    assert svm_poly.predict(X).shape == (len(X),)
