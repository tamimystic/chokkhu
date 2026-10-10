import numpy as np
import pytest
from chokkhu.models.ml import (
    GradientBoosting,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    DecisionTree,
)

@pytest.fixture
def binary_data():
    np.random.seed(42)
    X = np.random.randn(60, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

@pytest.fixture
def regression_data():
    np.random.seed(42)
    X = np.random.randn(60, 4)
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * np.random.randn(60)
    return X, y

def test_gradient_boosting_classifier_dynamic_params(binary_data):
    X, y = binary_data
    clf = GradientBoostingClassifier(
        n_estimators=8,
        learning_rate=0.05,
        subsample=0.8,
        max_depth=2,
        min_samples_split=3,
        min_samples_leaf=2,
        n_iter_no_change=2,
        validation_fraction=0.2,
        random_state=42,
    )
    clf.fit(X, y)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    assert len(preds) == len(y)
    assert probs.shape == (len(y), 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

def test_gradient_boosting_regressor_dynamic_params(regression_data):
    X, y = regression_data
    reg = GradientBoostingRegressor(
        n_estimators=8,
        learning_rate=0.05,
        subsample=0.8,
        max_depth=2,
        criterion="friedman_mse",
        n_iter_no_change=2,
        validation_fraction=0.2,
        random_state=42,
    )
    reg.fit(X, y)
    preds = reg.predict(X)
    assert preds.shape == y.shape
    assert np.isfinite(preds).all()

def test_adaboost_classifier_dynamic_estimator(binary_data):
    X, y = binary_data
    base = DecisionTree(max_depth=2, task="classification")
    clf = AdaBoostClassifier(
        estimator=base,
        n_estimators=6,
        learning_rate=0.8,
        random_state=42,
    )
    clf.fit(X, y)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    assert len(preds) == len(y)
    assert probs.shape == (len(y), 2)
    assert np.allclose(np.sum(probs, axis=1), 1.0)

def test_adaboost_regressor_dynamic_loss(regression_data):
    X, y = regression_data
    for loss in ["linear", "square", "exponential"]:
        reg = AdaBoostRegressor(
            n_estimators=5,
            learning_rate=0.5,
            loss=loss,
            random_state=42,
        )
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == y.shape

def test_extra_trees_dynamic_params(binary_data, regression_data):
    X_bin, y_bin = binary_data
    clf = ExtraTreesClassifier(
        n_estimators=6,
        max_depth=3,
        min_samples_split=3,
        min_samples_leaf=2,
        bootstrap=True,
        max_samples=0.8,
        random_state=42,
    )
    clf.fit(X_bin, y_bin)
    preds = clf.predict(X_bin)
    assert len(preds) == len(y_bin)

    X_reg, y_reg = regression_data
    reg = ExtraTreesRegressor(
        n_estimators=6,
        max_depth=3,
        bootstrap=True,
        max_samples=0.8,
        random_state=42,
    )
    reg.fit(X_reg, y_reg)
    preds_reg = reg.predict(X_reg)
    assert preds_reg.shape == y_reg.shape

def test_hist_gradient_boosting_dynamic_params(binary_data, regression_data):
    X_bin, y_bin = binary_data
    clf = HistGradientBoostingClassifier(
        max_iter=6,
        learning_rate=0.08,
        max_leaf_nodes=15,
        min_samples_leaf=2,
        l2_regularization=0.5,
        max_bins=64,
        random_state=42,
    )
    clf.fit(X_bin, y_bin)
    preds = clf.predict(X_bin)
    assert len(preds) == len(y_bin)

    X_reg, y_reg = regression_data
    reg = HistGradientBoostingRegressor(
        max_iter=6,
        learning_rate=0.08,
        max_leaf_nodes=15,
        min_samples_leaf=2,
        l2_regularization=0.5,
        max_bins=64,
        random_state=42,
    )
    reg.fit(X_reg, y_reg)
    preds_reg = reg.predict(X_reg)
    assert preds_reg.shape == y_reg.shape
