import numpy as np
import pytest

from chokkhu.models.ml.decision_tree import (
    DecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from chokkhu.models.ml.random_forest import (
    RandomForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from chokkhu.models.ml.isolation_forest import IsolationForest


@pytest.fixture
def classification_data():
    np.random.seed(42)
    X = np.random.randn(80, 6)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def regression_data():
    np.random.seed(42)
    X = np.random.randn(80, 5)
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * np.random.randn(80)
    return X, y


def test_decision_tree_classification_criteria_and_splitters(classification_data):
    X, y = classification_data
    for crit in ["gini", "entropy", "log_loss"]:
        for spl in ["best", "random"]:
            dt = DecisionTreeClassifier(
                criterion=crit,
                splitter=spl,
                max_depth=5,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
            )
            dt.fit(X, y)
            preds = dt.predict(X)
            proba = dt.predict_proba(X)
            assert preds.shape == (len(X),)
            assert proba.shape == (len(X), 2)
            assert np.allclose(np.sum(proba, axis=1), 1.0)
            assert dt.feature_importances_.shape == (X.shape[1],)
            assert np.isclose(np.sum(dt.feature_importances_), 1.0)


def test_decision_tree_regression_parameters(regression_data):
    X, y = regression_data
    for crit in ["squared_error", "absolute_error"]:
        dt = DecisionTreeRegressor(
            criterion=crit,
            max_depth=4,
            min_samples_split=0.1,
            min_samples_leaf=0.05,
            max_features=0.8,
            min_impurity_decrease=0.01,
            max_leaf_nodes=10,
            random_state=42,
        )
        dt.fit(X, y)
        preds = dt.predict(X)
        assert preds.shape == (len(X),)
        assert dt.feature_importances_.shape == (X.shape[1],)


def test_decision_tree_class_weights(classification_data):
    X, y = classification_data
    # Imbalanced sample weights
    dt_balanced = DecisionTree(class_weight="balanced", random_state=42)
    dt_balanced.fit(X, y)
    assert dt_balanced.predict(X).shape == (len(X),)

    dt_dict = DecisionTree(class_weight={0: 1.0, 1: 3.0}, random_state=42)
    dt_dict.fit(X, y)
    assert dt_dict.predict(X).shape == (len(X),)


def test_random_forest_classification_all_parameters(classification_data):
    X, y = classification_data
    rf = RandomForestClassifier(
        n_estimators=15,
        criterion="entropy",
        max_depth=4,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="log2",
        bootstrap=True,
        oob_score=True,
        max_samples=0.8,
        class_weight="balanced",
        random_state=42,
    )
    rf.fit(X, y)
    preds = rf.predict(X)
    probas = rf.predict_proba(X)
    log_probas = rf.predict_log_proba(X)

    assert preds.shape == (len(X),)
    assert probas.shape == (len(X), 2)
    assert log_probas.shape == (len(X), 2)
    assert len(rf.estimators_) == 15
    assert rf.oob_score_ > 0.0
    assert rf.feature_importances_.shape == (X.shape[1],)
    assert np.isclose(np.sum(rf.feature_importances_), 1.0)


def test_random_forest_regression_all_parameters(regression_data):
    X, y = regression_data
    rf = RandomForestRegressor(
        n_estimators=10,
        criterion="absolute_error",
        max_depth=4,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        bootstrap=True,
        oob_score=True,
        random_state=42,
    )
    rf.fit(X, y)
    preds = rf.predict(X)
    assert preds.shape == (len(X),)
    assert rf.oob_score_ <= 1.0
    assert rf.feature_importances_.shape == (X.shape[1],)


def test_random_forest_warm_start(classification_data):
    X, y = classification_data
    rf = RandomForest(n_estimators=5, warm_start=True, random_state=42)
    rf.fit(X, y)
    assert len(rf.trees) == 5

    rf.n_estimators = 10
    rf.fit(X, y)
    assert len(rf.trees) == 10


def test_isolation_forest_all_parameters():
    np.random.seed(42)
    X = np.random.randn(100, 4)
    # Inject 5 outliers
    X[:5] = 15.0

    iso = IsolationForest(
        n_estimators=25,
        max_samples=0.8,
        contamination=0.05,
        max_features=0.75,
        bootstrap=True,
        random_state=42,
    )
    iso.fit(X)
    preds = iso.predict(X)
    scores = iso.score_samples(X)
    dec_fn = iso.decision_function(X)

    assert preds.shape == (100,)
    assert scores.shape == (100,)
    assert dec_fn.shape == (100,)
    assert len(iso.estimators_) == 25
    assert len(iso.estimators_features_) == 25
    # The injected 5 outliers should have -1
    assert np.sum(preds == -1) >= 4
