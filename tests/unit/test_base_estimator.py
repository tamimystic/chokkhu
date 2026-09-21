"""Tests for ChokkhuModel base estimator protocol and compliance."""

import numpy as np
import pytest
from chokkhu.core.exceptions import NotFittedError
from chokkhu.models.base import check_is_fitted, clone
from chokkhu.models.ml.glm import RidgeRegression


def test_get_and_set_params():
    model = RidgeRegression(alpha=2.5, fit_intercept=True)
    params = model.get_params()
    assert "alpha" in params
    assert params["alpha"] == 2.5
    assert params["fit_intercept"] is True

    model.set_params(alpha=0.5, fit_intercept=False)
    assert model.alpha == 0.5
    assert model.fit_intercept is False


def test_estimator_clone():
    model = RidgeRegression(alpha=3.0)
    cloned = clone(model)
    assert cloned is not model
    assert cloned.alpha == 3.0
    assert cloned.coef_.size == 0
    assert not cloned.is_fitted


def test_check_is_fitted():
    model = RidgeRegression(alpha=1.0)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)

    X = np.random.randn(20, 3)
    y = X @ np.array([1.5, -2.0, 0.5]) + 0.1 * np.random.randn(20)
    model.fit(X, y)
    check_is_fitted(model)


def test_model_score():
    X = np.random.randn(30, 4)
    y = X @ np.array([1.0, 2.0, -1.0, 0.5])
    model = RidgeRegression(alpha=0.1)
    model.fit(X, y)
    score_val = model.score(X, y)
    assert isinstance(score_val, float)
    assert score_val > 0.8
