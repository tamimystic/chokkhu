"""Unit tests for Sovereign AutoML Subsystem (GP Surrogate, Bayesian Optimization, Hyperband, AutoTrainer)."""

from __future__ import annotations
import numpy as np

from chokkhu.automl import (
    GaussianProcessSurrogate,
    expected_improvement,
    upper_confidence_bound,
    BayesianOptimization,
    Hyperband,
    AutoTrainer,
    AutoMLResult,
    auto_train,
)


def test_gp_surrogate_and_acquisitions() -> None:
    X_train = np.array([[1.0], [3.0], [5.0]])
    y_train = np.array([2.0, 4.0, 1.0])

    gp = GaussianProcessSurrogate(length_scale=1.5, variance=1.0)
    gp.fit(X_train, y_train)

    X_query = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    mu, sigma2 = gp.predict(X_query)

    assert mu.shape == (5,)
    assert sigma2.shape == (5,)
    # At training points, variance should be small
    assert sigma2[0] < 0.1
    assert sigma2[2] < 0.1
    assert sigma2[4] < 0.1

    # Acquisition values
    ei = expected_improvement(mu, sigma2, best_y=4.0)
    ucb = upper_confidence_bound(mu, sigma2)
    assert len(ei) == 5
    assert len(ucb) == 5


def test_bayesian_optimization_1d() -> None:
    # Maximize f(x) = -(x - 3.0)^2 + 10.0
    def objective(params: dict) -> float:
        x = params["x"]
        return float(-((x - 3.0) ** 2) + 10.0)

    bo = BayesianOptimization(
        objective_fn=objective,
        param_bounds={"x": (0.0, 6.0)},
        n_init=4,
        n_iter=8,
        random_state=42,
    )
    best_params = bo.optimize()
    assert "x" in best_params
    assert abs(best_params["x"] - 3.0) < 1.0
    assert bo.best_score > 8.0


def test_hyperband_successive_halving() -> None:
    def objective(config: dict, resource: int) -> float:
        # Higher lr closer to 0.05 is better, higher resource gives better convergence
        lr = config["lr"]
        base_score = 1.0 - abs(lr - 0.05)
        return float(base_score + 0.1 * (resource / 27.0))

    def sampler() -> dict:
        return {"lr": float(np.random.uniform(0.01, 0.1))}

    hb = Hyperband(
        objective_fn=objective,
        param_sampler=sampler,
        max_resource=27,
        eta=3,
        random_state=42,
    )
    best_config = hb.optimize()
    assert "lr" in best_config
    assert hb.best_score > 0.8


def test_autotrainer_classification() -> None:
    np.random.seed(42)
    X = np.random.randn(40, 4)
    # Simple linear decision boundary
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    res = auto_train(
        X,
        y,
        task="classification",
        cv=2,
        candidate_models=["logistic_regression", "decision_tree", "knn"],
    )
    assert isinstance(res, AutoMLResult)
    assert res.task == "classification"
    assert not res.leaderboard.empty
    assert res.best_score > 0.5
    assert "CHOKKHU AUTO-ML LEADERBOARD" in res.summary()

    preds = res.predict(X[:5])
    assert len(preds) == 5


def test_autotrainer_regression() -> None:
    np.random.seed(42)
    X = np.random.randn(30, 3)
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + 1.0

    trainer = AutoTrainer(
        task="regression",
        cv=2,
        candidate_models=["linear_regression", "ridge", "decision_tree"],
    )
    res = trainer.fit(X, y)
    assert isinstance(res, AutoMLResult)
    assert res.task == "regression"
    assert res.best_score > 0.7

    preds = res.predict(X[:3])
    assert len(preds) == 3
