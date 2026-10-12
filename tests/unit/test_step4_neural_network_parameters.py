from __future__ import annotations

import numpy as np
import pytest

from chokkhu.models.dl.layers import Parameter
from chokkhu.models.dl.optimizers import Adagrad
from chokkhu.models.dl.schedulers import (
    ExponentialLR,
    InverseScalingLR,
    LinearLR,
)
from chokkhu.models.ml.neural_network import (
    MLPClassifier,
    MLPRegressor,
    NeuralNetwork,
)


def test_neural_network_base_class():
    nn = NeuralNetwork(hidden_layer_sizes=[10], max_iter=5)
    assert nn.max_iter == 5
    assert nn.layers_config == [10]


def test_neural_network_activations():
    X = np.random.randn(50, 4)
    y = (X[:, 0] > 0).astype(int)
    for act in [
        "relu",
        "sigmoid",
        "tanh",
        "gelu",
        "silu",
        "leaky_relu",
        "elu",
        "identity",
    ]:
        clf = MLPClassifier(
            hidden_layer_sizes=(16,),
            activation=act,
            max_iter=15,
            random_state=42,
        )
        clf.fit(X, y)
        preds = clf.predict(X[:5])
        assert len(preds) == 5
        assert set(preds).issubset({0, 1})
        probs = clf.predict_proba(X[:5])
        assert probs.shape == (5, 2)
        np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(5), atol=1e-5)


def test_neural_network_solvers():
    X = np.random.randn(60, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    for solver in ["adam", "sgd", "rmsprop", "adamw", "adagrad", "lbfgs"]:
        clf = MLPClassifier(
            hidden_layer_sizes=(16,),
            solver=solver,
            max_iter=20,
            learning_rate_init=0.01,
            random_state=42,
        )
        clf.fit(X, y)
        acc = clf.score(X, y)
        assert 0.0 <= acc <= 1.0
        assert clf.n_iter_ > 0
        assert clf.loss_ is not None


def test_neural_network_learning_rate_schedules():
    X = np.random.randn(50, 3)
    y = (X[:, 0] > 0).astype(int)
    for sched in ["constant", "invscaling", "adaptive"]:
        clf = MLPClassifier(
            hidden_layer_sizes=(16,),
            learning_rate=sched,
            learning_rate_init=0.05,
            power_t=0.5,
            max_iter=20,
            random_state=42,
        )
        clf.fit(X, y)
        assert len(clf.loss_curve_) > 0


def test_neural_network_early_stopping():
    X = np.random.randn(100, 4)
    y = (X[:, 0] > 0).astype(int)
    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=5,
        max_iter=200,
        random_state=42,
    )
    clf.fit(X, y)
    assert clf.validation_scores_ is not None
    assert len(clf.validation_scores_) > 0
    assert clf.n_iter_ <= 200


def test_neural_network_regularization_and_dropout():
    X = np.random.randn(60, 4)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    clf = MLPClassifier(
        hidden_layer_sizes=(32,),
        alpha=0.01,
        dropout=0.1,
        max_iter=20,
        random_state=42,
    )
    clf.fit(X, y)
    assert len(clf.coefs_) == 2
    assert clf.coefs_[0].shape == (4, 32)
    assert clf.coefs_[1].shape == (32, 1)


def test_neural_network_class_weights():
    X = np.random.randn(60, 3)
    y = np.array([0] * 50 + [1] * 10)
    clf = MLPClassifier(
        hidden_layer_sizes=(16,),
        class_weight="balanced",
        max_iter=25,
        random_state=42,
    )
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == 60


def test_mlp_regressor():
    X = np.random.randn(80, 4)
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5
    reg = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        learning_rate_init=0.01,
        max_iter=40,
        random_state=42,
    )
    reg.fit(X, y)
    preds = reg.predict(X[:5])
    assert preds.shape == (5,)
    score = reg.score(X, y)
    assert isinstance(score, float)
    assert reg.out_activation_ == "identity"


def test_neural_network_multiclass():
    X = np.random.randn(90, 4)
    y = np.array([0] * 30 + [1] * 30 + [2] * 30)
    clf = MLPClassifier(
        hidden_layer_sizes=(24,),
        max_iter=30,
        random_state=42,
    )
    clf.fit(X, y)
    assert clf.n_classes == 3
    assert clf.out_activation_ == "softmax"
    probs = clf.predict_proba(X[:6])
    assert probs.shape == (6, 3)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(6), atol=1e-5)
    log_probs = clf.predict_log_proba(X[:6])
    assert log_probs.shape == (6, 3)
    assert np.all(log_probs <= 0.0)


def test_dl_schedulers_and_optimizers():
    param = Parameter(np.ones((4, 4), dtype=np.float64))
    param.grad = np.full((4, 4), 0.1, dtype=np.float64)

    opt = Adagrad([param], lr=0.05)
    opt.step()
    assert not np.allclose(param.data, np.ones((4, 4)))

    exp_sch = ExponentialLR(opt, gamma=0.9)
    exp_sch.step()
    assert exp_sch.get_lr() == pytest.approx(0.05 * 0.9)

    inv_sch = InverseScalingLR(opt, power_t=0.5)
    inv_sch.step()
    assert inv_sch.get_lr() < 0.05

    lin_sch = LinearLR(opt, start_factor=1.0, end_factor=0.2, total_iters=10)
    lin_sch.step()
    assert lin_sch.get_lr() < 0.05
