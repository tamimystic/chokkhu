"""Unit tests for Time Series & Forecasting Universe (ARIMA, Exponential Smoothing, N-BEATS, N-HiTS, PatchTST)."""

import numpy as np

from chokkhu.core.tensor import Tensor
from chokkhu.models.timeseries import (
    decompose_series,
    create_lag_matrix,
    difference,
    inverse_difference,
    ARIMA,
    ExponentialSmoothing,
    NBEATS,
    NHITS,
    PatchTST,
)
from chokkhu.models import train


def test_decomposition_additive_and_multiplicative() -> None:
    t = np.linspace(0, 4 * np.pi, 50)
    trend = 0.5 * t
    seasonal = 2.0 * np.sin(t)
    noise = np.random.normal(0, 0.1, 50)

    # Additive
    y_add = trend + seasonal + noise
    tr, seas, res = decompose_series(y_add, period=7, model="additive")
    assert tr.shape == (50,)
    assert seas.shape == (50,)
    assert res.shape == (50,)

    # Multiplicative (strictly positive)
    y_mult = (trend + 10.0) * (seasonal + 3.0)
    tr_m, seas_m, res_m = decompose_series(y_mult, period=7, model="multiplicative")
    assert tr_m.shape == (50,)
    assert seas_m.shape == (50,)
    assert res_m.shape == (50,)


def test_lag_matrix_and_differencing() -> None:
    series = np.arange(20, dtype=np.float64)

    # Lag matrix
    X, y = create_lag_matrix(series, lags=5, horizon=1)
    assert X.shape == (15, 5)
    assert y.shape == (15,)
    np.testing.assert_array_equal(X[0], [0, 1, 2, 3, 4])
    assert y[0] == 5.0

    # Multi-horizon
    X_mh, y_mh = create_lag_matrix(series, lags=5, horizon=3)
    assert X_mh.shape == (13, 5)
    assert y_mh.shape == (13, 3)

    # Differencing & inverse
    diff1 = difference(series, d=1)
    assert len(diff1) == 19
    np.testing.assert_allclose(diff1, np.ones(19))

    inv = inverse_difference(diff1, initial_values=[0.0])
    np.testing.assert_allclose(inv, series)


def test_arima_fitting_and_forecasting() -> None:
    # Synthetic random walk with drift
    np.random.seed(42)
    steps = 40
    y = np.cumsum(np.random.randn(steps) + 0.5)

    arima = ARIMA(p=2, d=1, q=1)
    arima.fit(y)
    preds = arima.predict(steps=5)
    assert preds.shape == (5,)
    assert not np.isnan(preds).any()


def test_exponential_smoothing_holt_winters() -> None:
    y = np.array([10, 12, 14, 16, 18, 20, 22, 24, 26, 28], dtype=np.float64)
    es = ExponentialSmoothing(alpha=0.4, beta=0.2)
    es.fit(y)
    preds = es.predict(steps=4)
    assert preds.shape == (4,)
    # Increasing trend
    assert preds[-1] > preds[0]


def test_nbeats_forward_and_residual_stacking() -> None:
    x = Tensor(np.random.randn(4, 24), requires_grad=True)
    nbeats = NBEATS(input_length=24, horizon=6, hidden_dim=32, num_blocks=4)

    forecast = nbeats(x)
    assert forecast.shape == (4, 6)

    preds = nbeats.predict(np.random.randn(2, 24))
    assert preds.shape == (2, 6)


def test_nhits_hierarchical_pooling() -> None:
    x = Tensor(np.random.randn(3, 24), requires_grad=True)
    nhits = NHITS(input_length=24, horizon=6, hidden_dim=32, num_blocks=3)

    forecast = nhits(x)
    assert forecast.shape == (3, 6)


def test_patchtst_patching_and_prediction() -> None:
    x = Tensor(np.random.randn(4, 24), requires_grad=True)
    patchtst = PatchTST(input_length=24, horizon=6, patch_len=8, stride=4, embed_dim=16)

    forecast = patchtst(x)
    assert forecast.shape == (4, 6)


def test_train_timeseries_via_engine() -> None:
    y = np.cumsum(np.random.randn(30) + 0.2)

    # Train ARIMA
    model_arima = train(model="arima", X_train=y, p=1, d=1, q=1)
    assert isinstance(model_arima, ARIMA)
    f_arima = model_arima.predict(steps=3)
    assert f_arima.shape == (3,)

    # Train N-BEATS
    X_train = np.random.randn(8, 24)
    model_nbeats = train(model="nbeats", X_train=X_train, input_length=24, horizon=4)
    assert isinstance(model_nbeats, NBEATS)
    f_nbeats = model_nbeats.predict(X_train[:2])
    assert f_nbeats.shape == (2, 4)
