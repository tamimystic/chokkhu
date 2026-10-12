from __future__ import annotations

import numpy as np

from chokkhu.models.timeseries import ARIMA, SARIMAX, AutoARIMA, ExponentialSmoothing


def test_arima_confidence_intervals() -> None:
    np.random.seed(42)
    y = np.cumsum(np.random.randn(50) + 0.1)
    model = ARIMA(p=2, d=1, q=1, trend="c")
    model.fit(y)
    mean, lower, upper = model.forecast(steps=6, return_conf_int=True)
    assert len(mean) == 6 and len(lower) == 6 and len(upper) == 6
    assert np.all(lower < upper)
    assert np.all(lower <= mean) and np.all(mean <= upper)
    assert not np.isnan(model.aic) and not np.isnan(model.bic)


def test_sarimax_exogenous() -> None:
    np.random.seed(42)
    n = 60
    exog_train = np.random.randn(n, 2)
    y = (
        2.5 * exog_train[:, 0]
        - 1.2 * exog_train[:, 1]
        + np.cumsum(np.random.randn(n) * 0.1)
    )
    sarimax = SARIMAX(order=(1, 1, 1), trend="c")
    sarimax.fit(y, exog=exog_train)
    assert sarimax.exog_beta is not None and len(sarimax.exog_beta) == 2
    exog_test = np.random.randn(5, 2)
    mean, lower, upper = sarimax.forecast(steps=5, exog=exog_test, return_conf_int=True)
    assert len(mean) == 5
    assert np.all(lower <= upper)


def test_auto_arima() -> None:
    np.random.seed(42)
    y = np.cumsum(np.random.randn(40))
    auto = AutoARIMA(start_p=0, max_p=2, start_q=0, max_q=2, d=1)
    auto.fit(y)
    assert auto.is_fitted
    assert len(auto.best_order_) == 3
    preds = auto.predict(steps=4)
    assert len(preds) == 4
    mean, lower, upper = auto.forecast(steps=4, return_conf_int=True)
    assert len(mean) == 4 and np.all(lower <= upper)


def test_exponential_smoothing_damped() -> None:
    np.random.seed(42)
    y = np.linspace(10, 50, 30) + np.random.randn(30) * 0.5
    es = ExponentialSmoothing(
        alpha=0.3, beta=0.1, trend="additive", damped_trend=True, damping_slope=0.9
    )
    es.fit(y)
    assert es.is_fitted
    mean, lower, upper = es.predict(steps=5, return_conf_int=True)
    assert len(mean) == 5
    assert np.all(lower < upper)
