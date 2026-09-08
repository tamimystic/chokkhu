"""Time Series Preprocessing, Decomposition, and Feature Engineering from First Principles."""

from __future__ import annotations

from typing import Tuple, Union
import numpy as np


def decompose_series(
    series: Union[np.ndarray, list],
    period: int = 7,
    model: str = "additive",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform classical seasonal decomposition (Trend, Seasonal, Residual).

    Parameters
    ----------
    series : Union[np.ndarray, list]
        1D time series sequence.
    period : int
        Seasonal period length (e.g. 7 for daily with weekly seasonality).
    model : str
        'additive' (series = trend + seasonal + resid) or
        'multiplicative' (series = trend * seasonal * resid).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (trend, seasonal, residual)
    """
    y = np.asarray(series, dtype=np.float64).flatten()
    n = len(y)
    if n < 2 * period:
        raise ValueError(
            f"Series length {n} must be at least 2 * period ({2 * period})"
        )

    # 1. Compute moving average trend
    trend: np.ndarray = np.full(n, np.nan, dtype=np.float64)
    half_p = period // 2

    if period % 2 == 1:
        # Odd period simple moving average
        for i in range(half_p, n - half_p):
            trend[i] = np.mean(y[i - half_p : i + half_p + 1])
    else:
        # Even period 2xMA centered moving average
        for i in range(half_p, n - half_p):
            sub = y[i - half_p : i + half_p + 1]
            trend[i] = (0.5 * sub[0] + np.sum(sub[1:-1]) + 0.5 * sub[-1]) / period

    # Forward/backward fill edges of trend
    valid_idx = np.where(~np.isnan(trend))[0]
    trend[: valid_idx[0]] = trend[valid_idx[0]]
    trend[valid_idx[-1] :] = trend[valid_idx[-1]]

    # 2. De-trend
    if model == "additive":
        detrended = y - trend
    elif model == "multiplicative":
        trend_safe = np.where(trend == 0, 1e-8, trend)
        detrended = y / trend_safe
    else:
        raise ValueError(f"Unknown model: {model}")

    # 3. Compute seasonal averages for each phase
    seasonal_pattern: np.ndarray = np.zeros(period, dtype=np.float64)
    for p in range(period):
        indices = np.arange(p, n, period)
        seasonal_pattern[p] = np.nanmean(detrended[indices])

    if model == "additive":
        seasonal_pattern -= np.mean(seasonal_pattern)
    else:
        seasonal_pattern /= max(1e-8, np.mean(seasonal_pattern))

    seasonal = np.array([seasonal_pattern[i % period] for i in range(n)])

    # 4. Residual
    if model == "additive":
        residual = y - trend - seasonal
    else:
        seasonal_safe = np.where(seasonal == 0, 1e-8, seasonal)
        residual = y / (trend * seasonal_safe)

    return trend, seasonal, residual


def create_lag_matrix(
    series: Union[np.ndarray, list],
    lags: int = 10,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create autoregressive sliding window lag feature matrix X and target y."""
    y = np.asarray(series, dtype=np.float64).flatten()
    n = len(y)
    num_samples = n - lags - horizon + 1

    if num_samples <= 0:
        raise ValueError(
            f"Series of length {n} is too short for lags={lags}, horizon={horizon}"
        )

    X: np.ndarray = np.zeros((num_samples, lags), dtype=np.float64)
    Y: np.ndarray = np.zeros((num_samples, horizon), dtype=np.float64)

    for i in range(num_samples):
        X[i] = y[i : i + lags]
        Y[i] = y[i + lags : i + lags + horizon]

    if horizon == 1:
        return X, Y.flatten()
    return X, Y


def difference(series: Union[np.ndarray, list], d: int = 1) -> np.ndarray:
    """Compute d-th order differencing of time series."""
    y = np.asarray(series, dtype=np.float64).flatten()
    for _ in range(d):
        y = np.diff(y)
    return y


def inverse_difference(
    diff_series: Union[np.ndarray, list],
    initial_values: Union[np.ndarray, list],
) -> np.ndarray:
    """Invert differenced series using initial boundary values."""
    diff_y = np.asarray(diff_series, dtype=np.float64).flatten()
    inits = np.asarray(initial_values, dtype=np.float64).flatten()
    res = diff_y
    for init in reversed(inits):
        res = np.cumsum(np.insert(res, 0, init))
    return res
