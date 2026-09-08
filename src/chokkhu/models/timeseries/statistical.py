"""Statistical Time Series Models: ARIMA and Holt-Winters Exponential Smoothing."""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from ..base import ChokkhuModel
from .transforms import difference


class ARIMA(ChokkhuModel):
    """AutoRegressive Integrated Moving Average (ARIMA(p, d, q)) from First Principles."""

    def __init__(self, p: int = 1, d: int = 0, q: int = 1) -> None:
        self.p = p
        self.d = d
        self.q = q
        self.ar_params: np.ndarray = np.array([])
        self.ma_params: np.ndarray = np.array([])
        self.intercept: float = 0.0
        self.history: np.ndarray = np.array([])
        self.residuals: np.ndarray = np.array([])
        self.orig_series: np.ndarray = np.array([])

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> ARIMA:
        """Fit ARIMA(p, d, q) on univariate time series."""
        raw = np.asarray(X if y is None else y, dtype=np.float64).flatten()
        self.orig_series = raw.copy()

        # Step 1: Differencing
        if self.d > 0:
            diff_y = difference(raw, d=self.d)
        else:
            diff_y = raw.copy()

        self.history = diff_y.copy()
        n = len(diff_y)

        # Step 2: Fit AR(p) via Ordinary Least Squares
        if self.p > 0 and n > self.p:
            X_ar: np.ndarray = np.zeros((n - self.p, self.p), dtype=np.float64)
            y_ar = diff_y[self.p :]
            for i in range(n - self.p):
                X_ar[i] = diff_y[i : i + self.p][::-1]

            # Add intercept column
            X_design = np.column_stack([np.ones(len(y_ar)), X_ar])
            try:
                beta, _, _, _ = np.linalg.lstsq(X_design, y_ar, rcond=None)
                self.intercept = float(beta[0])
                self.ar_params = beta[1:]
            except np.linalg.LinAlgError:
                self.intercept = float(np.mean(y_ar))
                self.ar_params = np.zeros(self.p)

            pred_ar = X_design @ np.insert(self.ar_params, 0, self.intercept)
            self.residuals = y_ar - pred_ar
        else:
            self.intercept = float(np.mean(diff_y)) if len(diff_y) > 0 else 0.0
            self.ar_params = np.zeros(self.p)
            self.residuals = diff_y - self.intercept

        # Step 3: Fit MA(q) parameters on residuals
        if self.q > 0 and len(self.residuals) > self.q:
            m = len(self.residuals)
            X_ma: np.ndarray = np.zeros((m - self.q, self.q), dtype=np.float64)
            y_ma = self.residuals[self.q :]
            for i in range(m - self.q):
                X_ma[i] = self.residuals[i : i + self.q][::-1]

            try:
                theta, _, _, _ = np.linalg.lstsq(X_ma, y_ma, rcond=None)
                self.ma_params = theta
            except np.linalg.LinAlgError:
                self.ma_params = np.zeros(self.q)
        else:
            self.ma_params = np.zeros(self.q)

        return self

    def predict(self, X: Any = None, steps: int = 10) -> np.ndarray:
        """Forecast future steps."""
        if X is not None:
            if isinstance(X, int):
                steps = X
            elif hasattr(X, "__len__"):
                steps = len(X)

        curr_hist = list(self.history)
        curr_res = list(self.residuals)
        diff_forecasts = []

        for _ in range(steps):
            pred = self.intercept
            if self.p > 0:
                lags = np.array(curr_hist[-self.p :][::-1])
                if len(lags) < self.p:
                    lags = np.pad(lags, (0, self.p - len(lags)), mode="edge")
                pred += float(np.dot(self.ar_params, lags))

            if self.q > 0:
                ma_lags = np.array(curr_res[-self.q :][::-1])
                if len(ma_lags) < self.q:
                    ma_lags = np.pad(
                        ma_lags, (0, self.q - len(ma_lags)), mode="constant"
                    )
                pred += float(np.dot(self.ma_params, ma_lags))

            diff_forecasts.append(pred)
            curr_hist.append(pred)
            curr_res.append(0.0)  # Future expected residual is 0

        forecast_arr = np.array(diff_forecasts, dtype=np.float64)

        # Step 4: Invert differencing if d > 0
        if self.d > 0:
            last_val = self.orig_series[-1]
            out = last_val + np.cumsum(forecast_arr)
        else:
            out = forecast_arr

        return out


class ExponentialSmoothing(ChokkhuModel):
    """Holt-Winters Exponential Smoothing (Level, Trend, and Seasonality)."""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        seasonal_periods: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.seasonal_periods = seasonal_periods

        self.level: float = 0.0
        self.trend: float = 0.0
        self.seasonals: np.ndarray = np.array([])

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> ExponentialSmoothing:
        """Fit Holt-Winters Exponential Smoothing."""
        series = np.asarray(X if y is None else y, dtype=np.float64).flatten()
        n = len(series)

        self.level = float(series[0])
        self.trend = (
            float(series[1] - series[0]) if n > 1 and self.beta is not None else 0.0
        )

        if (
            self.gamma is not None
            and self.seasonal_periods is not None
            and self.seasonal_periods > 0
        ):
            P = self.seasonal_periods
            self.seasonals = np.array([float(series[i]) - self.level for i in range(P)])
        else:
            self.seasonals = np.array([])

        for i in range(1, n):
            val = float(series[i])
            last_level = self.level
            last_trend = self.trend

            # Seasonality adjustment
            if len(self.seasonals) > 0 and self.seasonal_periods is not None:
                p_idx = i % self.seasonal_periods
                season = self.seasonals[p_idx]
                self.level = self.alpha * (val - season) + (1.0 - self.alpha) * (
                    last_level + last_trend
                )
                if self.beta is not None:
                    self.trend = (
                        self.beta * (self.level - last_level)
                        + (1.0 - self.beta) * last_trend
                    )
                if self.gamma is not None:
                    self.seasonals[p_idx] = (
                        self.gamma * (val - self.level) + (1.0 - self.gamma) * season
                    )
            else:
                self.level = self.alpha * val + (1.0 - self.alpha) * (
                    last_level + last_trend
                )
                if self.beta is not None:
                    self.trend = (
                        self.beta * (self.level - last_level)
                        + (1.0 - self.beta) * last_trend
                    )

        return self

    def predict(self, X: Any = None, steps: int = 10) -> np.ndarray:
        """Forecast future values."""
        if X is not None:
            if isinstance(X, int):
                steps = X
            elif hasattr(X, "__len__"):
                steps = len(X)

        forecasts = []
        for h in range(1, steps + 1):
            pred = self.level + h * self.trend
            if len(self.seasonals) > 0 and self.seasonal_periods is not None:
                p_idx = (len(self.seasonals) + h - 1) % self.seasonal_periods
                pred += self.seasonals[p_idx]
            forecasts.append(pred)

        return np.array(forecasts, dtype=np.float64)
