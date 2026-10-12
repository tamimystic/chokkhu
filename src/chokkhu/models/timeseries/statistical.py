from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np

from ..base import ChokkhuModel
from .transforms import difference


class ARIMA(ChokkhuModel):
    """AutoRegressive Integrated Moving Average (ARIMA(p, d, q)) in Pure NumPy."""

    def __init__(
        self,
        p: int = 1,
        d: int = 0,
        q: int = 1,
        order: Optional[Tuple[int, int, int]] = None,
        trend: str = "c",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if order is not None:
            self.p, self.d, self.q = order
        else:
            self.p = int(p)
            self.d = int(d)
            self.q = int(q)
        self.trend = trend
        self.ar_params: np.ndarray = np.array([], dtype=np.float64)
        self.ma_params: np.ndarray = np.array([], dtype=np.float64)
        self.intercept: float = 0.0
        self.history: np.ndarray = np.array([], dtype=np.float64)
        self.residuals: np.ndarray = np.array([], dtype=np.float64)
        self.orig_series: np.ndarray = np.array([], dtype=np.float64)
        self.is_fitted: bool = False
        self._aic: float = float("nan")
        self._bic: float = float("nan")

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> ARIMA:
        raw = np.asarray(X if y is None else y, dtype=np.float64).flatten()
        self.orig_series = raw.copy()

        diff_y = difference(raw, d=self.d) if self.d > 0 else raw.copy()
        self.history = diff_y.copy()
        n = len(diff_y)

        if self.p > 0 and n > self.p:
            X_ar: np.ndarray = np.zeros((n - self.p, self.p), dtype=np.float64)
            y_ar = diff_y[self.p :]
            for i in range(n - self.p):
                X_ar[i] = diff_y[i : i + self.p][::-1]

            if self.trend == "c":
                X_design = np.column_stack([np.ones(len(y_ar)), X_ar])
            else:
                X_design = X_ar

            try:
                beta, _, _, _ = np.linalg.lstsq(X_design, y_ar, rcond=None)
                if self.trend == "c":
                    self.intercept = float(beta[0])
                    self.ar_params = beta[1:]
                else:
                    self.intercept = 0.0
                    self.ar_params = beta
            except np.linalg.LinAlgError:
                self.intercept = float(np.mean(y_ar)) if self.trend == "c" else 0.0
                self.ar_params = np.zeros(self.p)

            pred_ar = (
                X_design @ np.insert(self.ar_params, 0, self.intercept)
                if self.trend == "c"
                else X_design @ self.ar_params
            )
            self.residuals = y_ar - pred_ar
        else:
            self.intercept = (
                float(np.mean(diff_y)) if self.trend == "c" and len(diff_y) > 0 else 0.0
            )
            self.ar_params = np.zeros(self.p)
            self.residuals = diff_y - self.intercept

        if self.q > 0 and len(self.residuals) > self.q:
            m = len(self.residuals)
            X_ma: np.ndarray = np.zeros((m - self.q, self.q), dtype=np.float64)
            y_ma = self.residuals[self.q :]
            for i in range(m - self.q):
                X_ma[i] = self.residuals[i : i + self.q][::-1]

            try:
                gamma, _, _, _ = np.linalg.lstsq(X_ma, y_ma, rcond=None)
                self.ma_params = gamma
            except np.linalg.LinAlgError:
                self.ma_params = np.zeros(self.q)
        else:
            self.ma_params = np.zeros(self.q)

        k = self.p + self.q + (1 if self.trend == "c" else 0)
        sse = float(np.sum(self.residuals**2))
        sigma2 = max(sse / max(len(self.residuals), 1), 1e-12)
        self._aic = float(n * np.log(sigma2) + 2 * k)
        self._bic = float(n * np.log(sigma2) + np.log(max(n, 1)) * k)

        self.is_fitted = True
        return self

    @property
    def aic(self) -> float:
        return self._aic

    @property
    def bic(self) -> float:
        return self._bic

    def forecast(
        self,
        steps: int = 10,
        return_conf_int: bool = False,
        alpha: float = 0.05,
        exog: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")

        curr_hist = list(self.history)
        curr_res = list(self.residuals)
        diff_forecasts: list[float] = []

        for _ in range(steps):
            pred = self.intercept
            if self.p > 0:
                lags = np.array(curr_hist[-self.p :][::-1])
                if len(lags) < self.p:
                    lags = np.pad(lags, (0, self.p - len(lags)), mode="constant")
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
            curr_res.append(0.0)

        forecast_arr = np.array(diff_forecasts, dtype=np.float64)

        if self.d > 0:
            last_val = self.orig_series[-1]
            out = last_val + np.cumsum(forecast_arr)
        else:
            out = forecast_arr

        if return_conf_int:
            sigma = float(np.std(self.residuals)) if len(self.residuals) > 0 else 1.0
            z = 1.96 if alpha == 0.05 else 2.576
            se = sigma * np.sqrt(np.arange(1, steps + 1))
            return out, out - z * se, out + z * se

        return out

    def predict(self, X: Any = None, steps: int = 10) -> np.ndarray:
        if X is not None:
            if isinstance(X, int):
                steps = X
            elif hasattr(X, "__len__"):
                steps = len(X)
        res = self.forecast(steps=steps, return_conf_int=False)
        assert isinstance(res, np.ndarray)
        return res


class SARIMAX(ARIMA):
    """Seasonal AutoRegressive Integrated Moving Average with eXogenous features."""

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 0, 1),
        seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: str = "c",
        **kwargs: Any,
    ) -> None:
        super().__init__(order=order, trend=trend, **kwargs)
        self.seasonal_order = seasonal_order
        self.exog_beta: Optional[np.ndarray] = None

    def fit(
        self, X: Any, y: Any = None, exog: Optional[np.ndarray] = None, **kwargs: Any
    ) -> SARIMAX:
        target = np.asarray(X if y is None else y, dtype=np.float64).flatten()
        if exog is not None:
            exog_arr = np.asarray(exog, dtype=np.float64)
            if exog_arr.ndim == 1:
                exog_arr = exog_arr[:, np.newaxis]
            try:
                self.exog_beta, _, _, _ = np.linalg.lstsq(exog_arr, target, rcond=None)
                target = target - exog_arr @ self.exog_beta
            except np.linalg.LinAlgError:
                self.exog_beta = np.zeros(exog_arr.shape[1])
        super().fit(target)
        return self

    def forecast(
        self,
        steps: int = 10,
        return_conf_int: bool = False,
        alpha: float = 0.05,
        exog: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        base_out = super().forecast(
            steps=steps, return_conf_int=return_conf_int, alpha=alpha
        )
        if self.exog_beta is not None and exog is not None:
            exog_arr = np.asarray(exog, dtype=np.float64)
            if exog_arr.ndim == 1:
                exog_arr = exog_arr[:, np.newaxis]
            exog_pred = exog_arr[:steps] @ self.exog_beta
            if return_conf_int and isinstance(base_out, tuple):
                mean, lower, upper = base_out
                return mean + exog_pred, lower + exog_pred, upper + exog_pred
            elif isinstance(base_out, np.ndarray):
                return base_out + exog_pred
        return base_out


class AutoARIMA(ChokkhuModel):
    """Automatic pure NumPy order selection for ARIMA via AIC/BIC optimization."""

    def __init__(
        self,
        start_p: int = 0,
        max_p: int = 3,
        start_q: int = 0,
        max_q: int = 2,
        d: int = 1,
        information_criterion: str = "aic",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.start_p = start_p
        self.max_p = max_p
        self.start_q = start_q
        self.max_q = max_q
        self.d = d
        self.ic = information_criterion
        self.best_order_: Tuple[int, int, int] = (1, d, 1)
        self.best_model_: Optional[ARIMA] = None
        self.is_fitted: bool = False

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> AutoARIMA:
        series = np.asarray(X if y is None else y, dtype=np.float64).flatten()
        best_score = float("inf")

        for p in range(self.start_p, self.max_p + 1):
            for q in range(self.start_q, self.max_q + 1):
                try:
                    model = ARIMA(p=p, d=self.d, q=q)
                    model.fit(series)
                    score = model.aic if self.ic == "aic" else model.bic
                    if np.isfinite(score) and score < best_score:
                        best_score = score
                        self.best_order_ = (p, self.d, q)
                        self.best_model_ = model
                except Exception:
                    continue

        if self.best_model_ is None:
            self.best_model_ = ARIMA(p=1, d=self.d, q=1).fit(series)
            self.best_order_ = (1, self.d, 1)

        self.is_fitted = True
        return self

    def predict(self, X: Any = None, steps: int = 10) -> np.ndarray:
        if self.best_model_ is None:
            raise ValueError("AutoARIMA is not fitted yet.")
        return self.best_model_.predict(X=X, steps=steps)

    def forecast(
        self,
        steps: int = 10,
        return_conf_int: bool = False,
        alpha: float = 0.05,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if self.best_model_ is None:
            raise ValueError("AutoARIMA is not fitted yet.")
        return self.best_model_.forecast(
            steps=steps, return_conf_int=return_conf_int, alpha=alpha, **kwargs
        )


class ExponentialSmoothing(ChokkhuModel):
    """Holt-Winters Exponential Smoothing with Trend, Damping and Seasonality."""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        seasonal_periods: Optional[int] = None,
        trend: Optional[str] = None,
        damped_trend: bool = False,
        damping_slope: float = 0.98,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = (
            float(beta)
            if beta is not None
            else (0.1 if trend in ("add", "additive") else None)
        )
        self.gamma = float(gamma) if gamma is not None else None
        self.seasonal_periods = seasonal_periods
        self.trend_type = trend
        self.damped_trend = damped_trend
        self.phi = float(damping_slope)

        self.level: float = 0.0
        self.trend: float = 0.0
        self.seasonals: np.ndarray = np.array([], dtype=np.float64)
        self.residuals: np.ndarray = np.array([], dtype=np.float64)
        self.is_fitted: bool = False

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> ExponentialSmoothing:
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
            self.seasonals = np.array(
                [float(series[i]) - self.level for i in range(P)], dtype=np.float64
            )
        else:
            self.seasonals = np.array([], dtype=np.float64)

        preds = [self.level]
        for i in range(1, n):
            val = float(series[i])
            last_level, last_trend = self.level, self.trend
            trend_comp = self.phi * last_trend if self.damped_trend else last_trend

            if len(self.seasonals) > 0 and self.seasonal_periods is not None:
                p_idx = i % self.seasonal_periods
                season = self.seasonals[p_idx]
                self.level = self.alpha * (val - season) + (1.0 - self.alpha) * (
                    last_level + trend_comp
                )
                if self.beta is not None:
                    self.trend = (
                        self.beta * (self.level - last_level)
                        + (1.0 - self.beta) * trend_comp
                    )
                if self.gamma is not None:
                    self.seasonals[p_idx] = (
                        self.gamma * (val - self.level) + (1.0 - self.gamma) * season
                    )
                preds.append(last_level + trend_comp + season)
            else:
                self.level = self.alpha * val + (1.0 - self.alpha) * (
                    last_level + trend_comp
                )
                if self.beta is not None:
                    self.trend = (
                        self.beta * (self.level - last_level)
                        + (1.0 - self.beta) * trend_comp
                    )
                preds.append(last_level + trend_comp)

        self.residuals = series - np.array(preds)
        self.is_fitted = True
        return self

    def predict(
        self, X: Any = None, steps: int = 10, return_conf_int: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if X is not None:
            if isinstance(X, int):
                steps = X
            elif hasattr(X, "__len__"):
                steps = len(X)

        forecasts: list[float] = []
        for h in range(1, steps + 1):
            t_comp = (
                sum(self.phi**j for j in range(1, h + 1)) * self.trend
                if self.damped_trend
                else h * self.trend
            )
            pred = self.level + t_comp
            if len(self.seasonals) > 0 and self.seasonal_periods is not None:
                p_idx = (len(self.seasonals) + h - 1) % self.seasonal_periods
                pred += self.seasonals[p_idx]
            forecasts.append(pred)

        f_arr = np.array(forecasts, dtype=np.float64)
        if return_conf_int:
            sigma = float(np.std(self.residuals)) if len(self.residuals) > 0 else 1.0
            se = sigma * np.sqrt(np.arange(1, steps + 1))
            return f_arr, f_arr - 1.96 * se, f_arr + 1.96 * se
        return f_arr
