from __future__ import annotations

import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean = np.nanmean(X_arr, axis=0)
        self.std = np.nanstd(X_arr, axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return (X_arr - self.mean) / self.std

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr * self.std + self.mean


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min = None
        self.max = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        self.min = np.nanmin(X_arr, axis=0)
        self.max = np.nanmax(X_arr, axis=0)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        rng = self.max - self.min
        rng[rng == 0] = 1.0
        norm = (X_arr - self.min) / rng
        return (
            norm * (self.feature_range[1] - self.feature_range[0])
            + self.feature_range[0]
        )

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        rng = self.max - self.min
        rng[rng == 0] = 1.0
        unscaled = (X_arr - self.feature_range[0]) / (
            self.feature_range[1] - self.feature_range[0]
        )
        return unscaled * rng + self.min


class RobustScaler:
    def __init__(self):
        self.median = None
        self.iqr = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        self.median = np.nanmedian(X_arr, axis=0)
        q75 = np.nanpercentile(X_arr, 75, axis=0)
        q25 = np.nanpercentile(X_arr, 25, axis=0)
        self.iqr = q75 - q25
        self.iqr[self.iqr == 0] = 1.0
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return (X_arr - self.median) / self.iqr

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr * self.iqr + self.median


class MaxAbsScaler:
    def __init__(self):
        self.max_abs = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        self.max_abs = np.nanmax(np.abs(X_arr), axis=0)
        self.max_abs[self.max_abs == 0] = 1.0
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return X_arr / self.max_abs

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return np.asarray(X, dtype=np.float64) * self.max_abs


class L2Scaler:
    def fit(self, X):
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        norms = np.linalg.norm(X_arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X_arr / norms

    def fit_transform(self, X):
        return self.transform(X)


class PowerScaler:
    """Yeo-Johnson Power Transformation from scratch for stabilizing variance and normalizing."""

    def __init__(self):
        self.lambdas = None

    def _yeo_johnson_transform(self, x: np.ndarray, lmbda: float) -> np.ndarray:
        out = np.zeros_like(x)
        pos = x >= 0
        neg = ~pos
        if abs(lmbda) > 1e-5:
            out[pos] = ((x[pos] + 1.0) ** lmbda - 1.0) / lmbda
        else:
            out[pos] = np.log1p(x[pos])

        if abs(lmbda - 2.0) > 1e-5:
            out[neg] = -((-x[neg] + 1.0) ** (2.0 - lmbda) - 1.0) / (2.0 - lmbda)
        else:
            out[neg] = -np.log1p(-x[neg])
        return out

    def fit(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        n_features = X_arr.shape[1]
        self.lambdas = np.zeros(n_features)
        # Search best lambda over grid
        candidates = np.linspace(-2.0, 2.0, 41)
        for j in range(n_features):
            col = X_arr[:, j]
            best_lmbda = 1.0
            best_skew = float("inf")
            for lmbda in candidates:
                t = self._yeo_johnson_transform(col, lmbda)
                m3 = np.mean((t - np.mean(t)) ** 3)
                s3 = (np.std(t) + 1e-8) ** 3
                skew = abs(m3 / s3)
                if skew < best_skew:
                    best_skew = skew
                    best_lmbda = lmbda
            self.lambdas[j] = best_lmbda
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        out = np.zeros_like(X_arr)
        for j in range(X_arr.shape[1]):
            out[:, j] = self._yeo_johnson_transform(X_arr[:, j], self.lambdas[j])
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class QuantileScaler:
    """Quantile Scaler mapping empirical distributions to uniform or normal quantiles."""

    def __init__(self, output_distribution="uniform", n_quantiles=100):
        self.output_distribution = output_distribution
        self.n_quantiles = n_quantiles
        self.quantiles = None
        self.references = None

    def fit(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        n_q = min(self.n_quantiles, len(X_arr))
        percentiles = np.linspace(0, 100, n_q)
        self.quantiles = np.percentile(X_arr, percentiles, axis=0)
        self.references = percentiles / 100.0
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        out = np.zeros_like(X_arr)
        for j in range(X_arr.shape[1]):
            out[:, j] = np.interp(X_arr[:, j], self.quantiles[:, j], self.references)
        if self.output_distribution == "normal":
            from scipy import stats

            out = stats.norm.ppf(np.clip(out, 1e-4, 1.0 - 1e-4))
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def get_scaler(name: str, **kwargs):
    scalers = {
        "standard": StandardScaler,
        "minmax": lambda: MinMaxScaler(
            feature_range=kwargs.get("feature_range", (0, 1))
        ),
        "robust": RobustScaler,
        "maxabs": MaxAbsScaler,
        "l2": L2Scaler,
        "power": PowerScaler,
        "quantile": lambda: QuantileScaler(
            output_distribution=kwargs.get("output_distribution", "uniform")
        ),
    }
    if name not in scalers:
        raise ValueError(f"Unknown scaler: {name}")
    return scalers[name]()
