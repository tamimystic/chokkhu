from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd


class PolynomialFeatures:
    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = False,
    ) -> None:
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.n_input_features_: int | None = None
        self.feature_combinations_: list[tuple[int, ...]] | None = None

    def fit(self, X: Any) -> PolynomialFeatures:
        arr = np.asarray(X)
        self.n_input_features_ = arr.shape[1]
        combo_list: list[tuple[int, ...]] = []

        if self.include_bias:
            combo_list.append(())

        for d in range(1, self.degree + 1):
            if self.interaction_only:
                for c in itertools.combinations(range(self.n_input_features_), d):
                    combo_list.append(c)
            else:
                for c in itertools.combinations_with_replacement(
                    range(self.n_input_features_), d
                ):
                    combo_list.append(c)

        self.feature_combinations_ = combo_list
        return self

    def transform(self, X: Any) -> np.ndarray:
        if self.feature_combinations_ is None or self.n_input_features_ is None:
            raise ValueError("PolynomialFeatures instance is not fitted yet.")
        arr = np.asarray(X, dtype=np.float64)
        n_samples = arr.shape[0]
        output = np.empty(
            (n_samples, len(self.feature_combinations_)), dtype=np.float64
        )

        for i, combo in enumerate(self.feature_combinations_):
            if len(combo) == 0:
                output[:, i] = 1.0
            else:
                output[:, i] = np.prod(arr[:, combo], axis=1)

        return output

    def fit_transform(self, X: Any) -> np.ndarray:
        return self.fit(X).transform(X)


class LogTransformer:
    def __init__(self, columns: list[str] | None = None, base: str = "e") -> None:
        self.columns = columns
        self.base = base

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        res = df.copy()
        target_cols = (
            self.columns or res.select_dtypes(include=[np.number]).columns.tolist()
        )

        for c in target_cols:
            if c in res.columns and pd.api.types.is_numeric_dtype(res[c]):
                min_val = float(res[c].min())
                # Shift only negative numbers to zero, avoiding double-shift
                shift = abs(min_val) if min_val < 0 else 0.0
                vals = res[c].to_numpy(dtype=np.float64) + shift
                if self.base == "10":
                    res[c] = np.log10(vals + 1.0)
                elif self.base == "2":
                    res[c] = np.log2(vals + 1.0)
                else:
                    res[c] = np.log1p(vals)
        return res


class BinningTransformer:
    def __init__(
        self,
        n_bins: int = 5,
        strategy: str = "uniform",
        columns: list[str] | None = None,
    ) -> None:
        self.n_bins = n_bins
        self.strategy = strategy
        self.columns = columns
        self.bin_edges_: dict[str, np.ndarray] = {}

    def fit(self, df: pd.DataFrame) -> BinningTransformer:
        target_cols = (
            self.columns or df.select_dtypes(include=[np.number]).columns.tolist()
        )
        for c in target_cols:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                col_data = df[c].dropna().to_numpy(dtype=np.float64)
                if len(col_data) == 0:
                    continue
                if self.strategy == "quantile":
                    quantiles = np.linspace(0, 1, self.n_bins + 1)
                    edges = np.asarray(np.percentile(col_data, quantiles * 100.0))
                else:
                    edges = np.linspace(col_data.min(), col_data.max(), self.n_bins + 1)
                edges = np.unique(edges)
                self.bin_edges_[c] = edges
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        res = df.copy()
        for c, edges in self.bin_edges_.items():
            if c in res.columns and len(edges) > 1:
                res[c] = pd.cut(res[c], bins=edges, include_lowest=True, labels=False)
        return res

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
