from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any
from scipy import stats


class VarianceThresholdSelector:
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self.selected_columns: list[str] = []

    def fit(self, df: pd.DataFrame):
        num_df = df.select_dtypes(include=[np.number])
        variances = num_df.var()
        self.selected_columns = variances[variances >= self.threshold].index.tolist()
        return self

    def transform(self, df: pd.DataFrame):
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        keep = [c for c in num_cols if c in self.selected_columns] + cat_cols
        return df[keep]

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)


class CorrelationFilterSelector:
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.dropped_columns: list[str] = []

    def fit(self, df: pd.DataFrame, target: pd.Series = None):
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return self
        corr_matrix = num_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        drop_cols = set()
        for col in upper.columns:
            high_corr = upper.index[upper[col] > self.threshold].tolist()
            if high_corr:
                if target is not None and pd.api.types.is_numeric_dtype(target):
                    target_corr = {
                        c: (
                            abs(np.corrcoef(df[c], target)[0, 1])
                            if df[c].std() > 0
                            else 0
                        )
                        for c in [col] + high_corr
                    }
                    worst = min(target_corr, key=target_corr.get)
                    drop_cols.add(worst)
                else:
                    drop_cols.add(col)
        self.dropped_columns = list(drop_cols)
        return self

    def transform(self, df: pd.DataFrame):
        return df.drop(columns=[c for c in self.dropped_columns if c in df.columns])

    def fit_transform(self, df: pd.DataFrame, target: pd.Series = None):
        return self.fit(df, target).transform(df)


class MutualInfoSelector:
    def __init__(self, k=10):
        self.k = k
        self.selected_columns: list[str] = []

    def _calc_mi(self, x: pd.Series, y: pd.Series):
        if pd.api.types.is_numeric_dtype(x) and x.nunique() > 10:
            x_binned = pd.qcut(x, q=10, duplicates="drop").astype(str)
        else:
            x_binned = x.astype(str)
        c_xy = pd.crosstab(x_binned, y.astype(str))
        p_xy = c_xy / max(1, c_xy.sum().sum())
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        mi = 0.0
        for i in p_xy.index:
            for j in p_xy.columns:
                if p_xy.loc[i, j] > 0:
                    mi += p_xy.loc[i, j] * np.log2(
                        p_xy.loc[i, j] / (p_x[i] * p_y[j] + 1e-12)
                    )
        return mi

    def fit(self, df: pd.DataFrame, target: pd.Series):
        scores = {}
        for col in df.columns:
            try:
                scores[col] = self._calc_mi(
                    df[col].dropna(), target.loc[df[col].dropna().index]
                )
            except Exception:
                scores[col] = 0.0
        sorted_cols = sorted(scores, key=scores.get, reverse=True)
        self.selected_columns = sorted_cols[: min(self.k, len(sorted_cols))]
        return self

    def transform(self, df: pd.DataFrame):
        return df[[c for c in self.selected_columns if c in df.columns]]

    def fit_transform(self, df: pd.DataFrame, target: pd.Series):
        return self.fit(df, target).transform(df)


class ANOVASelector:
    def __init__(self, k=10):
        self.k = k
        self.selected_columns: list[str] = []

    def fit(self, df: pd.DataFrame, target: pd.Series):
        num_cols = df.select_dtypes(include=[np.number]).columns
        scores = {}
        classes = target.unique()
        for col in num_cols:
            groups = [df.loc[target == c, col].dropna() for c in classes]
            groups = [g for g in groups if len(g) > 1]
            if len(groups) > 1:
                try:
                    f_val, _ = stats.f_oneway(*groups)
                    scores[col] = f_val if not np.isnan(f_val) else 0.0
                except Exception:
                    scores[col] = 0.0
            else:
                scores[col] = 0.0
        sorted_cols = sorted(scores, key=scores.get, reverse=True)
        self.selected_columns = sorted_cols[: min(self.k, len(sorted_cols))]
        return self

    def transform(self, df: pd.DataFrame):
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        keep = [c for c in self.selected_columns if c in df.columns] + cat_cols
        return df[keep]

    def fit_transform(self, df: pd.DataFrame, target: pd.Series):
        return self.fit(df, target).transform(df)


class RFESelector:
    """Recursive Feature Elimination (RFE) from scratch."""

    def __init__(self, k: int = 5, step: int = 1):
        self.k = k
        self.step = step
        self.selected_columns: list[str] = []

    def fit(self, df: pd.DataFrame, target: pd.Series):
        from chokkhu.models.ml.linear_regression import LinearRegression
        from chokkhu.models.ml.logistic_regression import LogisticRegression

        num_cols = list(df.select_dtypes(include=[np.number]).columns)
        if len(num_cols) <= self.k:
            self.selected_columns = num_cols
            return self

        current_cols = list(num_cols)
        y_arr = target.to_numpy()
        is_class = not np.issubdtype(y_arr.dtype, np.floating)

        while len(current_cols) > self.k:
            X_curr = df[current_cols].to_numpy(dtype=np.float64)
            if is_class:
                model: Any = LogisticRegression(epochs=100, learning_rate=0.01)
                model.fit(X_curr, y_arr)
                weights = np.abs(model.weights)
            else:
                model = LinearRegression(method="normal_equation")
                model.fit(X_curr, y_arr)
                weights = np.abs(model.weights)

            # Find feature with minimum weight
            n_to_remove = min(self.step, len(current_cols) - self.k)
            worst_indices = np.argsort(weights)[:n_to_remove]
            current_cols = [
                c for i, c in enumerate(current_cols) if i not in worst_indices
            ]

        self.selected_columns = current_cols
        return self

    def transform(self, df: pd.DataFrame):
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        keep = [c for c in self.selected_columns if c in df.columns] + cat_cols
        return df[keep]

    def fit_transform(self, df: pd.DataFrame, target: pd.Series):
        return self.fit(df, target).transform(df)
