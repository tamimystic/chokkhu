"""Deep Feature Synthesis (DFS) & Automated Feature Engineering in pure NumPy/Pandas.

Implements automated synthesis of cross-feature interactions, ratios, differences,
cyclical date-time harmonic expansions, and leak-free categorical group aggregations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


class DeepFeatureSynthesizer:
    """Automated Deep Feature Synthesis (DFS) Transformer for Tabular Data.

    Generates rich predictive feature crosses with zero data leakage:
    - Cross-feature differences: x_i - x_j
    - Cross-feature ratios: x_i / (x_j + eps)
    - Polynomial cross-products: x_i * x_j
    - Cyclical harmonic date-time transforms: sin(2*pi*t/T), cos(2*pi*t/T)
    - Categorical group-by statistics: group mean, std, min, max
    """

    def __init__(
        self,
        include_differences: bool = True,
        include_ratios: bool = True,
        include_products: bool = True,
        include_group_aggregates: bool = True,
        include_cyclical_dates: bool = True,
        max_generated_features: int = 100,
        eps: float = 1e-6,
    ) -> None:
        self.include_differences = include_differences
        self.include_ratios = include_ratios
        self.include_products = include_products
        self.include_group_aggregates = include_group_aggregates
        self.include_cyclical_dates = include_cyclical_dates
        self.max_generated_features = int(max_generated_features)
        self.eps = float(eps)

        self.num_cols: List[str] = []
        self.cat_cols: List[str] = []
        self.date_cols: List[str] = []
        self.group_stats_: Dict[str, Any] = {}
        self.fitted_: bool = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "DeepFeatureSynthesizer":
        """Fit feature synthesis statistics (e.g. group aggregates) strictly on training data."""
        df = pd.DataFrame(X).copy()

        self.num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        self.date_cols = df.select_dtypes(
            include=["datetime", "datetime64"]
        ).columns.tolist()

        # Fit categorical group aggregates
        self.group_stats_ = {}
        if self.include_group_aggregates and self.cat_cols and self.num_cols:
            for cat_col in self.cat_cols[:3]:  # Top categorical features
                self.group_stats_[cat_col] = {}
                for num_col in self.num_cols[:5]:  # Top numerical features
                    grouped = df.groupby(cat_col)[num_col].agg(
                        ["mean", "std", "min", "max"]
                    )
                    self.group_stats_[cat_col][num_col] = grouped.to_dict()

        self.fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Synthesize features on train/test dataframe using fitted statistics with zero leakage."""
        if not self.fitted_:
            raise RuntimeError(
                "DeepFeatureSynthesizer must be fitted before calling transform()"
            )

        df = pd.DataFrame(X).copy()
        new_features: Dict[str, Union[pd.Series, np.ndarray]] = {}
        feat_count = 0

        # 1. Cyclical Harmonic Date Transformations
        if self.include_cyclical_dates and self.date_cols:
            for d_col in self.date_cols:
                dt_series = pd.to_datetime(df[d_col])
                # Day of week (0-6)
                dow = dt_series.dt.dayofweek
                new_features[f"{d_col}_dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
                new_features[f"{d_col}_dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)

                # Month (1-12)
                month = dt_series.dt.month
                new_features[f"{d_col}_month_sin"] = np.sin(
                    2.0 * np.pi * (month - 1) / 12.0
                )
                new_features[f"{d_col}_month_cos"] = np.cos(
                    2.0 * np.pi * (month - 1) / 12.0
                )

                # Hour (0-23)
                hour = dt_series.dt.hour
                new_features[f"{d_col}_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
                new_features[f"{d_col}_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
                feat_count += 6

        # 2. Pairwise Cross-Features (Differences, Ratios, Products)
        if self.num_cols:
            n_num = len(self.num_cols)
            for i in range(n_num):
                for j in range(i + 1, n_num):
                    if feat_count >= self.max_generated_features:
                        break

                    c1, c2 = self.num_cols[i], self.num_cols[j]
                    v1 = df[c1].to_numpy(dtype=np.float64)
                    v2 = df[c2].to_numpy(dtype=np.float64)

                    if self.include_differences:
                        new_features[f"{c1}_minus_{c2}"] = v1 - v2
                        feat_count += 1

                    if (
                        self.include_products
                        and feat_count < self.max_generated_features
                    ):
                        new_features[f"{c1}_mul_{c2}"] = v1 * v2
                        feat_count += 1

                    if self.include_ratios and feat_count < self.max_generated_features:
                        denom = np.where(
                            np.abs(v2) < self.eps, np.sign(v2 + 1e-12) * self.eps, v2
                        )
                        new_features[f"{c1}_div_{c2}"] = v1 / denom
                        feat_count += 1

        # 3. Categorical Group Aggregations
        if self.include_group_aggregates and self.group_stats_:
            for cat_col, num_dict in self.group_stats_.items():
                if cat_col not in df.columns:
                    continue
                cat_vals = df[cat_col]
                for num_col, stat_dict in num_dict.items():
                    if feat_count >= self.max_generated_features:
                        break
                    for stat_name, mapping in stat_dict.items():
                        if feat_count >= self.max_generated_features:
                            break
                        mapped = cat_vals.map(mapping)
                        # Impute unseen categories with overall mean
                        if mapped.isnull().any():
                            overall_mean = float(np.nanmean(list(mapping.values())))
                            mapped = mapped.fillna(overall_mean)
                        new_features[f"grp_{cat_col}_{num_col}_{stat_name}"] = (
                            mapped.to_numpy()
                        )
                        feat_count += 1

        # Concatenate generated features
        if new_features:
            synth_df = pd.DataFrame(new_features, index=df.index)
            df_out = pd.concat([df, synth_df], axis=1)
        else:
            df_out = df

        return df_out

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Fit and transform in a single leak-free operation."""
        return self.fit(X, y).transform(X)
