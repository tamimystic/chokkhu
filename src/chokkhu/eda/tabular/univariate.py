from __future__ import annotations

import re
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats


class UnivariateAnalyzer:

    @staticmethod
    def infer_data_types(df: pd.DataFrame) -> Dict[str, Dict[str, list]]:
        types: Dict[str, Dict[str, list]] = {
            "categorical": {"ordinal": [], "nominal": []},
            "numerical": {"discrete": [], "continuous": []},
            "specialized": {"datetime": [], "text": []},
        }
        for col in df.columns:
            series = df[col]
            n_unique = series.nunique()
            dtype = series.dtype
            if pd.api.types.is_datetime64_any_dtype(dtype):
                types["specialized"]["datetime"].append(col)
            elif pd.api.types.is_numeric_dtype(dtype):
                if pd.api.types.is_float_dtype(dtype):
                    types["numerical"]["continuous"].append(col)
                elif n_unique <= 25:
                    types["numerical"]["discrete"].append(col)
                else:
                    types["numerical"]["continuous"].append(col)
            elif isinstance(dtype, pd.CategoricalDtype) and dtype.ordered:
                types["categorical"]["ordinal"].append(col)
            elif any(
                k in str(col).lower()
                for k in ["rank", "grade", "rating", "stage", "tier", "level"]
            ):
                types["categorical"]["ordinal"].append(col)
            elif n_unique <= 100:
                types["categorical"]["nominal"].append(col)
            else:
                mean_len = series.dropna().astype(str).str.len().mean()
                if not pd.isna(mean_len) and mean_len > 50:
                    types["specialized"]["text"].append(col)
                else:
                    types["categorical"]["nominal"].append(col)
        return types

    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        type_mapping = UnivariateAnalyzer.infer_data_types(df)
        results["type_mapping"] = type_mapping
        ordinal_cols = type_mapping["categorical"]["ordinal"]
        ordinal_stats = {}
        for col in ordinal_cols:
            series = df[col].dropna()
            freq = series.value_counts()
            rare_labels = freq[freq < len(series) * 0.05].index.tolist()
            ordinal_stats[col] = {
                "frequencies": freq.to_dict(),
                "rare_labels": rare_labels,
                "cardinality": len(freq),
            }
        results["ordinal_stats"] = ordinal_stats
        nominal_cols = type_mapping["categorical"]["nominal"]
        nominal_stats = {}
        for col in nominal_cols:
            series = df[col].dropna()
            freq = series.value_counts()
            rare_labels = freq[freq < len(series) * 0.05].index.tolist()
            probs = freq / len(series)
            entropy = float(-1.0 * float(np.sum(probs * np.log2(probs + 1e-09))))
            lower_counts = series.astype(str).str.lower().nunique()
            inconsistent = lower_counts < series.nunique()
            nominal_stats[col] = {
                "frequencies": freq.head(20).to_dict(),
                "rare_labels": rare_labels,
                "cardinality": len(freq),
                "shannon_entropy": entropy,
                "string_inconsistency": inconsistent,
            }
        results["nominal_stats"] = nominal_stats
        discrete_cols = type_mapping["numerical"]["discrete"]
        discrete_stats = {}
        for col in discrete_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            freq = series.value_counts().sort_index()
            discrete_stats[col] = {
                "frequencies": freq.to_dict(),
                "mean": series.mean(),
                "median": series.median(),
                "mode": series.mode().iloc[0] if not series.mode().empty else np.nan,
            }
        results["discrete_stats"] = discrete_stats
        continuous_cols = type_mapping["numerical"]["continuous"]
        continuous_stats = {}
        for col in continuous_cols:
            series = df[col].dropna()
            if len(series) < 3:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            sample = (
                series if len(series) <= 5000 else series.sample(5000, random_state=42)
            )
            try:
                if len(series) > 5000:
                    stand = (series - series.mean()) / (series.std() + 1e-09)
                    _, p_value = stats.kstest(stand, "norm")
                    is_normal = p_value > 0.05
                else:
                    _, p_value = stats.shapiro(sample)
                    is_normal = p_value > 0.05
            except Exception:
                p_value, is_normal = (None, None)
            z_scores = np.abs(stats.zscore(series))
            z_outliers = (z_scores > 3).sum()
            tukey_outliers = (
                (series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)
            ).sum()
            median = series.median()
            mad = np.median(np.abs(series - median))
            hampel_outliers = (
                (np.abs(series - median) > 3 * 1.4826 * mad).sum() if mad != 0 else 0
            )
            continuous_stats[col] = {
                "mean": series.mean(),
                "median": median,
                "variance": series.var(),
                "std": series.std(),
                "skewness": series.skew(),
                "kurtosis": series.kurtosis(),
                "iqr": iqr,
                "shapiro_p_value": p_value,
                "is_normal": is_normal,
                "outliers_zscore": int(z_outliers),
                "outliers_tukey": int(tukey_outliers),
                "outliers_hampel": int(hampel_outliers),
            }
        results["continuous_stats"] = continuous_stats
        datetime_cols = type_mapping["specialized"]["datetime"]
        datetime_stats = {}
        for col in datetime_cols:
            series = df[col].dropna()
            if series.empty:
                continue
            sorted_series = series.sort_values()
            diffs = sorted_series.diff().dt.total_seconds()
            half = len(series) // 2
            v1, v2 = (
                series.iloc[:half].astype(np.int64).var(),
                series.iloc[half:].astype(np.int64).var(),
            )
            pseudo_stationary = abs(v1 - v2) / (v1 + 1e-09) < 0.2 if v1 != 0 else True
            monthly = series.groupby(series.dt.to_period("M")).size()
            trend = monthly.rolling(window=3, min_periods=1).mean().to_dict()
            datetime_stats[col] = {
                "min_date": str(series.min()),
                "max_date": str(series.max()),
                "unique_dates": series.nunique(),
                "max_gap_seconds": diffs.max() if not diffs.empty else 0,
                "pseudo_stationary": pseudo_stationary,
                "monthly_trend": {str(k): v for k, v in trend.items()},
            }
        results["datetime_stats"] = datetime_stats
        text_cols = type_mapping["specialized"]["text"]
        text_stats = {}
        for col in text_cols:
            series = df[col].dropna().astype(str)
            char_counts = series.str.len()
            word_counts = series.apply(lambda x: len(re.findall("\\w+", x)))
            from collections import Counter

            unigrams: Counter = Counter()
            bigrams: Counter = Counter()
            for text in series:
                words = re.findall("\\b\\w+\\b", text.lower())
                unigrams.update(words)
                if len(words) >= 2:
                    bigrams.update(zip(words[:-1], words[1:]))
            text_stats[col] = {
                "mean_char_length": char_counts.mean(),
                "max_char_length": char_counts.max(),
                "mean_word_count": word_counts.mean(),
                "max_word_count": word_counts.max(),
                "top_unigrams": dict(unigrams.most_common(10)),
                "top_bigrams": {" ".join(k): v for k, v in bigrams.most_common(10)},
            }
        results["text_stats"] = text_stats
        return results
