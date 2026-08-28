from __future__ import annotations

import pandas as pd

from chokkhu.core.logger import Logger

from .dtype_fixer import fix_dtypes
from .duplicates import remove_duplicates
from .missing import handle_missing
from .outliers import handle_outliers


def clean(
    data: pd.DataFrame | str,
    missing: str = "median",
    missing_threshold: float = 0.5,
    fill_value: object = 0,
    knn_k: int = 5,
    interpolate_method: str = "linear",
    interpolate_order: int = 2,
    iterative_max_iter: int = 10,
    outliers: str = "iqr",
    outlier_threshold: float = 1.5,
    zscore_threshold: float = 3.0,
    outlier_columns: list = None,
    percentile_low: float = 0.01,
    percentile_high: float = 0.99,
    outlier_action: str = "remove",
    duplicates: bool = True,
    duplicate_subset: list = None,
    duplicate_keep: str = "first",
    fix_data_types: bool = True,
    category_threshold: int = 20,
    date_formats: list = None,
    inplace: bool = False,
    verbose: bool = True,
    save_report: bool = False,
    report_dir: str = "./chokkhu_reports/",
) -> pd.DataFrame:
    if isinstance(data, str):
        from chokkhu.io import load

        df = load(data)
    else:
        df = data if inplace else data.copy()
    initial_shape = df.shape
    if fix_data_types:
        df = fix_dtypes(
            df, category_threshold=category_threshold, date_formats=date_formats
        )
    if duplicates:
        df = remove_duplicates(df, subset=duplicate_subset, keep=duplicate_keep)
    if missing is not None:
        df = handle_missing(
            df,
            strategy=missing,
            threshold=missing_threshold,
            fill_value=fill_value,
            knn_k=knn_k,
            interpolate_method=interpolate_method,
            interpolate_order=interpolate_order,
            iterative_max_iter=iterative_max_iter,
        )
    if outliers is not None:
        df = handle_outliers(
            df,
            method=outliers,
            threshold=outlier_threshold,
            zscore_threshold=zscore_threshold,
            columns=outlier_columns,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
            action=outlier_action,
        )
    if verbose:
        Logger.info(f"Cleaned dataset: {initial_shape} -> {df.shape}")
    return df


__all__ = [
    "clean",
    "handle_missing",
    "handle_outliers",
    "remove_duplicates",
    "fix_dtypes",
]
