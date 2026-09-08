from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import chokkhu


def test_missing_mean_median_mode():
    df = pd.DataFrame(
        {"num": [10.0, 20.0, 30.0, np.nan, 50.0], "cat": ["a", "b", "a", np.nan, "a"]}
    )
    cleaned_mean = chokkhu.clean(
        df, missing="mean", outliers=None, duplicates=False, fix_data_types=False
    )
    assert not cleaned_mean["num"].isna().any()
    assert cleaned_mean.loc[3, "num"] == pytest.approx(27.5)
    assert cleaned_mean.loc[3, "cat"] == "a"

    cleaned_median = chokkhu.clean(
        df, missing="median", outliers=None, duplicates=False, fix_data_types=False
    )
    assert cleaned_median.loc[3, "num"] == 25.0


def test_missing_knn_and_iterative():
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "b": [2.0, 4.0, np.nan, 8.0, 10.0, 12.0]}
    )
    cleaned_knn = chokkhu.clean(
        df,
        missing="knn",
        knn_k=2,
        outliers=None,
        duplicates=False,
        fix_data_types=False,
    )
    assert not cleaned_knn["b"].isna().any()
    assert 4.0 <= cleaned_knn.loc[2, "b"] <= 8.0

    cleaned_iter = chokkhu.clean(
        df, missing="iterative", outliers=None, duplicates=False, fix_data_types=False
    )
    assert not cleaned_iter["b"].isna().any()


def test_outlier_iqr_and_zscore():
    df = pd.DataFrame({"val": [10, 12, 11, 13, 12, 11, 1000]})
    cleaned_iqr = chokkhu.clean(
        df,
        missing=None,
        outliers="iqr",
        outlier_threshold=1.5,
        duplicates=False,
        fix_data_types=False,
    )
    assert len(cleaned_iqr) == 6
    assert 1000 not in cleaned_iqr["val"].values

    capped = chokkhu.clean(
        df, missing=None, outliers="winsorize", duplicates=False, fix_data_types=False
    )
    assert len(capped) == 7
    assert capped["val"].max() < 1000


def test_duplicates_and_dtype_fixer():
    df = pd.DataFrame(
        {
            "num_str": ["1.5", "2.5", "3.5", "1.5"],
            "bool_str": ["yes", "no", "yes", "yes"],
        }
    )
    cleaned = chokkhu.clean(
        df, missing=None, outliers=None, duplicates=True, fix_data_types=True
    )
    assert len(cleaned) == 3
    assert pd.api.types.is_float_dtype(cleaned["num_str"].dtype) or np.issubdtype(
        cleaned["num_str"].dtype, np.floating
    )
    assert (
        pd.api.types.is_bool_dtype(cleaned["bool_str"].dtype)
        or str(cleaned["bool_str"].dtype).lower() in ["bool", "boolean"]
        or cleaned["bool_str"].dtype == bool
    )
