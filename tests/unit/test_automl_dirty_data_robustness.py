from __future__ import annotations

import numpy as np
import pandas as pd


import chokkhu as ck
from chokkhu.cleaning.dtype_fixer import fix_dtypes
from chokkhu.cleaning.missing import handle_missing
from chokkhu.pipeline import PipelineResult


def test_clean_infinite_and_all_nan_columns():
    df = pd.DataFrame(
        {
            "normal_num": [1.0, 2.0, np.nan, 4.0, 5.0],
            "inf_num": [10.0, np.inf, 30.0, -np.inf, 50.0],
            "all_nan": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "cat_with_nan": ["A", "B", np.nan, "A", "B"],
        }
    )

    cleaned = ck.clean(
        df, missing="median", outliers=None, duplicates=False, fix_data_types=False
    )
    # Check that infinite values are replaced and imputed
    assert not np.isinf(cleaned["inf_num"]).any()
    assert not cleaned["inf_num"].isna().any()
    assert cleaned.loc[1, "inf_num"] == 30.0

    # All-NaN column should be safely filled with 0 (fill_value) without error
    assert not cleaned["all_nan"].isna().any()
    assert (cleaned["all_nan"] == 0).all()


def test_fix_dtypes_boolean_with_nans():
    df = pd.DataFrame(
        {
            "bool_dirty": ["yes", "no", np.nan, "True", "FALSE", "1", "0"],
            "num_dirty": ["1.2", "3.4", np.nan, "5.6", "7.8", "9.0", "10.1"],
        }
    )

    fixed = fix_dtypes(df)
    assert not pd.api.types.is_object_dtype(fixed["bool_dirty"].dtype)
    assert not pd.api.types.is_object_dtype(fixed["num_dirty"].dtype)
    assert fixed.loc[0, "bool_dirty"] is True or fixed.loc[0, "bool_dirty"] == 1
    assert pd.isna(fixed.loc[2, "bool_dirty"])


def test_missing_imputation_strategies_on_messy_slices():
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan, 5.0],
            "b": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "c": ["x", "y", np.nan, "x", np.nan],
        }
    )

    # KNN Imputation with 100% missing column fallback
    knn_res = handle_missing(df, strategy="knn", knn_k=2, fill_value=0.0)
    assert not knn_res["a"].isna().any()
    assert not knn_res["b"].isna().any()
    assert (knn_res["b"] == 0.0).all()

    # Iterative Imputation with 100% missing column fallback
    iter_res = handle_missing(
        df, strategy="iterative", iterative_max_iter=3, fill_value=0.0
    )
    assert not iter_res["a"].isna().any()
    assert not iter_res["b"].isna().any()


def test_preprocessor_unseen_categories_and_missing_features():
    train_df = pd.DataFrame(
        {
            "num_feat": [10.0, 20.0, 30.0, 40.0],
            "cat_feat": ["apple", "banana", "apple", "cherry"],
            "target": [0, 1, 0, 1],
        }
    )

    processed_train, state = ck.preprocess(
        train_df, target="target", scale="standard", encode="onehot"
    )

    # Inference data with unseen category ("dragonfruit") and extra/missing columns
    test_df = pd.DataFrame(
        {
            "num_feat": [25.0, 35.0],
            "cat_feat": ["dragonfruit", "banana"],  # dragonfruit is unseen
        }
    )

    processed_test = state.transform(test_df)
    assert len(processed_test) == 2
    # Ensure onehot columns are present and consistent
    assert all(c in processed_test.columns for c in state.feature_names_out)
    assert processed_test.shape[1] == len(state.feature_names_out)


def test_end_to_end_pipeline_dirty_data():
    np.random.seed(42)
    n = 120

    # Messy dataset with infs, nans, constant columns, and noisy target
    raw_df = pd.DataFrame(
        {
            "feat_num1": np.random.randn(n) * 10,
            "feat_num2": [
                np.inf if i == 5 else (-np.inf if i == 15 else np.random.randn() * 5)
                for i in range(n)
            ],
            "feat_nan": [np.nan if i % 3 == 0 else np.random.randn() for i in range(n)],
            "feat_const": [42.0] * n,  # Constant zero-variance feature
            "feat_cat": np.random.choice(["cat1", "cat2", "cat3"], size=n),
            "target": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
        }
    )

    result = ck.pipeline(
        data=raw_df,
        target="target",
        model="random_forest",
        task="classification",
        clean=True,
        preprocess={
            "scale": "standard",
            "encode": "onehot",
            "select_features": "variance",
        },
        test_size=0.25,
        random_state=42,
        verbose=False,
    )

    assert isinstance(result, PipelineResult)
    assert result.model is not None
    assert "accuracy" in result.evaluation

    # Test inference on dirty unseen data
    unseen_test = pd.DataFrame(
        {
            "feat_num1": [1.0, 2.0],
            "feat_num2": [np.nan, 3.0],
            "feat_nan": [0.5, np.nan],
            "feat_const": [42.0, 42.0],
            "feat_cat": ["cat1", "unseen_cat"],
        }
    )
    preds = result.predict(unseen_test)
    assert len(preds) == 2
    assert set(preds).issubset({0, 1})
