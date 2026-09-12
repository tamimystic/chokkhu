"""Unit tests for DeepFeatureSynthesizer (DFS) in Chokkhu Universal Pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from chokkhu.pipeline.feature_synthesis import DeepFeatureSynthesizer


def test_deep_feature_synthesis_numerical_interactions():
    """Verify DFS generates pairwise differences, ratios, and cross products without data leakage."""
    df = pd.DataFrame(
        {
            "num_a": [10.0, 20.0, 30.0, 40.0],
            "num_b": [2.0, 4.0, 5.0, 8.0],
            "num_c": [1.0, 2.0, 3.0, 4.0],
        }
    )

    dfs = DeepFeatureSynthesizer(
        include_differences=True,
        include_ratios=True,
        include_products=True,
        include_group_aggregates=False,
        include_cyclical_dates=False,
        max_generated_features=50,
    )

    df_synth = dfs.fit_transform(df)

    assert "num_a_minus_num_b" in df_synth.columns
    assert "num_a_mul_num_b" in df_synth.columns
    assert "num_a_div_num_b" in df_synth.columns
    np.testing.assert_allclose(df_synth["num_a_div_num_b"], [5.0, 5.0, 6.0, 5.0])

    # Verify zero-leakage transform on new test data
    test_df = pd.DataFrame(
        {
            "num_a": [50.0],
            "num_b": [10.0],
            "num_c": [5.0],
        }
    )
    test_synth = dfs.transform(test_df)
    assert test_synth["num_a_div_num_b"].iloc[0] == 5.0
    assert test_synth["num_a_minus_num_b"].iloc[0] == 40.0


def test_deep_feature_synthesis_group_aggregates_and_dates():
    """Verify DFS creates categorical group aggregates and cyclical date features."""
    df = pd.DataFrame(
        {
            "category": ["A", "A", "B", "B", "A"],
            "value": [100.0, 200.0, 10.0, 20.0, 300.0],
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="D"),
        }
    )

    dfs = DeepFeatureSynthesizer(
        include_differences=False,
        include_ratios=False,
        include_products=False,
        include_group_aggregates=True,
        include_cyclical_dates=True,
    )

    df_synth = dfs.fit_transform(df)

    # Check date harmonics
    assert "timestamp_dow_sin" in df_synth.columns
    assert "timestamp_month_sin" in df_synth.columns

    # Check categorical group aggregates
    assert "grp_category_value_mean" in df_synth.columns
    # Group A mean is (100+200+300)/3 = 200, Group B mean is (10+20)/2 = 15
    np.testing.assert_allclose(
        df_synth["grp_category_value_mean"].iloc[:4], [200.0, 200.0, 15.0, 15.0]
    )

    # Transform test set with unseen category (should impute overall mean)
    test_df = pd.DataFrame(
        {
            "category": ["A", "C"],
            "value": [50.0, 60.0],
            "timestamp": pd.date_range("2026-02-01", periods=2, freq="D"),
        }
    )
    test_synth = dfs.transform(test_df)
    assert not test_synth["grp_category_value_mean"].isnull().any()
