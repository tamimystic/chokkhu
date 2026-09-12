"""Unit tests for Frontier 4.24: Data Valuation, Coreset Selection & Distribution Shift."""

from __future__ import annotations

import numpy as np

from chokkhu.evaluation.drift import (
    MaximumMeanDiscrepancyTest,
    PopulationStabilityIndex,
)
from chokkhu.data.valuation import (
    DataShapleyValuation,
    FacilityLocationCoresetSelector,
)


def test_mmd_test_identical_distributions_no_drift():
    """Verify MMD test yields near-zero statistic and high p-value for identical distributions."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 4))
    Y = rng.standard_normal((50, 4))

    mmd = MaximumMeanDiscrepancyTest(num_permutations=50, random_state=42)
    res = mmd.test(X, Y, alpha=0.05)

    assert not res["drift_detected"]
    assert res["p_value"] >= 0.05


def test_mmd_test_shifted_distributions_detects_drift():
    """Verify MMD test detects mean/variance covariate shift with statistically significant p-value."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 3))
    # Substantial distribution shift: mean shifted by 3.0
    Y = rng.standard_normal((50, 3)) + 3.0

    mmd = MaximumMeanDiscrepancyTest(num_permutations=50, random_state=42)
    res = mmd.test(X, Y, alpha=0.05)

    assert res["drift_detected"]
    assert res["p_value"] < 0.05
    assert res["mmd_squared"] > 0.1


def test_population_stability_index_drift_detection():
    """Verify PSI distinguishes stable features from drifted features."""
    rng = np.random.default_rng(42)
    ref = rng.standard_normal((200, 2))
    # Feature 0 is stable, Feature 1 is heavily shifted
    test = np.column_stack(
        [
            rng.standard_normal(200),
            rng.standard_normal(200) + 4.0,
        ]
    )

    psi = PopulationStabilityIndex(num_bins=10)
    res = psi.compute_dataset_psi(ref, test)

    feature_psis = res["feature_psi"]
    assert isinstance(feature_psis, list)
    assert feature_psis[0] < 0.2  # stable
    assert feature_psis[1] > 0.5  # shifted
    assert res["drift_detected"]


def test_data_shapley_valuation_additive_utility():
    """Verify Truncated Monte Carlo Data Shapley recovers additive instance values."""
    # Synthetic ground truth instance value contributions
    true_values = np.array([10.0, 5.0, 1.0, -4.0, 0.0])
    N = len(true_values)

    def utility_fn(subset_indices: np.ndarray) -> float:
        if len(subset_indices) == 0:
            return 0.0
        return float(np.sum(true_values[subset_indices]))

    shapley = DataShapleyValuation(
        num_permutations=40, truncation_tolerance=0.001, random_state=42
    )
    estimated_phi = shapley.evaluate_shapley(utility_fn, num_samples=N)

    assert len(estimated_phi) == N
    # Ranking of estimated Shapley values should match ground truth values
    assert estimated_phi[0] > estimated_phi[1] > estimated_phi[2] > estimated_phi[3]


def test_coreset_facility_location_greedy_selection():
    """Verify Facility Location core-set selection picks representative subset and sums to dataset size."""
    rng = np.random.default_rng(42)
    # 4 distinct clusters of 20 points each (total 80 points)
    c1 = rng.standard_normal((20, 2)) + np.array([10.0, 10.0])
    c2 = rng.standard_normal((20, 2)) + np.array([-10.0, 10.0])
    c3 = rng.standard_normal((20, 2)) + np.array([10.0, -10.0])
    c4 = rng.standard_normal((20, 2)) + np.array([-10.0, -10.0])
    X = np.vstack([c1, c2, c3, c4])

    selector = FacilityLocationCoresetSelector(coreset_size=8, random_state=42)
    indices, weights, coverage = selector.select(X)

    assert len(indices) == 8
    assert len(weights) == 8
    assert coverage > 0.0
    # Voronoi partition weights must sum exactly to total sample count N=80
    np.testing.assert_allclose(np.sum(weights), 80.0)
