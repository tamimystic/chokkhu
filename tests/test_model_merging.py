"""Unit tests for Frontier 4.16: Model Merging, Weight Surgery & Task Arithmetic."""

import numpy as np
import pytest

from chokkhu.models.merging import (
    TIESMerging,
    DARE,
    SLERP,
    RegMean,
    FrankWolfeEnsemble,
)


def test_ties_trimming_fraction():
    """Verify TIES trimming retains top k fraction and zeroes out lower magnitudes."""
    ties = TIESMerging(density=0.4)
    tensor = np.array([-10.0, 1.0, 2.0, -8.0, 0.5])  # size 5, top 40% = 2 items: -10.0 and -8.0
    trimmed = ties.trim_tensor(tensor)
    
    assert trimmed[0] == -10.0
    assert trimmed[3] == -8.0
    assert trimmed[1] == 0.0
    assert trimmed[2] == 0.0
    assert trimmed[4] == 0.0


def test_ties_sign_consensus_and_disjoint_merge():
    """Verify majority sign election and disjoint merge filtering."""
    ties = TIESMerging(density=1.0, scaling_factor=1.0)
    base = np.array([10.0, 10.0])
    
    # Task 1: [+2.0, -3.0]
    task1 = np.array([12.0, 7.0])
    # Task 2: [+4.0, +1.0]
    task2 = np.array([14.0, 11.0])
    # Task 3: [-1.0, -4.0]
    task3 = np.array([9.0, 6.0])

    # For coord 0: deltas are [+2.0, +4.0, -1.0] -> sum = +5.0 (majority positive)
    # Agreeing tasks: Task 1 (+2.0) and Task 2 (+4.0) -> mean = +3.0.
    # For coord 1: deltas are [-3.0, +1.0, -4.0] -> sum = -6.0 (majority negative)
    # Agreeing tasks: Task 1 (-3.0) and Task 3 (-4.0) -> mean = -3.5.
    
    merged = ties.merge_tensors(base, [task1, task2, task3])
    expected = base + np.array([3.0, -3.5])
    np.testing.assert_allclose(merged, expected, rtol=1e-5)


def test_ties_weight_dict_merging():
    """Test dictionary-level parameter merging with TIES."""
    ties = TIESMerging(density=0.5, scaling_factor=1.0)
    base_dict = {
        "layer1.weight": np.ones((4, 4)),
        "layer1.bias": np.zeros(4),
    }
    task1_dict = {
        "layer1.weight": np.ones((4, 4)) + 0.5,
        "layer1.bias": np.full(4, 1.0),
    }
    task2_dict = {
        "layer1.weight": np.ones((4, 4)) + 1.0,
        "layer1.bias": np.full(4, 2.0),
    }

    merged = ties.merge_weight_dicts(base_dict, [task1_dict, task2_dict])
    assert "layer1.weight" in merged
    assert "layer1.bias" in merged
    assert merged["layer1.weight"].shape == (4, 4)
    assert merged["layer1.bias"].shape == (4,)
    # Output should be higher than base weights
    assert np.all(merged["layer1.bias"] > base_dict["layer1.bias"])


def test_dare_unbiased_expectation():
    """Verify DARE preserves expectation of delta across many samples."""
    delta = np.full((1000, 1000), 5.0)
    dare = DARE(drop_rate=0.7, seed=42)
    sparsified = dare.sparsify_task_vector(delta)

    # Retained proportion should be ~ 30%
    sparsity = np.mean(sparsified == 0.0)
    assert 0.65 < sparsity < 0.75

    # Empirical mean should closely approximate original value (5.0)
    empirical_mean = np.mean(sparsified)
    np.testing.assert_allclose(empirical_mean, 5.0, rtol=0.05)


def test_dare_merge_linear_and_ties():
    """Test DARE in linear and ties merge modes."""
    base = np.zeros((10, 10))
    task1 = np.ones((10, 10)) * 2.0
    task2 = np.ones((10, 10)) * 4.0

    dare_linear = DARE(drop_rate=0.5, mode="linear", seed=123)
    merged_lin = dare_linear.merge_tensors(base, [task1, task2])
    assert merged_lin.shape == (10, 10)
    assert np.all(merged_lin >= 0.0)

    dare_ties = DARE(drop_rate=0.5, mode="ties", seed=123)
    merged_ties = dare_ties.merge_tensors(base, [task1, task2])
    assert merged_ties.shape == (10, 10)
    assert np.all(merged_ties >= 0.0)


def test_slerp_hypersphere_interpolation():
    """Verify SLERP angular interpolation and endpoint boundaries."""
    slerp = SLERP(t=0.5)
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])

    # At t=0.5, orthogonal unit vectors interpolate to (1/sqrt(2), 1/sqrt(2), 0)
    interp = slerp.interpolate_tensors(v1, v2, t=0.5)
    expected = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0])
    np.testing.assert_allclose(interp, expected, atol=1e-6)

    # Boundaries
    np.testing.assert_allclose(slerp.interpolate_tensors(v1, v2, t=0.0), v1)
    np.testing.assert_allclose(slerp.interpolate_tensors(v1, v2, t=1.0), v2)


def test_slerp_collinear_fallback():
    """Verify SLERP handles collinear vectors gracefully without NaN."""
    slerp = SLERP()
    v1 = np.array([2.0, 4.0, 6.0])
    v2 = np.array([4.0, 8.0, 12.0])  # Parallel collinear

    interp = slerp.interpolate_tensors(v1, v2, t=0.5)
    np.testing.assert_allclose(interp, (v1 + v2) / 2.0, rtol=1e-5)
    assert not np.isnan(interp).any()


def test_regmean_closed_form_exactness():
    """Verify RegMean closed form solution minimizes expected activation error."""
    regmean = RegMean(eps=1e-6)
    
    # Feature dimension 4, Output dimension 3
    d_in, d_out = 4, 3
    rng = np.random.default_rng(42)
    
    X1 = rng.standard_normal((50, d_in))
    X2 = rng.standard_normal((50, d_in))
    
    G1 = regmean.compute_gram_matrix(X1)
    G2 = regmean.compute_gram_matrix(X2)
    
    W1 = rng.standard_normal((d_in, d_out))
    W2 = rng.standard_normal((d_in, d_out))

    merged_W = regmean.merge_layers([W1, W2], [G1, G2])
    assert merged_W.shape == (d_in, d_out)

    # Manual analytic solution: (G1 + G2)^(-1) (G1 W1 + G2 W2)
    expected_W = np.linalg.pinv(G1 + G2 + 1e-6 * np.eye(d_in)) @ (G1 @ W1 + G2 @ W2)
    np.testing.assert_allclose(merged_W, expected_W, rtol=1e-5)


def test_regmean_diagonal_approximation():
    """Test RegMean with diagonal Gram approximation."""
    regmean = RegMean(diagonal_approx=True, eps=1e-5)
    G1 = np.diag([2.0, 4.0])
    G2 = np.diag([6.0, 8.0])
    W1 = np.array([[1.0], [2.0]])
    W2 = np.array([[3.0], [4.0]])

    # Row 0: (2*1 + 6*3) / (2 + 6) = 20 / 8 = 2.5
    # Row 1: (4*2 + 8*4) / (4 + 8) = 40 / 12 = 3.333333
    merged_W = regmean.merge_layers([W1, W2], [G1, G2])
    expected_W = np.array([[20.0 / 8.0], [40.0 / 12.0]])
    np.testing.assert_allclose(merged_W, expected_W, rtol=1e-4)


def test_frank_wolfe_simplex_optimization():
    """Verify Frank-Wolfe optimizes mixture weights on simplex to match target."""
    rng = np.random.default_rng(42)
    N = 100
    
    # 3 model predictions
    p1 = np.linspace(0, 10, N)
    p2 = np.linspace(10, 0, N)
    p3 = np.full(N, 5.0)

    # Target is exact mixture: 0.5 * p1 + 0.3 * p2 + 0.2 * p3
    y_true = 0.5 * p1 + 0.3 * p2 + 0.2 * p3

    fw = FrankWolfeEnsemble(max_iters=100, loss="mse")
    fw.fit([p1, p2, p3], y_true)

    # Assert weights are on probability simplex
    assert fw.weights_ is not None
    assert len(fw.weights_) == 3
    assert np.all(fw.weights_ >= 0.0)
    np.testing.assert_allclose(np.sum(fw.weights_), 1.0, atol=1e-6)

    # Predict should closely match target
    preds = fw.predict([p1, p2, p3])
    np.testing.assert_allclose(preds, y_true, atol=0.1)


def test_frank_wolfe_weight_dict_merging():
    """Test Frank-Wolfe convex combination of weight dictionaries."""
    fw = FrankWolfeEnsemble(max_iters=20)
    fw.weights_ = np.array([0.25, 0.75])

    dict1 = {"w": np.zeros((3, 3))}
    dict2 = {"w": np.ones((3, 3)) * 4.0}

    merged = fw.merge_weight_dicts([dict1, dict2])
    np.testing.assert_allclose(merged["w"], np.ones((3, 3)) * 3.0)
