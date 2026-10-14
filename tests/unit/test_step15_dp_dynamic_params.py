import numpy as np
import pytest

from chokkhu.privacy import (
    DP_SGD,
    ExponentialMechanism,
    GaussianMechanism,
    LaplaceMechanism,
    RenyiDPAccountant,
)


def test_exponential_mechanism_selection() -> None:
    candidates = ["option_A", "option_B", "option_C"]
    utilities = [1.0, 10.0, 2.0]  # option_B has highest utility by far

    exp_mech = ExponentialMechanism(epsilon=2.0, sensitivity=1.0, random_state=42)
    assert exp_mech.get_privacy_budget() == (2.0, 0.0)

    # Over 100 trials, option_B should be chosen vast majority of times
    counts = {"option_A": 0, "option_B": 0, "option_C": 0}
    for _ in range(100):
        choice = exp_mech.select(candidates, utilities)
        counts[choice] += 1

    assert counts["option_B"] > 85

    # Test error cases
    with pytest.raises(ValueError):
        ExponentialMechanism(epsilon=-1.0)
    with pytest.raises(ValueError):
        exp_mech.select(["A"], [1.0, 2.0])


def test_renyi_dp_accountant() -> None:
    accountant = RenyiDPAccountant()
    assert accountant.get_epsilon(1e-5) == 0.0

    # 10 steps of DP-SGD with noise multiplier 1.2
    for _ in range(10):
        accountant.step_gaussian(noise_multiplier=1.2, subsampling_ratio=0.05)

    eps_10 = accountant.get_epsilon(target_delta=1e-5)
    assert 0.0 < eps_10 < 10.0

    # More steps must increase epsilon
    for _ in range(10):
        accountant.step_gaussian(noise_multiplier=1.2, subsampling_ratio=0.05)

    eps_20 = accountant.get_epsilon(target_delta=1e-5)
    assert eps_20 > eps_10


def test_laplace_and_gaussian_mechanisms() -> None:
    laplace = LaplaceMechanism(epsilon=0.5, sensitivity=2.0, random_state=42)
    val = laplace.perturb(10.0)
    assert isinstance(val, float)
    assert laplace.get_privacy_budget() == (0.5, 0.0)

    arr = laplace.perturb(np.array([1.0, 2.0]))
    assert arr.shape == (2,)

    gauss = GaussianMechanism(epsilon=1.0, delta=1e-4, sensitivity=1.0, random_state=42)
    val_g = gauss.perturb(5.0)
    assert isinstance(val_g, float)
    assert gauss.get_privacy_budget() == (1.0, 1e-4)


def test_dp_sgd_optimizer() -> None:
    opt = DP_SGD(lr=0.05, l2_norm_clip=1.0, noise_multiplier=0.5, random_state=42)
    params = {"w": np.ones((2, 2)), "b": np.zeros((2,))}
    sample_grads = [
        {"w": np.ones((2, 2)) * 0.5, "b": np.ones((2,)) * 0.5},
        {"w": np.ones((2, 2)) * 2.0, "b": np.ones((2,)) * 2.0},
    ]
    updated = opt.step(params, sample_grads)
    assert "w" in updated and "b" in updated
    privacy = opt.compute_privacy_spent(total_samples=100, batch_size=2)
    assert privacy["epsilon"] > 0.0
