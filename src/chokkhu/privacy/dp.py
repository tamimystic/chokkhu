"""Differential Privacy mechanisms, DP-SGD optimizer, and privacy accounting."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class LaplaceMechanism:
    """Laplace Mechanism for epsilon-differential privacy query perturbation.

    Adds calibrated noise from Laplace(0, sensitivity / epsilon).
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        sensitivity: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        if epsilon <= 0.0:
            raise ValueError(f"Epsilon must be strictly positive, got {epsilon}")
        if sensitivity <= 0.0:
            raise ValueError(
                f"Sensitivity must be strictly positive, got {sensitivity}"
            )
        self.epsilon = float(epsilon)
        self.sensitivity = float(sensitivity)
        self.scale = self.sensitivity / self.epsilon
        self.rng = np.random.default_rng(random_state)

    def perturb(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Add Laplace noise to exact query result."""
        val = np.asarray(value, dtype=np.float64)
        noise = self.rng.laplace(0.0, self.scale, size=val.shape)
        perturbed = val + noise
        if isinstance(value, (int, float)):
            return float(perturbed.item())
        return perturbed

    def get_privacy_budget(self) -> Tuple[float, float]:
        """Return (epsilon, delta) pair."""
        return (self.epsilon, 0.0)


class GaussianMechanism:
    """Gaussian Mechanism for (epsilon, delta)-differential privacy.

    Adds calibrated Gaussian noise N(0, sigma^2) with sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        if epsilon <= 0.0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if not (0.0 < delta < 1.0):
            raise ValueError(f"Delta must be in (0, 1), got {delta}")
        if sensitivity <= 0.0:
            raise ValueError(f"Sensitivity must be positive, got {sensitivity}")
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.sensitivity = float(sensitivity)
        self.sigma = (
            self.sensitivity * math.sqrt(2.0 * math.log(1.25 / self.delta))
        ) / self.epsilon
        self.rng = np.random.default_rng(random_state)

    def perturb(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Add Gaussian noise to exact query value."""
        val = np.asarray(value, dtype=np.float64)
        noise = self.rng.normal(0.0, self.sigma, size=val.shape)
        perturbed = val + noise
        if isinstance(value, (int, float)):
            return float(perturbed.item())
        return perturbed

    def get_privacy_budget(self) -> Tuple[float, float]:
        """Return (epsilon, delta) pair."""
        return (self.epsilon, self.delta)


class ExponentialMechanism:
    """Exponential Mechanism for epsilon-differential privacy selection (McSherry & Talwar, 2007)."""

    def __init__(
        self,
        epsilon: float = 1.0,
        sensitivity: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        if epsilon <= 0.0:
            raise ValueError(f"Epsilon must be positive, got {epsilon}")
        if sensitivity <= 0.0:
            raise ValueError(f"Sensitivity must be positive, got {sensitivity}")
        self.epsilon = float(epsilon)
        self.sensitivity = float(sensitivity)
        self.rng = np.random.default_rng(random_state)

    def select(
        self, candidates: List[Any], utilities: Union[List[float], np.ndarray]
    ) -> Any:
        """Select an item from candidates with probability proportional to exp(eps * u / (2 * Delta))."""
        u = np.asarray(utilities, dtype=np.float64)
        if len(candidates) != len(u):
            raise ValueError("Number of candidates must match length of utilities")
        if len(candidates) == 0:
            raise ValueError("Candidates list cannot be empty")

        scaled = (self.epsilon * u) / (2.0 * self.sensitivity)
        max_scaled: float = float(np.max(scaled))
        weights = np.exp(scaled - max_scaled)
        sum_weights: float = float(np.sum(weights))
        probs = weights / (sum_weights if sum_weights > 0 else 1.0)

        idx = int(self.rng.choice(len(candidates), p=probs))
        return candidates[idx]

    def get_privacy_budget(self) -> Tuple[float, float]:
        """Return (epsilon, delta) pair."""
        return (self.epsilon, 0.0)


class DP_SGD:
    """Differentially Private Stochastic Gradient Descent (DP-SGD) optimizer.

    Implements Abadi et al. (2016) per-sample gradient clipping and calibrated Gaussian noise injection.
    """

    def __init__(
        self,
        lr: float = 0.01,
        l2_norm_clip: float = 1.0,
        noise_multiplier: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        self.lr = float(lr)
        self.l2_norm_clip = float(l2_norm_clip)
        self.noise_multiplier = float(noise_multiplier)
        self.rng = np.random.default_rng(random_state)
        self.steps = 0

    def clip_and_accumulate_gradients(
        self, per_sample_grads: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Clip each sample's gradient by L2 norm bound C and sum them with Gaussian noise."""
        batch_size = len(per_sample_grads)
        if batch_size == 0:
            raise ValueError("Per-sample gradients list cannot be empty.")

        param_keys = per_sample_grads[0].keys()
        clipped_grads: Dict[str, np.ndarray] = {
            k: np.zeros_like(per_sample_grads[0][k]) for k in param_keys
        }

        # 1. Per-sample gradient clipping
        for sample_grad in per_sample_grads:
            total_norm_sq = sum(np.sum(sample_grad[k] ** 2) for k in param_keys)
            sample_norm = math.sqrt(float(total_norm_sq))
            clip_coef = min(1.0, self.l2_norm_clip / (sample_norm + 1e-12))

            for k in param_keys:
                clipped_grads[k] += sample_grad[k] * clip_coef

        # 2. Add calibrated Gaussian noise
        sigma = self.noise_multiplier * self.l2_norm_clip
        noisy_grads: Dict[str, np.ndarray] = {}
        for k in param_keys:
            noise = self.rng.normal(0.0, sigma, size=clipped_grads[k].shape)
            noisy_grads[k] = (clipped_grads[k] + noise) / float(batch_size)

        self.steps += 1
        return noisy_grads

    def step(
        self,
        params: Dict[str, np.ndarray],
        per_sample_grads: List[Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """Perform one parameter update step with differentially private gradients."""
        noisy_grads = self.clip_and_accumulate_gradients(per_sample_grads)
        for k in params:
            if k in noisy_grads:
                params[k] = params[k] - self.lr * noisy_grads[k]
        return params

    def compute_privacy_spent(
        self, total_samples: int, batch_size: int, target_delta: float = 1e-5
    ) -> Dict[str, float]:
        """Estimate privacy budget (epsilon, delta) via Moments Accountant approximation."""
        q = float(batch_size) / float(total_samples)
        t = self.steps
        sigma = self.noise_multiplier
        if sigma <= 0.0 or t == 0:
            return {"epsilon": float("inf"), "delta": target_delta}

        eps = (q * math.sqrt(t * 2.0 * math.log(1.0 / target_delta))) / (sigma + 1e-12)
        return {"epsilon": float(eps), "delta": float(target_delta), "steps": float(t)}


class RenyiDPAccountant:
    """Rényi Differential Privacy (RDP) Composition Accountant (Mironov, 2017)."""

    def __init__(self, orders: Optional[List[float]] = None) -> None:
        if orders is None:
            orders = [1.5, 2.0, 3.0, 4.0, 5.0, 8.0, 16.0, 32.0, 64.0, 128.0]
        self.orders = [float(o) for o in orders if o > 1.0]
        self.rdp_total: np.ndarray = np.zeros(len(self.orders), dtype=np.float64)
        self.steps = 0

    def step_gaussian(
        self, noise_multiplier: float, subsampling_ratio: float = 1.0
    ) -> None:
        """Compose one step of Gaussian mechanism under subsampling."""
        if noise_multiplier <= 0.0:
            raise ValueError("noise_multiplier must be positive")
        sigma = float(noise_multiplier)
        q = float(subsampling_ratio)
        for idx, alpha in enumerate(self.orders):
            rdp_step = (q**2 * alpha) / (2.0 * sigma**2)
            self.rdp_total[idx] += rdp_step
        self.steps += 1

    def get_epsilon(self, target_delta: float = 1e-5) -> float:
        """Convert composed RDP to standard (epsilon, delta)-DP via optimal order minimization."""
        if target_delta <= 0.0 or target_delta >= 1.0:
            raise ValueError(f"target_delta must be in (0, 1), got {target_delta}")
        if self.steps == 0:
            return 0.0
        eps_candidates = []
        for idx, alpha in enumerate(self.orders):
            eps_candidates.append(
                self.rdp_total[idx] + np.log(1.0 / target_delta) / (alpha - 1.0)
            )
        return float(np.min(eps_candidates))
