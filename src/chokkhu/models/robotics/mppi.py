"""Model Predictive Path Integral (MPPI) Trajectory Optimizer in pure NumPy."""

from __future__ import annotations

from typing import Callable, Tuple
import numpy as np


class MPPITrajectoryOptimizer:
    r"""Model Predictive Path Integral (MPPI) Control Optimizer in pure NumPy (Williams et al. 2017).

    Non-linear stochastic trajectory optimization using importance sampling over perturbed
    candidate action rollouts for real-time robotic model-predictive control.

    Parameters
    ----------
    action_dim : int
        Control action dimensionality.
    horizon : int, default=15
        Planning lookahead horizon steps ($T$).
    num_samples : int, default=64
        Number of parallel noisy control sequence rollouts ($K$).
    temperature : float, default=1.0
        Information-theoretic temperature parameter $\lambda$ weighting trajectory returns.
    noise_sigma : float, default=0.5
        Exploration noise standard deviation $\Sigma = \sigma^2 I$.
    seed : int, default=42
    """

    def __init__(
        self,
        action_dim: int,
        horizon: int = 15,
        num_samples: int = 64,
        temperature: float = 1.0,
        noise_sigma: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.num_samples = int(num_samples)
        self.temperature = float(temperature)
        self.noise_sigma = float(noise_sigma)

        self.rng = np.random.default_rng(seed)
        # Nominal control sequence: (horizon, action_dim)
        self.nominal_actions: np.ndarray = np.zeros(
            (horizon, action_dim), dtype=np.float32
        )

    def optimize(
        self,
        initial_state: np.ndarray,
        rollout_cost_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optimizes control trajectory using MPPI importance sampling.

        Parameters
        ----------
        initial_state : np.ndarray
            Current system state vector $s_0$.
        rollout_cost_fn : Callable[[np.ndarray, np.ndarray], np.ndarray]
            Cost evaluation function mapping (initial_state, perturbed_actions_batch) -> costs (K,).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (best_next_action (action_dim,), full_planned_trajectory (horizon, action_dim))
        """
        K = self.num_samples
        T = self.horizon
        D = self.action_dim

        # 1. Sample control perturbations: (K, T, D)
        noise = self.rng.normal(0, self.noise_sigma, size=(K, T, D)).astype(np.float32)
        perturbed_actions = self.nominal_actions[np.newaxis, :, :] + noise

        # 2. Evaluate trajectory costs: (K,)
        costs = np.asarray(
            rollout_cost_fn(initial_state, perturbed_actions), dtype=np.float32
        )

        # 3. Softmax importance weights: w_k = exp(-1/lambda * (S_k - min S))
        min_cost: float = float(np.min(costs))
        scaled_costs = -(costs - min_cost) / max(1e-6, self.temperature)
        exp_w = np.exp(np.clip(scaled_costs, -20.0, 0.0))
        weights = exp_w / (np.sum(exp_w) + 1e-12)  # (K,)

        # 4. Weighted nominal control sequence update
        # (K, 1, 1) * (K, T, D) -> sum over K -> (T, D)
        weighted_perturbations = np.sum(
            weights[:, np.newaxis, np.newaxis] * noise, axis=0
        )
        self.nominal_actions += weighted_perturbations

        best_action = self.nominal_actions[0].copy()

        # 5. Slide warm-start window forward by 1 step
        self.nominal_actions = np.roll(self.nominal_actions, -1, axis=0)
        self.nominal_actions[-1] = np.zeros(D, dtype=np.float32)

        return best_action, self.nominal_actions
