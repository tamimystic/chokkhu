"""Bayesian Optimization Engine with Gaussian Process Surrogate."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Union
import numpy as np

from .surrogate import (
    GaussianProcessSurrogate,
    expected_improvement,
    upper_confidence_bound,
)


class BayesianOptimization:
    """Sequential Model-Based Optimization (SMBO) using Gaussian Process Surrogate."""

    def __init__(
        self,
        objective_fn: Callable[[Dict[str, Any]], float],
        param_bounds: Dict[str, Tuple[Union[int, float], Union[int, float]]],
        acquisition: str = "ei",
        n_init: int = 5,
        n_iter: int = 20,
        random_state: int = 42,
    ) -> None:
        self.objective_fn = objective_fn
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.acquisition = acquisition.lower()
        self.n_init = n_init
        self.n_iter = n_iter
        self.random_state = random_state

        self.surrogate = GaussianProcessSurrogate()
        self.X_history: List[np.ndarray] = []
        self.y_history: List[float] = []
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = float("-inf")

    def _normalize_point(self, point: np.ndarray) -> np.ndarray:
        """Map raw parameter bounds to [0, 1]."""
        norm_pt = np.zeros_like(point)
        for i, name in enumerate(self.param_names):
            low, high = self.param_bounds[name]
            norm_pt[i] = (point[i] - low) / max(1e-9, high - low)
        return norm_pt

    def _denormalize_point(self, norm_pt: np.ndarray) -> Dict[str, Any]:
        """Map [0, 1] normalized point back to parameter dict."""
        res: Dict[str, Any] = {}
        for i, name in enumerate(self.param_names):
            low, high = self.param_bounds[name]
            val: float = float(low + norm_pt[i] * (high - low))
            if isinstance(low, int) and isinstance(high, int):
                res[name] = int(round(val))
            else:
                res[name] = float(val)
        return res

    def optimize(self) -> Dict[str, Any]:
        """Run Bayesian Optimization loop."""
        np.random.seed(self.random_state)
        dim = len(self.param_names)

        # 1. Initial random sampling (Quasi-random)
        init_samples = np.random.uniform(0.0, 1.0, size=(self.n_init, dim))
        for sample in init_samples:
            param_dict = self._denormalize_point(sample)
            score = self.objective_fn(param_dict)
            self.X_history.append(sample)
            self.y_history.append(score)
            if score > self.best_score:
                self.best_score = score
                self.best_params = param_dict

        # 2. Sequential Acquisition Loop
        for _ in range(self.n_iter):
            X_arr = np.array(self.X_history)
            y_arr = np.array(self.y_history)

            self.surrogate.fit(X_arr, y_arr)

            # Candidate point search via dense random sampling
            candidates = np.random.uniform(0.0, 1.0, size=(1000, dim))
            mu, sigma2 = self.surrogate.predict(candidates)

            if self.acquisition == "ei":
                acq_vals = expected_improvement(mu, sigma2, best_y=self.best_score)
            elif self.acquisition == "ucb":
                acq_vals = upper_confidence_bound(mu, sigma2)
            else:
                acq_vals = mu

            next_idx = int(np.argmax(acq_vals))
            next_point = candidates[next_idx]

            param_dict = self._denormalize_point(next_point)
            score = self.objective_fn(param_dict)

            self.X_history.append(next_point)
            self.y_history.append(score)

            if score > self.best_score:
                self.best_score = score
                self.best_params = param_dict

        return self.best_params
