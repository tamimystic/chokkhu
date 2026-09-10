"""Sovereign BOHB (Bayesian Optimization and HyperBand) in Pure NumPy."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


class BOHB:
    """Bayesian Optimization and HyperBand (BOHB) Multi-Fidelity Hyperparameter Tuner."""

    def __init__(
        self,
        eval_func: Callable[[Dict[str, float], float], float],
        param_bounds: Dict[str, Tuple[float, float]],
        min_budget: float = 1.0,
        max_budget: float = 27.0,
        eta: int = 3,
        random_fraction: float = 0.33,
        random_state: int = 42,
    ) -> None:
        self.eval_func = eval_func
        self.param_bounds = param_bounds
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        self.random_fraction = random_fraction
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Compute Hyperband bracket structure
        self.s_max = int(np.floor(np.log(max_budget / min_budget) / np.log(eta)))
        self.B = (self.s_max + 1) * max_budget

        self.history: List[Dict[str, Any]] = []
        self.best_config: Optional[Dict[str, float]] = None
        self.best_score = -np.inf

    def _sample_random_config(self) -> Dict[str, float]:
        """Sample a uniform random hyperparameter configuration from bounds."""
        config = {}
        for name, (low, high) in self.param_bounds.items():
            config[name] = float(self.rng.uniform(low, high))
        return config

    def _sample_bo_config(self) -> Dict[str, float]:
        """Surrogate-guided candidate sampling using best historical configurations."""
        if len(self.history) < 5 or self.rng.rand() < self.random_fraction:
            return self._sample_random_config()

        # Extract top 25% configs as positive guidance
        sorted_history = sorted(self.history, key=lambda x: x["score"], reverse=True)
        top_k = max(2, len(sorted_history) // 4)
        good_configs = [h["config"] for h in sorted_history[:top_k]]

        # Perturb a randomly chosen good configuration with Gaussian noise
        base_config = self.rng.choice(good_configs)
        new_config = {}
        for name, (low, high) in self.param_bounds.items():
            val = base_config[name]
            std = (high - low) * 0.1
            perturbed = float(self.rng.normal(val, std))
            new_config[name] = float(np.clip(perturbed, low, high))

        return new_config

    def optimize(self) -> Tuple[Dict[str, float], float]:
        """Execute full multi-bracket BOHB hyperparameter search."""
        for s in reversed(range(self.s_max + 1)):
            # Initial number of configurations in bracket
            n = int(np.ceil((self.B / self.max_budget) * (self.eta**s) / (s + 1)))
            # Initial budget per configuration
            r = self.min_budget * (self.eta ** (self.s_max - s))

            # Sample n configurations
            T = [self._sample_bo_config() for _ in range(n)]

            for i in range(s + 1):
                n_i = int(np.floor(n * (self.eta ** (-i))))
                r_i = float(r * (self.eta**i))

                # Evaluate all configurations in current tier
                val_losses = []
                for config in T:
                    score = float(self.eval_func(config, r_i))
                    val_losses.append(score)

                    entry = {
                        "bracket": s,
                        "tier": i,
                        "budget": r_i,
                        "config": config,
                        "score": score,
                    }
                    self.history.append(entry)

                    if score > self.best_score:
                        self.best_score = score
                        self.best_config = dict(config)

                # Sort and keep top 1 / eta configurations for next tier
                indices = np.argsort(val_losses)[::-1]
                num_keep = max(1, int(np.floor(n_i / self.eta)))
                T = [T[idx] for idx in indices[:num_keep]]

        if self.best_config is None:
            self.best_config = self._sample_random_config()
            self.best_score = float(self.eval_func(self.best_config, self.max_budget))

        return self.best_config, self.best_score
