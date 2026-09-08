"""Hyperband Bandit-Based Hyperparameter Optimization (Li et al., 2018)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List
import numpy as np


class Hyperband:
    """Hyperband successive halving algorithm with resource allocation."""

    def __init__(
        self,
        objective_fn: Callable[[Dict[str, Any], int], float],
        param_sampler: Callable[[], Dict[str, Any]],
        max_resource: int = 81,
        eta: int = 3,
        random_state: int = 42,
    ) -> None:
        self.objective_fn = objective_fn
        self.param_sampler = param_sampler
        self.max_resource = max_resource
        self.eta = eta
        self.random_state = random_state

        self.s_max = int(np.floor(np.log(max_resource) / np.log(eta)))
        self.B = (self.s_max + 1) * max_resource

        self.best_config: Dict[str, Any] = {}
        self.best_score: float = float("-inf")
        self.history: List[Dict[str, Any]] = []

    def optimize(self) -> Dict[str, Any]:
        """Execute Hyperband successive halving bracket schedule."""
        np.random.seed(self.random_state)

        for s in reversed(range(self.s_max + 1)):
            # Initial number of configurations
            n = int(np.ceil((self.B / self.max_resource / (s + 1)) * (self.eta**s)))
            # Initial resource per configuration
            r = self.max_resource * (self.eta ** (-s))

            # Sample n configurations
            T = [self.param_sampler() for _ in range(n)]

            for i in range(s + 1):
                n_i = int(np.floor(n * (self.eta ** (-i))))
                r_i = int(np.floor(r * (self.eta**i)))

                # Evaluate all configurations in T with resource r_i
                scores = []
                for config in T:
                    score = self.objective_fn(config, r_i)
                    scores.append(score)
                    self.history.append(
                        {"config": config, "resource": r_i, "score": score}
                    )

                    if score > self.best_score:
                        self.best_score = score
                        self.best_config = config

                # Successive halving: retain top 1/eta configurations
                if n_i > 1 and i < s:
                    top_indices = np.argsort(scores)[::-1][
                        : max(1, int(np.floor(n_i / self.eta)))
                    ]
                    T = [T[idx] for idx in top_indices]

        return self.best_config
