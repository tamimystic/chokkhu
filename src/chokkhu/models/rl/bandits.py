from __future__ import annotations

import numpy as np


class EpsilonGreedyBandit:
    """
    Multi-Armed Bandit with decaying Epsilon-Greedy exploration.

    Parameters
    ----------
    n_arms : int
        Number of arms.
    epsilon : float, default=1.0
        Initial exploration rate.
    epsilon_min : float, default=0.01
        Minimum exploration rate.
    decay : float, default=0.99
        Multiplicative decay factor per step.
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        decay: float = 0.99,
        seed: int = 42,
    ) -> None:
        self.n_arms: int = n_arms
        self.epsilon: float = epsilon
        self.epsilon_min: float = epsilon_min
        self.decay: float = decay
        self.rng = np.random.RandomState(seed)

        self.counts: np.ndarray = np.zeros(n_arms, dtype=np.int64)
        self.values: np.ndarray = np.zeros(n_arms, dtype=np.float64)

    def select_arm(self) -> int:
        """Select an arm to pull."""
        if self.rng.rand() < self.epsilon:
            return int(self.rng.randint(self.n_arms))
        return int(np.argmax(self.values))

    def update(self, arm: int, reward: float) -> None:
        """Update sample mean estimate for selected arm."""
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        self.epsilon = max(self.epsilon_min, self.epsilon * self.decay)


class UCB1Bandit:
    """
    Upper Confidence Bound (UCB1) Multi-Armed Bandit.

    Parameters
    ----------
    n_arms : int
        Number of arms.
    c : float, default=2.0
        Exploration confidence parameter (standard UCB1 uses sqrt(2)).
    """

    def __init__(self, n_arms: int, c: float = 2.0) -> None:
        self.n_arms: int = n_arms
        self.c: float = c
        self.total_steps: int = 0
        self.counts: np.ndarray = np.zeros(n_arms, dtype=np.int64)
        self.values: np.ndarray = np.zeros(n_arms, dtype=np.float64)

    def select_arm(self) -> int:
        """Select arm maximizing upper confidence bound."""
        # Try every arm once first
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm

        ucb_values = self.values + np.sqrt(
            (self.c * np.log(self.total_steps)) / self.counts
        )
        return int(np.argmax(ucb_values))

    def update(self, arm: int, reward: float) -> None:
        """Update empirical mean for selected arm."""
        self.total_steps += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class ThompsonSamplingBandit:
    """
    Thompson Sampling (Posterior Sampling) Multi-Armed Bandit with Beta-Bernoulli conjugate prior.

    Parameters
    ----------
    n_arms : int
        Number of arms.
    prior_alpha : float, default=1.0
        Prior pseudo-successes.
    prior_beta : float, default=1.0
        Prior pseudo-failures.
    """

    def __init__(
        self,
        n_arms: int,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.n_arms: int = n_arms
        self.rng = np.random.RandomState(seed)
        self.alpha: np.ndarray = np.full(n_arms, prior_alpha, dtype=np.float64)
        self.beta: np.ndarray = np.full(n_arms, prior_beta, dtype=np.float64)

    def select_arm(self) -> int:
        """Sample from posterior Beta distribution for each arm and pick argmax."""
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        """Update Beta parameters: alpha += reward, beta += (1 - reward)."""
        r = 1.0 if reward > 0.5 else 0.0
        self.alpha[arm] += r
        self.beta[arm] += 1.0 - r


class LinUCBBandit:
    """
    Disjoint Linear Contextual Bandit (LinUCB with exploration parameter alpha).

    Parameters
    ----------
    n_arms : int
        Number of arms.
    n_features : int
        Dimension of context vector.
    alpha : float, default=1.0
        Exploration confidence coefficient.
    """

    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0) -> None:
        self.n_arms: int = n_arms
        self.n_features: int = n_features
        self.alpha: float = alpha

        # A_a = d x d identity matrix, b_a = d x 1 zero vector
        self.A: list[np.ndarray] = [
            np.eye(n_features, dtype=np.float64) for _ in range(n_arms)
        ]
        self.b: list[np.ndarray] = [
            np.zeros(n_features, dtype=np.float64) for _ in range(n_arms)
        ]

    def select_arm(self, context: np.ndarray) -> int:
        """Select arm using contextual upper confidence bound."""
        x = np.asarray(context, dtype=np.float64).ravel()
        p: np.ndarray = np.zeros(self.n_arms, dtype=np.float64)

        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta_a = np.dot(A_inv, self.b[a])
            # Confidence bound: x^T theta + alpha * sqrt(x^T A^-1 x)
            expected_payoff = float(np.dot(x, theta_a))
            std = float(np.sqrt(np.dot(x, np.dot(A_inv, x))))
            p[a] = expected_payoff + self.alpha * std

        return int(np.argmax(p))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """Update ridge regression parameters for pulled arm."""
        x = np.asarray(context, dtype=np.float64).ravel()
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
