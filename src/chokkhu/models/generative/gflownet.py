"""Generative Flow Networks (GFlowNets) in Pure NumPy.

References:
- Bengio et al. (2021): "Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation" (NeurIPS 2021).
- Malkin et al. (2022): "Trajectory Balance: Improved Credit Assignment in GFlowNets" (NeurIPS 2022).
"""

from __future__ import annotations

from typing import Callable, List, Tuple
import numpy as np


class TrajectoryBalanceGFlowNet:
    """Trajectory Balance Generative Flow Network (TB-GFlowNet) for Discrete Object Generation.

    Learns forward policy P_F(s_{t+1} | s_t) and log partition Z such that terminal states x
    are sampled with probability proportional to reward function R(x):
        Loss_TB = ( log Z + sum_t log P_F(s_{t+1}|s_t) - log R(x) - sum_t log P_B(s_t|s_{t+1}) )^2
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 32,
        horizon: int = 5,
        learning_rate: float = 0.01,
        random_state: int = 42,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.learning_rate = float(learning_rate)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Forward policy MLP: s -> logits over actions
        scale1 = np.sqrt(2.0 / self.state_dim)
        self.W1: np.ndarray = self.rng.randn(self.state_dim, self.hidden_dim) * scale1
        self.b1: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float64)

        scale2 = np.sqrt(2.0 / self.hidden_dim)
        self.W2: np.ndarray = self.rng.randn(self.hidden_dim, self.action_dim) * scale2
        self.b2: np.ndarray = np.zeros(self.action_dim, dtype=np.float64)

        # Learnable log partition Z
        self.log_Z: float = 0.0

    def _forward_policy(self, s: np.ndarray) -> np.ndarray:
        """Compute action probability distribution P_F(a | s)."""
        s_arr = np.asarray(s, dtype=np.float64)
        h = np.maximum(0.0, np.dot(s_arr, self.W1) + self.b1)
        logits = np.dot(h, self.W2) + self.b2
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

    def sample_trajectories(
        self, batch_size: int = 16
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Sample on-policy generation trajectories s_0 -> ... -> s_H."""
        bs = int(batch_size)
        curr_state: np.ndarray = np.zeros((bs, self.state_dim), dtype=np.float64)

        states_history: List[np.ndarray] = [curr_state]
        actions_history: List[np.ndarray] = []
        log_pf_history: List[np.ndarray] = []

        for _ in range(self.horizon):
            probs = self._forward_policy(curr_state)  # [bs, action_dim]
            actions: np.ndarray = np.zeros(bs, dtype=np.int64)
            log_pfs: np.ndarray = np.zeros(bs, dtype=np.float64)

            for i in range(bs):
                a = self.rng.choice(self.action_dim, p=probs[i])
                actions[i] = a
                log_pfs[i] = np.log(max(probs[i, a], 1e-12))

            # State transition: flip binary coordinate
            next_state = np.copy(curr_state)
            for i in range(bs):
                dim_idx: int = int(actions[i] % self.state_dim)
                next_state[i, dim_idx] = 1.0

            actions_history.append(actions)
            log_pf_history.append(log_pfs)
            curr_state = next_state
            states_history.append(curr_state)

        return curr_state, states_history, actions_history, log_pf_history

    def fit(
        self,
        reward_fn: Callable[[np.ndarray], np.ndarray],
        num_iterations: int = 100,
        batch_size: int = 16,
        verbose: bool = False,
    ) -> List[float]:
        """Train GFlowNet using Trajectory Balance objective."""
        loss_history: List[float] = []

        for iteration in range(num_iterations):
            terminal_states, states_hist, actions_hist, log_pf_hist = (
                self.sample_trajectories(batch_size)
            )

            # Compute rewards R(x) > 0
            rewards = np.maximum(reward_fn(terminal_states), 1e-6)
            log_rewards = np.log(rewards)

            # Sum log P_F along trajectories
            sum_log_pf = np.sum(np.column_stack(log_pf_hist), axis=1)

            # Uniform backward policy log P_B
            log_pb_step = -np.log(float(self.horizon))
            sum_log_pb = log_pb_step * self.horizon

            # Trajectory Balance discrepancy: delta = log_Z + sum(log_pf) - log_R - sum(log_pb)
            tb_delta = self.log_Z + sum_log_pf - log_rewards - sum_log_pb
            loss = float(np.mean(tb_delta**2))
            loss_history.append(loss)

            # Gradient updates
            d_tb = 2.0 * tb_delta / float(batch_size)
            d_log_Z = float(np.sum(d_tb))
            self.log_Z -= self.learning_rate * d_log_Z

            # Policy gradients via REINFORCE / TB surrogate
            for step in range(self.horizon):
                s_t = states_hist[step]
                a_t = actions_hist[step]

                probs = self._forward_policy(s_t)
                d_logits = np.zeros_like(probs)
                for i in range(batch_size):
                    d_logits[i] = d_tb[i] * probs[i]
                    d_logits[i, a_t[i]] -= d_tb[i]

                h = np.maximum(0.0, np.dot(s_t, self.W1) + self.b1)
                d_W2 = np.dot(h.T, d_logits)
                d_b2 = np.sum(d_logits, axis=0)

                d_h = np.dot(d_logits, self.W2.T) * (h > 0.0)
                d_W1 = np.dot(s_t.T, d_h)
                d_b1 = np.sum(d_h, axis=0)

                self.W2 -= self.learning_rate * np.clip(d_W2, -5.0, 5.0)
                self.b2 -= self.learning_rate * np.clip(d_b2, -5.0, 5.0)
                self.W1 -= self.learning_rate * np.clip(d_W1, -5.0, 5.0)
                self.b1 -= self.learning_rate * np.clip(d_b1, -5.0, 5.0)

            if verbose and (iteration + 1) % 25 == 0:
                print(
                    f"GFlowNet Iter {iteration+1}/{num_iterations} - "
                    f"TB Loss: {loss:.4f}, log_Z: {self.log_Z:.2f}"
                )

        return loss_history

    def sample(self, num_samples: int = 50) -> np.ndarray:
        """Sample generated terminal discrete candidates."""
        terminal_states, _, _, _ = self.sample_trajectories(num_samples)
        return terminal_states
