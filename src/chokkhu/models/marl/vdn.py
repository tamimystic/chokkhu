r"""Value-Decomposition Networks (VDN) for Multi-Agent RL in pure NumPy."""

from __future__ import annotations


import numpy as np


class VDN:
    r"""Value-Decomposition Networks (VDN) in pure NumPy (Sunehag et al. 2017).

    Decomposes joint team action-value function into additive sum of individual utilities:
    $Q_{tot}(s, \mathbf{a}) = \sum_{i=1}^N Q_i(o_i, a_i)$.

    Parameters
    ----------
    n_agents : int, default=4
        Number of cooperative agents.
    """

    def __init__(self, n_agents: int = 4) -> None:
        self.n_agents = int(n_agents)

    def forward(self, agent_qs: np.ndarray) -> np.ndarray:
        r"""Computes additive joint value $Q_{tot} = \sum_i Q_i$.

        Parameters
        ----------
        agent_qs : np.ndarray
            Individual utilities matrix of shape (B, n_agents) or (n_agents,).

        Returns
        -------
        np.ndarray
            Joint Q_tot values of shape (B, 1) or (1, 1).
        """
        qs = np.asarray(agent_qs, dtype=np.float32)
        if qs.ndim == 1:
            qs = qs[np.newaxis, :]
        q_tot = np.sum(qs, axis=-1, keepdims=True)
        return q_tot

    def td_loss(
        self,
        agent_qs: np.ndarray,
        rewards: np.ndarray,
        next_agent_qs: np.ndarray,
        dones: np.ndarray,
        gamma: float = 0.99,
    ) -> float:
        """Computes Mean Squared TD error loss for VDN decomposition."""
        curr_tot = self.forward(agent_qs).flatten()
        next_tot = self.forward(next_agent_qs).flatten()

        rew = np.asarray(rewards, dtype=np.float32).flatten()
        don = np.asarray(dones, dtype=np.float32).flatten()

        targets = rew + gamma * (1.0 - don) * next_tot
        return float(np.mean((curr_tot - targets) ** 2))
