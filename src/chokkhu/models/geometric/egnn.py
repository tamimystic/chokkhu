"""E(n)-Equivariant Graph Neural Networks (Satorras et al., 2021) in Pure NumPy.

Guarantees strict Euclidean rotation, translation, and reflection equivariance for n-dimensional coordinates
and invariant message passing for node scalar representations.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import numpy as np


class EnEquivariantLayer:
    """E(n) Equivariant Graph Convolution Layer for arbitrary spatial dimension n >= 2."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        coord_dim: int = 3,
        edge_dim: int = 0,
        random_state: int = 42,
    ) -> None:
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.coord_dim = int(coord_dim)
        self.edge_dim = int(edge_dim)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Message MLP phi_m: (h_i, h_j, ||x_i - x_j||^2, e_ij) -> m_ij
        msg_in_dim = 2 * self.in_dim + 1 + self.edge_dim
        self.W_m1: np.ndarray = self.rng.randn(msg_in_dim, self.hidden_dim).astype(
            np.float64
        ) * np.sqrt(2.0 / msg_in_dim)
        self.b_m1: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float64)
        self.W_m2: np.ndarray = self.rng.randn(self.hidden_dim, self.hidden_dim).astype(
            np.float64
        ) * np.sqrt(2.0 / self.hidden_dim)
        self.b_m2: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float64)

        # Coordinate update MLP phi_x: m_ij -> scalar weight for (x_i - x_j)
        self.W_x1: np.ndarray = self.rng.randn(self.hidden_dim, self.hidden_dim).astype(
            np.float64
        ) * np.sqrt(2.0 / self.hidden_dim)
        self.b_x1: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float64)
        self.W_x2: np.ndarray = (
            self.rng.randn(self.hidden_dim, 1).astype(np.float64) * 0.01
        )
        self.b_x2: np.ndarray = np.zeros(1, dtype=np.float64)

        # Node feature update MLP phi_h: (h_i, sum_j m_ij) -> h_i^(l+1)
        node_in_dim = self.in_dim + self.hidden_dim
        self.W_h1: np.ndarray = self.rng.randn(node_in_dim, self.hidden_dim).astype(
            np.float64
        ) * np.sqrt(2.0 / node_in_dim)
        self.b_h1: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float64)
        self.W_h2: np.ndarray = self.rng.randn(self.hidden_dim, self.out_dim).astype(
            np.float64
        ) * np.sqrt(2.0 / self.hidden_dim)
        self.b_h2: np.ndarray = np.zeros(self.out_dim, dtype=np.float64)

    def forward(
        self,
        h: np.ndarray,
        x: np.ndarray,
        edges: np.ndarray,
        edge_attr: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass.

        Args:
            h: [N, in_dim] invariant scalar features
            x: [N, coord_dim] equivariant coordinates
            edges: [2, E] edge indices (src, dst)
            edge_attr: Optional [E, edge_dim] edge attributes

        Returns:
            h_new: [N, out_dim] updated invariant features
            x_new: [N, coord_dim] updated equivariant coordinates
        """
        N = h.shape[0]
        src, dst = edges[0], edges[1]

        # Coordinate differences & invariant squared distances
        coord_diff = x[src] - x[dst]  # [E, coord_dim]
        sq_dist = np.sum(coord_diff**2, axis=-1, keepdims=True)  # [E, 1]

        # Construct message input: [h_i, h_j, ||x_i - x_j||^2, edge_attr]
        msg_parts = [h[src], h[dst], sq_dist]
        if edge_attr is not None:
            msg_parts.append(edge_attr)
        msg_in = np.concatenate(msg_parts, axis=1)

        # Message computation with SiLU
        z_m1 = msg_in @ self.W_m1 + self.b_m1
        h_m1 = z_m1 / (1.0 + np.exp(-np.clip(z_m1, -15.0, 15.0)))
        z_m2 = h_m1 @ self.W_m2 + self.b_m2
        m_ij = z_m2 / (1.0 + np.exp(-np.clip(z_m2, -15.0, 15.0)))  # [E, hidden_dim]

        # Coordinate weights via phi_x
        z_x1 = m_ij @ self.W_x1 + self.b_x1
        h_x1 = z_x1 / (1.0 + np.exp(-np.clip(z_x1, -15.0, 15.0)))
        x_weights = h_x1 @ self.W_x2 + self.b_x2  # [E, 1]

        # Coordinate updates: sum_j (x_i - x_j) * phi_x(m_ij)
        coord_updates = coord_diff * x_weights  # [E, coord_dim]
        agg_coord = np.zeros((N, self.coord_dim), dtype=np.float64)
        np.add.at(agg_coord, src, coord_updates)
        x_new = x + agg_coord / float(max(1, N - 1))

        # Invariant message aggregation: sum_j m_ij
        agg_m = np.zeros((N, self.hidden_dim), dtype=np.float64)
        np.add.at(agg_m, src, m_ij)

        # Node feature update
        node_in = np.concatenate([h, agg_m], axis=1)
        z_h1 = node_in @ self.W_h1 + self.b_h1
        h_h1 = z_h1 / (1.0 + np.exp(-np.clip(z_h1, -15.0, 15.0)))
        z_h2 = h_h1 @ self.W_h2 + self.b_h2
        h_new = z_h2 / (1.0 + np.exp(-np.clip(z_h2, -15.0, 15.0)))

        return h_new, x_new


class EnEquivariantGNN:
    """Multi-Layer E(n)-Equivariant Graph Neural Network for molecules, physics, and point clouds."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        coord_dim: int = 3,
        num_layers: int = 3,
        random_state: int = 42,
    ) -> None:
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.coord_dim = coord_dim
        self.num_layers = num_layers
        self.random_state = random_state

        self.layers: List[EnEquivariantLayer] = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        for i in range(num_layers):
            layer = EnEquivariantLayer(
                in_dim=dims[i],
                hidden_dim=hidden_dim,
                out_dim=dims[i + 1],
                coord_dim=coord_dim,
                random_state=random_state + i,
            )
            self.layers.append(layer)

    def forward(
        self,
        h: np.ndarray,
        x: np.ndarray,
        edges: np.ndarray,
        edge_attr: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through all E(n)-equivariant layers."""
        curr_h = np.asarray(h, dtype=np.float64)
        curr_x = np.asarray(x, dtype=np.float64)

        for layer in self.layers:
            curr_h, curr_x = layer.forward(curr_h, curr_x, edges, edge_attr=edge_attr)

        return curr_h, curr_x
