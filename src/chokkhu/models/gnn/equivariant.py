from __future__ import annotations

import numpy as np


class EGNNLayer:
    """
    E(n) Equivariant Graph Neural Network (EGNN) Layer.
    Guarantees strict translation, rotation, and reflection equivariance for 3D coordinates
    and invariance for node scalar features.

    Parameters
    ----------
    in_dim : int
        Dimension of scalar node features h_i.
    hidden_dim : int
        Hidden dimension for MLP message & coordinate functions.
    out_dim : int
        Output scalar feature dimension.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        seed: int = 42,
    ) -> None:
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        rng = np.random.RandomState(seed)

        # Message MLP phi_m: (h_i, h_j, ||x_i - x_j||^2) -> m_ij
        msg_in_dim = 2 * in_dim + 1
        self.W_m1: np.ndarray = rng.randn(msg_in_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / msg_in_dim)
        self.b_m1: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)
        self.W_m2: np.ndarray = rng.randn(hidden_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)
        self.b_m2: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)

        # Coordinate MLP phi_x: m_ij -> scalar weight for (x_i - x_j)
        self.W_x: np.ndarray = rng.randn(hidden_dim, 1).astype(np.float32) * 0.01
        self.b_x: np.ndarray = np.zeros(1, dtype=np.float32)

        # Node feature update MLP phi_h: (h_i, m_i) -> h_i^(l+1)
        node_in_dim = in_dim + hidden_dim
        self.W_h1: np.ndarray = rng.randn(node_in_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / node_in_dim)
        self.b_h1: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)
        self.W_h2: np.ndarray = rng.randn(hidden_dim, out_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)
        self.b_h2: np.ndarray = np.zeros(out_dim, dtype=np.float32)

    def forward(
        self,
        h: np.ndarray,
        x: np.ndarray,
        edges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Forward pass.
        h: [N, in_dim] scalar node features
        x: [N, 3] continuous 3D coordinates
        edges: [2, E] edge index array

        Returns
        -------
        h_new : [N, out_dim] updated invariant features
        x_new : [N, 3] updated equivariant coordinates
        """
        N = h.shape[0]
        src, dst = edges[0], edges[1]

        # Compute squared Euclidean distances: ||x_i - x_j||^2
        coord_diff = x[src] - x[dst]  # [E, 3]
        sq_dist = np.sum(coord_diff**2, axis=-1, keepdims=True)  # [E, 1]

        # Compute messages m_ij = phi_m(h_i, h_j, ||x_i - x_j||^2)
        msg_input = np.concatenate(
            [h[src], h[dst], sq_dist], axis=1
        )  # [E, 2*in_dim + 1]
        m_hidden = np.maximum(0.0, np.dot(msg_input, self.W_m1) + self.b_m1)
        m_ij = np.maximum(
            0.0, np.dot(m_hidden, self.W_m2) + self.b_m2
        )  # [E, hidden_dim]

        # Equivariant coordinate update: x_i^(l+1) = x_i + sum_j (x_i - x_j) * phi_x(m_ij)
        trans_weights = np.dot(m_ij, self.W_x) + self.b_x  # [E, 1]
        coord_updates = coord_diff * trans_weights  # [E, 3]

        agg_coord_update = np.zeros((N, 3), dtype=np.float32)
        np.add.at(agg_coord_update, src, coord_updates)
        x_new = x + agg_coord_update / float(max(1, N - 1))

        # Invariant feature update: h_i^(l+1) = phi_h(h_i, sum_j m_ij)
        agg_msg = np.zeros((N, self.hidden_dim), dtype=np.float32)
        np.add.at(agg_msg, src, m_ij)

        h_input = np.concatenate([h, agg_msg], axis=1)
        h_hidden = np.maximum(0.0, np.dot(h_input, self.W_h1) + self.b_h1)
        h_new = np.dot(h_hidden, self.W_h2) + self.b_h2

        return h_new, x_new


class EGNN:
    """
    Multi-Layer E(n) Equivariant Graph Neural Network.

    Parameters
    ----------
    in_dim : int
        Input scalar feature dimension.
    hidden_dim : int
        Hidden dimension.
    out_dim : int
        Output scalar feature dimension.
    num_layers : int, default=3
        Number of EGNN layers.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        out_dim: int = 16,
        num_layers: int = 3,
        seed: int = 42,
    ) -> None:
        self.layers = []
        cur_dim = in_dim
        for i in range(num_layers):
            next_dim = out_dim if i == num_layers - 1 else hidden_dim
            self.layers.append(EGNNLayer(cur_dim, hidden_dim, next_dim, seed=seed + i))
            cur_dim = next_dim

    def forward(
        self,
        h: np.ndarray,
        x: np.ndarray,
        edges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        for layer in self.layers:
            h, x = layer.forward(h, x, edges)
        return h, x
