"""Cellular Sheaf Neural Networks & Sheaf Laplacian Diffusion in Pure NumPy & SciPy.

References:
- Bodnar et al. (2022): "Neural Sheaf Diffusion: A Topological Perspective on Heterophily and Oversmoothing" (NeurIPS 2022).
- Hansen & Ghrist (2019): "Toward a Spectral Theory of Cellular Sheaves".
"""

from __future__ import annotations

from typing import List
import numpy as np


class CellularSheaf:
    """Cellular Sheaf over a Graph with vector stalks and learnable/computable restriction maps.

    A cellular sheaf assigns:
    - Node stalk F(v) = R^{d_v} to each node v in V
    - Edge stalk F(e) = R^{d_e} to each directed edge e = (u -> v) in E
    - Linear restriction maps F_{u <| e} in R^{d_e x d_u} and F_{v <| e} in R^{d_e x d_v}
    """

    def __init__(
        self,
        num_nodes: int,
        edges: np.ndarray,
        node_dim: int = 2,
        edge_dim: int = 2,
        random_state: int = 42,
    ) -> None:
        self.num_nodes = int(num_nodes)
        self.edges = np.asarray(edges, dtype=np.int64)
        if self.edges.ndim != 2 or self.edges.shape[0] != 2:
            raise ValueError("edges must have shape (2, num_edges)")

        self.num_edges = self.edges.shape[1]
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Initialize orthogonal / random restriction maps:
        # F_src: (num_edges, edge_dim, node_dim)
        # F_dst: (num_edges, edge_dim, node_dim)
        self.F_src = np.zeros(
            (self.num_edges, self.edge_dim, self.node_dim), dtype=np.float64
        )
        self.F_dst = np.zeros(
            (self.num_edges, self.edge_dim, self.node_dim), dtype=np.float64
        )

        for e in range(self.num_edges):
            # Generate random orthogonal or standard normal maps
            q1, _ = np.linalg.qr(
                self.rng.randn(
                    max(self.edge_dim, self.node_dim), max(self.edge_dim, self.node_dim)
                )
            )
            q2, _ = np.linalg.qr(
                self.rng.randn(
                    max(self.edge_dim, self.node_dim), max(self.edge_dim, self.node_dim)
                )
            )
            self.F_src[e] = q1[: self.edge_dim, : self.node_dim]
            self.F_dst[e] = q2[: self.edge_dim, : self.node_dim]

    def set_restriction_maps(self, F_src: np.ndarray, F_dst: np.ndarray) -> None:
        """Set explicit restriction maps for all edges."""
        self.F_src = np.asarray(F_src, dtype=np.float64)
        self.F_dst = np.asarray(F_dst, dtype=np.float64)

    def compute_coboundary(self) -> np.ndarray:
        """Construct the sheaf coboundary matrix delta of shape (num_edges * edge_dim, num_nodes * node_dim).

        For edge e = (u -> v):
        (delta x)_e = F_{v <| e} x_v - F_{u <| e} x_u
        """
        N, E = self.num_nodes, self.num_edges
        d_v, d_e = self.node_dim, self.edge_dim

        delta = np.zeros((E * d_e, N * d_v), dtype=np.float64)

        for e_idx in range(E):
            u = self.edges[0, e_idx]
            v = self.edges[1, e_idx]

            row_start = e_idx * d_e
            row_end = row_start + d_e

            # Source restriction -F_{u <| e}
            col_u_start = u * d_v
            col_u_end = col_u_start + d_v
            delta[row_start:row_end, col_u_start:col_u_end] -= self.F_src[e_idx]

            # Target restriction +F_{v <| e}
            col_v_start = v * d_v
            col_v_end = col_v_start + d_v
            delta[row_start:row_end, col_v_start:col_v_end] += self.F_dst[e_idx]

        return delta

    def compute_laplacian(self) -> np.ndarray:
        """Compute the Sheaf Laplacian matrix Delta_F = delta^T delta of shape (N * d_v, N * d_v)."""
        delta = self.compute_coboundary()
        laplacian = delta.T @ delta
        # Symmetrize for numerical exactness
        return 0.5 * (laplacian + laplacian.T)

    def dirichlet_energy(self, x: np.ndarray) -> float:
        """Compute the Sheaf Dirichlet energy E(x) = 1/2 x^T Delta_F x = 1/2 ||delta x||_2^2."""
        x_flat = np.asarray(x, dtype=np.float64).flatten()
        delta = self.compute_coboundary()
        cochain = delta @ x_flat
        return float(0.5 * np.sum(cochain**2))

    def compute_normalized_laplacian(self, eps: float = 1e-8) -> np.ndarray:
        """Compute the symmetrically normalized Sheaf Laplacian L_norm = D^{-1/2} Delta_F D^{-1/2}."""
        L = self.compute_laplacian()
        d_v = self.node_dim
        N = self.num_nodes

        # Block diagonal degree inverse square root
        D_inv_sqrt = np.zeros_like(L)
        for i in range(N):
            block = L[i * d_v : (i + 1) * d_v, i * d_v : (i + 1) * d_v]
            # Eigen-decomposition of block
            vals, vecs = np.linalg.eigh(block + eps * np.eye(d_v))
            inv_sqrt_vals = 1.0 / np.sqrt(np.maximum(vals, eps))
            block_inv_sqrt = vecs @ np.diag(inv_sqrt_vals) @ vecs.T
            D_inv_sqrt[i * d_v : (i + 1) * d_v, i * d_v : (i + 1) * d_v] = (
                block_inv_sqrt
            )

        return D_inv_sqrt @ L @ D_inv_sqrt


class SheafDiffusionLayer:
    """Neural Sheaf Diffusion Layer implementing discrete heat diffusion: X^(t+1) = sigma((I - alpha Delta_F) X W)."""

    def __init__(
        self,
        node_dim: int,
        in_channels: int,
        out_channels: int,
        alpha: float = 0.5,
        random_state: int = 42,
    ) -> None:
        self.node_dim = int(node_dim)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.alpha = float(alpha)
        self.rng = np.random.RandomState(random_state)

        # Weight matrix: (in_channels, out_channels)
        limit = np.sqrt(6.0 / (in_channels + out_channels))
        self.W: np.ndarray = self.rng.uniform(
            -limit, limit, size=(in_channels, out_channels)
        ).astype(np.float64)
        self.b: np.ndarray = np.zeros(out_channels, dtype=np.float64)

    def forward(
        self,
        x: np.ndarray,
        sheaf: CellularSheaf,
        activation: str = "silu",
    ) -> np.ndarray:
        """Forward diffusion step.

        Args:
            x: Node features of shape (num_nodes, node_dim, in_channels) or (num_nodes, in_channels)
            sheaf: CellularSheaf instance defining the geometry
            activation: Activation function name ('silu', 'relu', 'tanh', 'linear')

        Returns:
            Updated features of shape (num_nodes, node_dim, out_channels) or (num_nodes, out_channels)
        """
        N = sheaf.num_nodes
        d_v = sheaf.node_dim
        x_arr = np.asarray(x, dtype=np.float64)

        if x_arr.ndim == 2:
            # (N, in_channels) -> broadcast across stalk
            x_arr = x_arr[:, np.newaxis, :]
            if x_arr.shape[1] != d_v:
                x_arr = np.repeat(x_arr, d_v, axis=1)

        # Shape: (N * d_v, in_channels)
        in_c = x_arr.shape[-1]
        x_flat = x_arr.reshape(N * d_v, in_c)

        # Sheaf diffusion: X_diff = (I - alpha * Delta_F) X
        L_norm = sheaf.compute_normalized_laplacian()
        I_mat: np.ndarray = np.eye(N * d_v, dtype=np.float64)
        diffusion_operator = I_mat - self.alpha * L_norm

        x_diff = diffusion_operator @ x_flat  # [N * d_v, in_c]

        # Feature transformation: X_diff @ W + b
        out_flat = x_diff @ self.W + self.b  # [N * d_v, out_channels]

        # Non-linear activation
        if activation == "relu":
            out_flat = np.maximum(0.0, out_flat)
        elif activation == "silu":
            out_flat = out_flat / (1.0 + np.exp(-np.clip(out_flat, -15.0, 15.0)))
        elif activation == "tanh":
            out_flat = np.tanh(out_flat)

        return out_flat.reshape(N, d_v, self.out_channels)


class SheafNeuralNetwork:
    """Multi-Layer Neural Sheaf Network (NSN) for robust node classification under heterophily."""

    def __init__(
        self,
        node_dim: int = 2,
        edge_dim: int = 2,
        hidden_dim: int = 32,
        out_dim: int = 2,
        num_layers: int = 2,
        alpha: float = 0.5,
        random_state: int = 42,
    ) -> None:
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.alpha = alpha
        self.random_state = random_state

        self.layers: List[SheafDiffusionLayer] = []
        dims = [1] + [hidden_dim] * (num_layers - 1) + [out_dim]

        for i in range(num_layers):
            layer = SheafDiffusionLayer(
                node_dim=node_dim,
                in_channels=dims[i],
                out_channels=dims[i + 1],
                alpha=alpha,
                random_state=random_state + i,
            )
            self.layers.append(layer)

    def forward(
        self,
        x: np.ndarray,
        sheaf: CellularSheaf,
    ) -> np.ndarray:
        """Forward pass through all sheaf diffusion layers."""
        h = np.asarray(x, dtype=np.float64)
        for i, layer in enumerate(self.layers):
            act = "silu" if i < len(self.layers) - 1 else "linear"
            h = layer.forward(h, sheaf, activation=act)

        # Average over stalk dimension to obtain node predictions
        return np.mean(h, axis=1)  # (N, out_dim)
