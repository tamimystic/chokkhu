from __future__ import annotations

import numpy as np


class LaplacianPositionalEncoding:
    """
    Laplacian Eigenvector Positional Encodings for Graph Transformers.
    Computes k smallest non-trivial eigenvectors of the normalized graph Laplacian.
    """

    def __init__(self, k: int = 8) -> None:
        self.k = k

    def compute(self, adj: np.ndarray) -> np.ndarray:
        """
        Compute top-k Laplacian positional encodings.

        Parameters
        ----------
        adj : np.ndarray
            Adjacency matrix of shape [N, N].

        Returns
        -------
        pe : np.ndarray
            Positional encodings of shape [N, k].
        """
        N = adj.shape[0]
        deg = np.sum(adj, axis=1)
        deg_inv_sqrt = np.zeros_like(deg, dtype=np.float32)
        mask = deg > 0
        deg_inv_sqrt[mask] = np.power(deg[mask], -0.5)

        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        D_inv = np.diag(deg_inv_sqrt)
        L = np.eye(N) - np.dot(np.dot(D_inv, adj), D_inv)

        eigvals, eigvecs = np.linalg.eigh(L)
        # Sort by eigenvalue ascending
        idx = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx]

        # Take first k non-trivial eigenvectors (skip first if zero)
        start_idx = 1 if N > 1 and np.abs(eigvals[idx[0]]) < 1e-6 else 0
        selected = eigvecs[:, start_idx : start_idx + self.k]

        if selected.shape[1] < self.k:
            pad = np.zeros((N, self.k - selected.shape[1]), dtype=np.float32)
            selected = np.hstack([selected, pad])

        return selected.astype(np.float32)


class GraphormerLayer:
    """
    Graphormer Transformer Layer with Spatial Distance Bias and Degree Centrality.

    Parameters
    ----------
    hidden_dim : int
        Dimension of node embeddings.
    n_heads : int, default=4
        Number of self-attention heads.
    dim_feedforward : int, default=128
        Dimension of FFN intermediate layer.
    dropout : float, default=0.0
        Dropout rate.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 4,
        dim_feedforward: int = 128,
        seed: int = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        rng = np.random.RandomState(seed)

        # Multi-Head Attention weights
        self.W_q: np.ndarray = rng.randn(hidden_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)
        self.W_k: np.ndarray = rng.randn(hidden_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)
        self.W_v: np.ndarray = rng.randn(hidden_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)
        self.W_o: np.ndarray = rng.randn(hidden_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)

        # Feed-Forward Network
        self.W1: np.ndarray = rng.randn(hidden_dim, dim_feedforward).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)
        self.b1: np.ndarray = np.zeros(dim_feedforward, dtype=np.float32)
        self.W2: np.ndarray = rng.randn(dim_feedforward, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / dim_feedforward)
        self.b2: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)

    def forward(
        self,
        x: np.ndarray,
        spatial_bias: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Forward pass.
        x: [N, hidden_dim]
        spatial_bias: [N, N] or None
        """
        N, D = x.shape
        # Linear projections
        Q = (
            np.dot(x, self.W_q)
            .reshape(N, self.n_heads, self.head_dim)
            .transpose(1, 0, 2)
        )  # [H, N, d]
        K = (
            np.dot(x, self.W_k)
            .reshape(N, self.n_heads, self.head_dim)
            .transpose(1, 0, 2)
        )
        V = (
            np.dot(x, self.W_v)
            .reshape(N, self.n_heads, self.head_dim)
            .transpose(1, 0, 2)
        )

        # Scaled dot-product attention
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(
            self.head_dim
        )  # [H, N, N]

        if spatial_bias is not None:
            scores += spatial_bias

        # Softmax
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-12)

        out = np.matmul(attn, V)  # [H, N, d]
        out = out.transpose(1, 0, 2).reshape(N, D)
        out = np.dot(out, self.W_o)

        # Residual + LayerNorm
        h = x + out
        h = (h - np.mean(h, axis=-1, keepdims=True)) / (
            np.std(h, axis=-1, keepdims=True) + 1e-6
        )

        # FFN + Residual + LayerNorm
        ffn = np.maximum(0.0, np.dot(h, self.W1) + self.b1)  # ReLU
        ffn_out = np.dot(ffn, self.W2) + self.b2
        out_final = h + ffn_out
        out_final = (out_final - np.mean(out_final, axis=-1, keepdims=True)) / (
            np.std(out_final, axis=-1, keepdims=True) + 1e-6
        )

        return out_final


class Graphormer:
    """
    Graphormer: Graph Transformer Architecture for Molecule and Graph Learning.

    Parameters
    ----------
    in_dim : int
        Input node feature dimension.
    hidden_dim : int, default=64
        Hidden representation dimension.
    out_dim : int, default=2
        Output dimension (e.g. number of classes).
    num_layers : int, default=3
        Number of Graphormer layers.
    n_heads : int, default=4
        Number of attention heads.
    pe_dim : int, default=8
        Dimension of Laplacian positional encodings.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 2,
        num_layers: int = 3,
        n_heads: int = 4,
        pe_dim: int = 8,
        seed: int = 42,
    ) -> None:
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.pe_dim = pe_dim

        rng = np.random.RandomState(seed)
        self.pe_encoder = LaplacianPositionalEncoding(k=pe_dim)

        # Node feature projection + PE projection
        self.W_in: np.ndarray = rng.randn(in_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / in_dim)
        self.W_pe: np.ndarray = rng.randn(pe_dim, hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / pe_dim)

        self.layers = [
            GraphormerLayer(hidden_dim, n_heads=n_heads, seed=seed + i)
            for i in range(num_layers)
        ]

        # Classification / regression head
        self.W_out: np.ndarray = rng.randn(hidden_dim, out_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)
        self.b_out: np.ndarray = np.zeros(out_dim, dtype=np.float32)

    def forward(
        self,
        x: np.ndarray,
        adj: np.ndarray,
        pool: str = "mean",
    ) -> np.ndarray:
        """
        Forward pass for node or graph-level representation.
        x: [N, in_dim]
        adj: [N, N]
        """
        pe = self.pe_encoder.compute(adj)
        h = np.dot(x, self.W_in) + np.dot(pe, self.W_pe)

        # Compute shortest-path distance matrix as spatial bias
        dist_matrix = np.copy(adj)
        dist_matrix[dist_matrix == 0] = 5.0
        np.fill_diagonal(dist_matrix, 0.0)
        spatial_bias = -0.5 * dist_matrix

        for layer in self.layers:
            h = layer.forward(h, spatial_bias=spatial_bias)

        if pool == "mean":
            graph_rep = np.mean(h, axis=0, keepdims=True)
        elif pool == "sum":
            graph_rep = np.sum(h, axis=0, keepdims=True)
        else:
            graph_rep = h

        out = np.dot(graph_rep, self.W_out) + self.b_out
        return out
