from __future__ import annotations

import numpy as np


class RGCNLayer:
    """
    Relational Graph Convolutional Network (R-GCN) Layer for Knowledge Graphs
    and Multi-Relational Data.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_dim : int
        Output feature dimension.
    num_relations : int
        Number of distinct relation types.
    num_bases : int | None, default=None
        Number of basis matrices for parameter regularization.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        num_bases: int | None = None,
        seed: int = 42,
    ) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases

        rng = np.random.RandomState(seed)

        # Self-loop transformation
        self.W_self: np.ndarray = rng.randn(in_dim, out_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / in_dim)
        self.bias: np.ndarray = np.zeros(out_dim, dtype=np.float32)

        if num_bases is not None:
            # Basis decomposition: W_r = sum_b a_rb * V_b
            self.bases: np.ndarray = rng.randn(num_bases, in_dim, out_dim).astype(
                np.float32
            ) * np.sqrt(2.0 / in_dim)
            self.coeffs: np.ndarray = (
                rng.randn(num_relations, num_bases).astype(np.float32) * 0.1
            )
        else:
            # Direct relation weight matrices
            self.W_rel: np.ndarray = rng.randn(num_relations, in_dim, out_dim).astype(
                np.float32
            ) * np.sqrt(2.0 / in_dim)

    def forward(
        self,
        x: np.ndarray,
        edge_indices_by_rel: dict[int, np.ndarray],
    ) -> np.ndarray:
        """
        Forward pass.
        x: [N, in_dim]
        edge_indices_by_rel: {rel_id: [2, E_r]} where edge_indices_by_rel[r][0] = source, [1] = target
        """
        N = x.shape[0]
        out = np.dot(x, self.W_self)

        for r, edges in edge_indices_by_rel.items():
            if edges.size == 0:
                continue

            if self.num_bases is not None:
                # Reconstruct W_r from basis
                W_r = np.tensordot(
                    self.coeffs[r], self.bases, axes=(0, 0)
                )  # [in_dim, out_dim]
            else:
                W_r = self.W_rel[r]

            src, dst = edges[0], edges[1]
            src_features = x[src]  # [E_r, in_dim]
            msg = np.dot(src_features, W_r)  # [E_r, out_dim]

            # Aggregate to destination nodes
            deg_r = np.bincount(dst, minlength=N).astype(np.float32)
            deg_r[deg_r == 0] = 1.0

            agg = np.zeros((N, self.out_dim), dtype=np.float32)
            np.add.at(agg, dst, msg)
            out += agg / deg_r.reshape(-1, 1)

        out += self.bias
        return np.maximum(0.0, out)  # ReLU activation


class RGCNClassifier:
    """
    R-GCN Multi-Relational Entity Classifier.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden representation dimension.
    out_dim : int
        Number of output classes.
    num_relations : int
        Number of relations.
    num_bases : int | None, default=4
        Number of basis matrices.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_relations: int,
        num_bases: int | None = 4,
        seed: int = 42,
    ) -> None:
        self.layer1 = RGCNLayer(
            in_dim, hidden_dim, num_relations, num_bases=num_bases, seed=seed
        )
        self.layer2 = RGCNLayer(
            hidden_dim, out_dim, num_relations, num_bases=num_bases, seed=seed + 1
        )

    def forward(
        self,
        x: np.ndarray,
        edge_indices_by_rel: dict[int, np.ndarray],
    ) -> np.ndarray:
        h = self.layer1.forward(x, edge_indices_by_rel)
        out = self.layer2.forward(h, edge_indices_by_rel)
        # Softmax
        exp_out = np.exp(out - np.max(out, axis=1, keepdims=True))
        return exp_out / np.sum(exp_out, axis=1, keepdims=True)
