"""Graph Neural Network Layers: GCNLayer, GATLayer, GraphSAGELayer, GINLayer."""

from __future__ import annotations

from typing import Optional, Union
import numpy as np

from chokkhu.core.tensor import Tensor
from ..dl.layers import Module, Parameter
from .utils import normalize_adjacency


class GCNLayer(Module):
    """Graph Convolutional Network Layer (Kipf & Welling, 2017).

    H' = A_hat * H * W + b
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        normalize_adj: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.normalize_adj = normalize_adj

        # Xavier Uniform initialization
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.weight = Parameter(
            np.random.uniform(-limit, limit, (in_features, out_features))
        )
        if bias:
            self.bias: Optional[Parameter] = Parameter(np.zeros((out_features,)))
        else:
            self.bias = None

    def forward(
        self,
        x: Union[np.ndarray, Tensor],
        adj: Union[np.ndarray, Tensor],
    ) -> Tensor:
        """Forward pass for Graph Convolution."""
        X_t = x if isinstance(x, Tensor) else Tensor(x, requires_grad=True)
        A_mat = (
            adj.data if isinstance(adj, Tensor) else np.asarray(adj, dtype=np.float64)
        )

        if self.normalize_adj:
            A_hat = normalize_adjacency(A_mat, self_loops=True, symmetric=True)
        else:
            A_hat = A_mat

        # H * W
        hw = np.matmul(X_t.data, self.weight.data)
        # A_hat * (H * W)
        out = np.matmul(A_hat, hw)

        if self.bias is not None:
            out = out + self.bias.data

        return Tensor(out, requires_grad=X_t.requires_grad or self.weight.requires_grad)


class GATLayer(Module):
    """Graph Attention Network Layer (Veličković et al., 2018).

    Computes multi-head self-attention over graph neighborhoods.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 1,
        concat: bool = True,
        alpha: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.alpha = alpha

        limit = np.sqrt(6.0 / (in_features + out_features * num_heads))
        self.weight = Parameter(
            np.random.uniform(-limit, limit, (num_heads, in_features, out_features))
        )
        # Attention parameter vectors a_src, a_dst
        self.a_src = Parameter(
            np.random.uniform(-limit, limit, (num_heads, out_features, 1))
        )
        self.a_dst = Parameter(
            np.random.uniform(-limit, limit, (num_heads, out_features, 1))
        )

    def forward(
        self,
        x: Union[np.ndarray, Tensor],
        adj: Union[np.ndarray, Tensor],
    ) -> Tensor:
        """Forward multi-head graph attention."""
        X_t = x if isinstance(x, Tensor) else Tensor(x, requires_grad=True)
        X_data = X_t.data
        N = X_data.shape[0]

        A = adj.data if isinstance(adj, Tensor) else np.asarray(adj, dtype=np.float64)
        # Add self-loops if not present
        A_with_loops = np.maximum(A, np.eye(N, dtype=np.float64))

        head_outputs = []

        for h in range(self.num_heads):
            # Wh: (N, out_features)
            Wh = np.matmul(X_data, self.weight.data[h])

            # e_i = Wh * a_src, e_j = Wh * a_dst
            f_src = np.matmul(Wh, self.a_src.data[h])  # (N, 1)
            f_dst = np.matmul(Wh, self.a_dst.data[h])  # (N, 1)

            # Raw attention logits: e_ij = LeakyReLU(f_src_i + f_dst_j)
            logits = f_src + f_dst.T
            # LeakyReLU
            logits = np.where(logits > 0, logits, logits * self.alpha)

            # Mask non-edges with large negative value
            mask = A_with_loops > 0
            masked_logits = np.where(mask, logits, -1e9)

            # Softmax along neighbors (columns)
            max_val = np.max(masked_logits, axis=1, keepdims=True)
            exp_logits = np.exp(masked_logits - max_val) * mask
            sum_exp = np.sum(exp_logits, axis=1, keepdims=True)
            sum_exp = np.where(sum_exp > 0, sum_exp, 1.0)
            attn_weights = exp_logits / sum_exp

            # Aggregated output: h_i' = sum_j alpha_ij * Wh_j
            h_out = np.matmul(attn_weights, Wh)
            head_outputs.append(h_out)

        if self.concat:
            out = np.concatenate(head_outputs, axis=-1)
        else:
            out = np.mean(np.stack(head_outputs, axis=0), axis=0)

        return Tensor(out, requires_grad=X_t.requires_grad or self.weight.requires_grad)


class GraphSAGELayer(Module):
    """GraphSAGE Layer (Hamilton et al., 2017).

    Inductive representation learning via Mean, Max-Pooling, or Sum Aggregation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        aggregator: str = "mean",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.aggregator = aggregator.lower()
        self.normalize = normalize

        limit = np.sqrt(6.0 / (2 * in_features + out_features))
        self.weight = Parameter(
            np.random.uniform(-limit, limit, (2 * in_features, out_features))
        )
        if self.aggregator == "max":
            self.pool_weight = Parameter(
                np.random.uniform(-limit, limit, (in_features, in_features))
            )

    def forward(
        self,
        x: Union[np.ndarray, Tensor],
        adj: Union[np.ndarray, Tensor],
    ) -> Tensor:
        """Forward aggregation and update."""
        X_t = x if isinstance(x, Tensor) else Tensor(x, requires_grad=True)
        X_data = X_t.data
        N = X_data.shape[0]

        A = adj.data if isinstance(adj, Tensor) else np.asarray(adj, dtype=np.float64)

        # Aggregate neighbor features
        if self.aggregator == "mean":
            deg = np.sum(A, axis=1, keepdims=True)
            deg_safe = np.where(deg > 0, deg, 1.0)
            agg = np.matmul(A, X_data) / deg_safe
        elif self.aggregator == "sum":
            agg = np.matmul(A, X_data)
        elif self.aggregator == "max":
            # Max pooling over projected neighbor features
            proj = np.matmul(X_data, self.pool_weight.data)
            agg = np.zeros_like(X_data)
            for i in range(N):
                neighbors = np.where(A[i] > 0)[0]
                if len(neighbors) > 0:
                    agg[i] = np.max(proj[neighbors], axis=0)
                else:
                    agg[i] = proj[i]
        else:
            raise ValueError(f"Unsupported aggregator: {self.aggregator}")

        # Concatenate self features and aggregated neighbor features
        combined = np.concatenate([X_data, agg], axis=1)
        out = np.matmul(combined, self.weight.data)

        if self.normalize:
            norm = np.linalg.norm(out, axis=1, keepdims=True)
            norm = np.where(norm > 0, norm, 1.0)
            out = out / norm

        return Tensor(out, requires_grad=X_t.requires_grad or self.weight.requires_grad)


class GINLayer(Module):
    """Graph Isomorphism Network (GIN) Layer (Xu et al., 2019).

    h_v' = MLP((1 + eps) * h_v + sum_{u in N(v)} h_u)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 0.0,
        train_eps: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.train_eps = train_eps

        # 2-layer MLP for maximal expressive power
        limit1 = np.sqrt(6.0 / (in_features + out_features))
        limit2 = np.sqrt(6.0 / (out_features + out_features))
        self.w1 = Parameter(
            np.random.uniform(-limit1, limit1, (in_features, out_features))
        )
        self.b1 = Parameter(np.zeros((out_features,)))
        self.w2 = Parameter(
            np.random.uniform(-limit2, limit2, (out_features, out_features))
        )
        self.b2 = Parameter(np.zeros((out_features,)))

    def forward(
        self,
        x: Union[np.ndarray, Tensor],
        adj: Union[np.ndarray, Tensor],
    ) -> Tensor:
        """Forward GIN message passing."""
        X_t = x if isinstance(x, Tensor) else Tensor(x, requires_grad=True)
        X_data = X_t.data

        A = adj.data if isinstance(adj, Tensor) else np.asarray(adj, dtype=np.float64)

        # sum_{u in N(v)} h_u
        neigh_sum = np.matmul(A, X_data)
        # (1 + eps) * h_v + sum
        h_sum = (1.0 + self.eps) * X_data + neigh_sum

        # MLP(h_sum)
        h1 = np.maximum(0, np.matmul(h_sum, self.w1.data) + self.b1.data)
        out = np.matmul(h1, self.w2.data) + self.b2.data

        return Tensor(out, requires_grad=X_t.requires_grad or self.w1.requires_grad)
