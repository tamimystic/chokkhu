"""Graph Neural Network Utilities: Adjacency Normalization, Conversions, and Global Pooling."""

from __future__ import annotations

from typing import Optional, Union
import numpy as np

from chokkhu.core.tensor import Tensor


def normalize_adjacency(
    adj: Union[np.ndarray, Tensor],
    self_loops: bool = True,
    symmetric: bool = True,
) -> np.ndarray:
    """Compute normalized adjacency matrix.

    Symmetric normalization (Kipf & Welling, 2017):
        A_hat = D~^(-1/2) * A~ * D~^(-1/2), where A~ = A + I_N

    Random walk normalization:
        A_hat = D~^(-1) * A~, where A~ = A + I_N
    """
    A = adj.data if isinstance(adj, Tensor) else np.asarray(adj, dtype=np.float64)
    N = A.shape[0]

    if self_loops:
        A = A + np.eye(N, dtype=np.float64)

    deg = np.sum(A, axis=1)

    if symmetric:
        deg_inv_sqrt = np.zeros_like(deg)
        pos_mask = deg > 0
        deg_inv_sqrt[pos_mask] = np.power(deg[pos_mask], -0.5)
        D_mat = np.diag(deg_inv_sqrt)
        A_norm = np.matmul(np.matmul(D_mat, A), D_mat)
    else:
        deg_inv = np.zeros_like(deg)
        pos_mask = deg > 0
        deg_inv[pos_mask] = np.power(deg[pos_mask], -1.0)
        D_mat = np.diag(deg_inv)
        A_norm = np.matmul(D_mat, A)

    return A_norm


def dense_to_edge_index(adj: Union[np.ndarray, Tensor]) -> np.ndarray:
    """Convert a dense adjacency matrix to COO edge_index of shape (2, num_edges)."""
    A = adj.data if isinstance(adj, Tensor) else np.asarray(adj)
    src, dst = np.nonzero(A)
    return np.vstack((src, dst))


def edge_index_to_dense(
    edge_index: np.ndarray, num_nodes: Optional[int] = None
) -> np.ndarray:
    """Convert COO edge_index of shape (2, num_edges) to a dense adjacency matrix."""
    if num_nodes is None:
        num_nodes = int(np.max(edge_index) + 1) if edge_index.size > 0 else 0
    adj: np.ndarray = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    if edge_index.shape[1] > 0:
        adj[edge_index[0], edge_index[1]] = 1.0
    return adj


def global_pool(
    x: Union[np.ndarray, Tensor],
    batch: Optional[np.ndarray] = None,
    mode: str = "mean",
) -> Tensor:
    """Perform global pooling over graph node representations.

    Parameters
    ----------
    x : Union[np.ndarray, Tensor]
        Node feature tensor of shape (N, F).
    batch : Optional[np.ndarray]
        Batch indicator vector of length N mapping each node to graph ID (0 to B-1).
        If None, all nodes belong to a single graph.
    mode : str
        Pooling mode: 'mean', 'sum', or 'max'.
    """
    x_t = x if isinstance(x, Tensor) else Tensor(x, requires_grad=False)
    data = x_t.data

    if batch is None:
        if mode == "mean":
            pooled = np.mean(data, axis=0, keepdims=True)
        elif mode == "sum":
            pooled = np.sum(data, axis=0, keepdims=True)
        elif mode == "max":
            pooled = np.max(data, axis=0, keepdims=True)
        else:
            raise ValueError(f"Unknown pooling mode: {mode}")
        return Tensor(pooled, requires_grad=x_t.requires_grad)

    num_graphs = int(np.max(batch) + 1) if batch.size > 0 else 1
    out = np.zeros((num_graphs, data.shape[1]), dtype=np.float64)

    for i in range(num_graphs):
        mask = batch == i
        if np.any(mask):
            sub = data[mask]
            if mode == "mean":
                out[i] = np.mean(sub, axis=0)
            elif mode == "sum":
                out[i] = np.sum(sub, axis=0)
            elif mode == "max":
                out[i] = np.max(sub, axis=0)
            else:
                raise ValueError(f"Unknown pooling mode: {mode}")

    return Tensor(out, requires_grad=x_t.requires_grad)
