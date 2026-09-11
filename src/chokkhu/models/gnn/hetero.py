"""Heterogeneous Graph Neural Networks (HeteroGCN) and Spatio-Temporal GNNs (ST-GCN) in pure NumPy."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


class HeteroGCN:
    """Heterogeneous Relational Graph Convolutional Network (HeteroGCN) in pure NumPy.

    Performs type-specific relational message passing across multi-relational
    graphs with distinct node and relation types (Schlichtkrull et al. 2018).

    Parameters
    ----------
    in_channels_dict : Dict[str, int]
        Mapping of node types to input feature dimensionality.
    out_channels : int
        Output feature dimensionality for all node types.
    relations : Sequence[Tuple[str, str, str]]
        List of relational triplets: (src_type, relation_name, dst_type).
    aggregate : str, default="sum"
        Relational aggregation strategy across relation types ("sum" or "mean").
    seed : int, default=42
    """

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        out_channels: int,
        relations: Sequence[Tuple[str, str, str]],
        aggregate: str = "sum",
        seed: int = 42,
    ) -> None:
        self.in_channels_dict = in_channels_dict
        self.out_channels = int(out_channels)
        self.relations = list(relations)
        self.aggregate = aggregate

        rng = np.random.default_rng(seed)
        self.weights: Dict[str, np.ndarray] = {}

        for src_type, rel_name, dst_type in self.relations:
            in_dim = in_channels_dict.get(src_type, out_channels)
            key = f"{src_type}__{rel_name}__{dst_type}"
            scale = np.sqrt(2.0 / in_dim)
            self.weights[key] = rng.normal(
                0, scale, size=(in_dim, out_channels)
            ).astype(np.float32)

    def forward(
        self,
        x_dict: Dict[str, np.ndarray],
        edge_index_dict: Dict[Tuple[str, str, str], np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Performs relational message passing across heterogeneous graph.

        Parameters
        ----------
        x_dict : Dict[str, np.ndarray]
            Node features mapping: node_type -> (num_nodes, in_dim).
        edge_index_dict : Dict[Tuple[str, str, str], np.ndarray]
            Edge indices mapping: (src_type, rel, dst_type) -> (2, num_edges).

        Returns
        -------
        Dict[str, np.ndarray]
            Updated node representations: node_type -> (num_nodes, out_channels).
        """
        # Intermediate message accumulator per destination node type
        messages_by_dst: Dict[str, List[np.ndarray]] = {}

        for rel in self.relations:
            src_type, rel_name, dst_type = rel
            rel_key = f"{src_type}__{rel_name}__{dst_type}"

            if (
                rel not in edge_index_dict
                or src_type not in x_dict
                or dst_type not in x_dict
            ):
                continue

            edges = edge_index_dict[rel]
            src_x = x_dict[src_type]
            dst_x = x_dict[dst_type]

            num_dst_nodes = len(dst_x)
            W_rel = self.weights[rel_key]

            # 1. Linear transformation of source features: (num_src_nodes, out_channels)
            h_src = np.matmul(src_x, W_rel)

            # 2. Scatter message aggregation along destination nodes
            rel_msg: np.ndarray = np.zeros(
                (num_dst_nodes, self.out_channels), dtype=np.float32
            )
            deg_dst: np.ndarray = np.zeros(num_dst_nodes, dtype=np.float32)

            src_indices = edges[0]
            dst_indices = edges[1]

            for s, d in zip(src_indices, dst_indices):
                rel_msg[d] += h_src[s]
                deg_dst[d] += 1.0

            # Degree normalization
            deg_dst[deg_dst == 0] = 1.0
            rel_msg = rel_msg / deg_dst[:, np.newaxis]

            if dst_type not in messages_by_dst:
                messages_by_dst[dst_type] = []
            messages_by_dst[dst_type].append(rel_msg)

        # 3. Aggregate across all incoming relation types for each destination node type
        out_dict: Dict[str, np.ndarray] = {}
        for node_type, msg_list in messages_by_dst.items():
            if self.aggregate == "mean":
                stacked = np.stack(msg_list, axis=0)
                out_dict[node_type] = np.mean(stacked, axis=0)
            else:  # sum
                out_dict[node_type] = np.sum(msg_list, axis=0)

        # Retain original node types that received no messages
        for n_type, feat in x_dict.items():
            if n_type not in out_dict:
                out_dict[n_type] = feat

        return out_dict


class SpatioTemporalGCN:
    """Spatial-Temporal Graph Convolutional Network (ST-GCN) in pure NumPy (Yan et al. 2018).

    Combines spatial graph convolutions across graph nodes with 1D temporal
    convolutions across the sequence time dimension for dynamic spatio-temporal modeling.

    Parameters
    ----------
    in_channels : int
        Input feature dimension per node.
    out_channels : int
        Output feature dimension per node.
    temporal_kernel_size : int, default=9
        1D temporal convolution kernel width across time frames.
    stride : int, default=1
        Temporal stride parameter.
    seed : int, default=42
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_kernel_size: int = 9,
        stride: int = 1,
        seed: int = 42,
    ) -> None:
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.temporal_kernel_size = int(temporal_kernel_size)
        self.stride = int(stride)

        rng = np.random.default_rng(seed)
        scale_s = np.sqrt(2.0 / in_channels)
        scale_t = np.sqrt(2.0 / (out_channels * temporal_kernel_size))

        # Spatial transformation matrix: (in_channels, out_channels)
        self.w_spatial: np.ndarray = rng.normal(
            0, scale_s, size=(in_channels, out_channels)
        ).astype(np.float32)

        # Temporal 1D convolution weights: (out_channels, out_channels, temporal_kernel_size)
        self.w_temporal: np.ndarray = rng.normal(
            0, scale_t, size=(out_channels, out_channels, temporal_kernel_size)
        ).astype(np.float32)
        self.b_temporal: np.ndarray = np.zeros(out_channels, dtype=np.float32)

        # Residual projection if channel dimensions change
        self.w_res: Optional[np.ndarray] = None
        if in_channels != out_channels:
            self.w_res = rng.normal(
                0, scale_s, size=(in_channels, out_channels)
            ).astype(np.float32)

    def forward(self, x: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Forward pass for Spatio-Temporal Graph Convolution.

        Parameters
        ----------
        x : np.ndarray
            Input spatio-temporal tensor of shape (B, T, N, in_channels).
        A : np.ndarray
            Spatial adjacency / normalized Laplacian matrix of shape (N, N).

        Returns
        -------
        np.ndarray
            Output spatio-temporal tensor of shape (B, T_out, N, out_channels).
        """
        B, T, N, C_in = x.shape
        x_arr = np.asarray(x, dtype=np.float32)
        A_norm = np.asarray(A, dtype=np.float32)

        # 1. Spatial Graph Convolution: Z_t = A @ X_t @ W_spatial
        # (B, T, N, in_channels) @ W_spatial -> (B, T, N, out_channels)
        x_proj = np.matmul(x_arr, self.w_spatial)

        # Graph message passing via einsum: A @ X
        # A: (N, N), x_proj: (B, T, N, out_channels) -> (B, T, N, out_channels)
        z_spatial = np.einsum("vw,btwd->btvd", A_norm, x_proj)
        z_spatial = np.maximum(0.0, z_spatial)  # ReLU

        # 2. Temporal 1D Convolution along time dimension T
        pad_t = (self.temporal_kernel_size - 1) // 2
        z_pad = np.pad(
            z_spatial, ((0, 0), (pad_t, pad_t), (0, 0), (0, 0)), mode="constant"
        )

        t_out = (T + 2 * pad_t - self.temporal_kernel_size) // self.stride + 1
        out = np.zeros((B, t_out, N, self.out_channels), dtype=np.float32)

        for step in range(t_out):
            t_idx = step * self.stride
            window = z_pad[
                :, t_idx : t_idx + self.temporal_kernel_size, :, :
            ]  # (B, K_t, N, D)
            # Dot product with temporal kernel
            conv_step = (
                np.einsum("btnd,odk->bno", window, self.w_temporal) + self.b_temporal
            )
            out[:, step, :, :] = conv_step

        # 3. Residual connection
        if self.w_res is not None:
            res = np.matmul(x_arr[:, :: self.stride, :, :], self.w_res)
        else:
            res = x_arr[:, :: self.stride, :, :]

        if res.shape == out.shape:
            out = out + res

        return np.maximum(0.0, out)
