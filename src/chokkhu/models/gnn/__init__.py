"""Graph Neural Networks (GNNs) Subsystem from First Principles."""

from __future__ import annotations
from .utils import (
    normalize_adjacency,
    dense_to_edge_index,
    edge_index_to_dense,
    global_pool,
)
from .layers import (
    GCNLayer,
    GATLayer,
    GraphSAGELayer,
    GINLayer,
)
from .architectures import (
    GCN,
    GAT,
    GraphSAGE,
    GIN,
)

__all__ = [
    "normalize_adjacency",
    "dense_to_edge_index",
    "edge_index_to_dense",
    "global_pool",
    "GCNLayer",
    "GATLayer",
    "GraphSAGELayer",
    "GINLayer",
    "GCN",
    "GAT",
    "GraphSAGE",
    "GIN",
]
