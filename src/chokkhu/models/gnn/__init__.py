"""Graph Neural Networks (GNNs) Subsystem from First Principles."""

from __future__ import annotations

from .architectures import (
    GAT,
    GCN,
    GIN,
    GraphSAGE,
)
from .contrastive import GRACE, GraphCL
from .equivariant import EGNN, EGNNLayer
from .graph_transformer import Graphormer, GraphormerLayer, LaplacianPositionalEncoding
from .hetero import HeteroGCN, SpatioTemporalGCN
from .hypergraph import HGNN, HypergraphConvolution
from .layers import (
    GATLayer,
    GCNLayer,
    GINLayer,
    GraphSAGELayer,
)
from .relational import RGCNClassifier, RGCNLayer
from .temporal import TemporalGraphNetwork
from .utils import (
    dense_to_edge_index,
    edge_index_to_dense,
    global_pool,
    normalize_adjacency,
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
    "LaplacianPositionalEncoding",
    "GraphormerLayer",
    "Graphormer",
    "RGCNLayer",
    "RGCNClassifier",
    "EGNNLayer",
    "EGNN",
    "TemporalGraphNetwork",
    "HypergraphConvolution",
    "HGNN",
    "HeteroGCN",
    "SpatioTemporalGCN",
    "GraphCL",
    "GRACE",
]
