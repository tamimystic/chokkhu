"""Unit tests for Graph Neural Networks (GNN) Universe (GCN, GAT, GraphSAGE, GIN)."""

from __future__ import annotations
import numpy as np

from chokkhu.core.tensor import Tensor
from chokkhu.models.gnn import (
    GCN,
    GAT,
    GraphSAGE,
    GIN,
    GATLayer,
    normalize_adjacency,
    dense_to_edge_index,
    edge_index_to_dense,
    global_pool,
)
from chokkhu.models import train


def test_normalize_adjacency_and_conversions() -> None:
    # 3-node cycle graph
    adj = np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.float64,
    )

    # Symmetric normalization: with self-loops degree is 3 for all nodes
    # A_hat = (1/3) * (A + I) = 1/3 for all elements
    norm_sym = normalize_adjacency(adj, self_loops=True, symmetric=True)
    assert norm_sym.shape == (3, 3)
    np.testing.assert_allclose(norm_sym, np.full((3, 3), 1.0 / 3.0), atol=1e-5)

    # Random walk normalization
    norm_rw = normalize_adjacency(adj, self_loops=True, symmetric=False)
    assert norm_rw.shape == (3, 3)
    np.testing.assert_allclose(norm_rw, np.full((3, 3), 1.0 / 3.0), atol=1e-5)

    # Conversions
    edge_index = dense_to_edge_index(adj)
    assert edge_index.shape == (2, 6)
    reconstructed_adj = edge_index_to_dense(edge_index, num_nodes=3)
    np.testing.assert_array_equal(adj, reconstructed_adj)


def test_global_pool_modes() -> None:
    x = Tensor(
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ]
        )
    )
    # 2 nodes in graph 0, 2 nodes in graph 1
    batch = np.array([0, 0, 1, 1])

    # Mean pooling
    p_mean = global_pool(x, batch=batch, mode="mean")
    assert p_mean.shape == (2, 2)
    np.testing.assert_allclose(p_mean.data, [[2.0, 3.0], [6.0, 7.0]])

    # Sum pooling
    p_sum = global_pool(x, batch=batch, mode="sum")
    assert p_sum.shape == (2, 2)
    np.testing.assert_allclose(p_sum.data, [[4.0, 6.0], [12.0, 14.0]])

    # Max pooling
    p_max = global_pool(x, batch=batch, mode="max")
    assert p_max.shape == (2, 2)
    np.testing.assert_allclose(p_max.data, [[3.0, 4.0], [7.0, 8.0]])


def test_gcn_forward_node_and_graph_classification() -> None:
    N, F = 5, 8
    X = Tensor(np.random.randn(N, F), requires_grad=True)
    adj = np.random.randint(0, 2, size=(N, N)).astype(np.float64)
    np.fill_diagonal(adj, 1.0)

    # Node classification GCN
    gcn_node = GCN(
        in_features=F, hidden_dim=16, out_features=3, num_layers=2, task="node"
    )
    out_node = gcn_node(X, adj)
    assert out_node.shape == (N, 3)

    # Graph classification GCN
    batch = np.array([0, 0, 0, 1, 1])
    gcn_graph = GCN(
        in_features=F,
        hidden_dim=16,
        out_features=2,
        num_layers=2,
        task="graph",
        pool_mode="mean",
    )
    out_graph = gcn_graph(X, adj, batch=batch)
    assert out_graph.shape == (2, 2)


def test_gat_multi_head_attention() -> None:
    N, F = 6, 8
    X = Tensor(np.random.randn(N, F), requires_grad=True)
    adj = np.random.randint(0, 2, size=(N, N)).astype(np.float64)
    np.fill_diagonal(adj, 1.0)

    # GAT layer with 3 heads, concat=True -> out_dim = 3 * 4 = 12
    gat_layer = GATLayer(in_features=F, out_features=4, num_heads=3, concat=True)
    out_layer = gat_layer(X, adj)
    assert out_layer.shape == (N, 12)

    # Full GAT model for node classification
    gat_model = GAT(
        in_features=F,
        hidden_dim=8,
        out_features=3,
        num_heads=2,
        num_layers=2,
        task="node",
    )
    out_model = gat_model(X, adj)
    assert out_model.shape == (N, 3)


def test_graphsage_aggregators() -> None:
    N, F = 5, 8
    X = Tensor(np.random.randn(N, F), requires_grad=True)
    adj = np.random.randint(0, 2, size=(N, N)).astype(np.float64)
    np.fill_diagonal(adj, 1.0)

    # Mean aggregator
    sage_mean = GraphSAGE(
        in_features=F, hidden_dim=12, out_features=4, aggregator="mean"
    )
    out_mean = sage_mean(X, adj)
    assert out_mean.shape == (N, 4)

    # Max aggregator
    sage_max = GraphSAGE(in_features=F, hidden_dim=12, out_features=4, aggregator="max")
    out_max = sage_max(X, adj)
    assert out_max.shape == (N, 4)

    # Sum aggregator
    sage_sum = GraphSAGE(in_features=F, hidden_dim=12, out_features=4, aggregator="sum")
    out_sum = sage_sum(X, adj)
    assert out_sum.shape == (N, 4)


def test_gin_isomorphism_network() -> None:
    N, F = 6, 8
    X = Tensor(np.random.randn(N, F), requires_grad=True)
    adj = np.random.randint(0, 2, size=(N, N)).astype(np.float64)
    np.fill_diagonal(adj, 1.0)
    batch = np.array([0, 0, 0, 1, 1, 1])

    gin = GIN(
        in_features=F,
        hidden_dim=16,
        out_features=2,
        num_layers=3,
        task="graph",
        pool_mode="sum",
    )
    out = gin(X, adj, batch=batch)
    assert out.shape == (2, 2)


def test_train_gnn_via_engine() -> None:
    N, F = 6, 8
    X = np.random.randn(N, F)
    y = np.array([0, 1, 0, 1, 0, 1])
    adj = np.eye(N, dtype=np.float64)

    model = train(
        model="gcn",
        X_train=X,
        y_train=y,
        adj=adj,
        in_features=F,
        hidden_dim=8,
        out_features=2,
        epochs=2,
    )
    assert isinstance(model, GCN)
    preds = model.predict(X, adj=adj)
    assert preds.shape == (N, 2)
