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
    LaplacianPositionalEncoding,
    GraphormerLayer,
    Graphormer,
    RGCNLayer,
    RGCNClassifier,
    EGNNLayer,
    EGNN,
    TemporalGraphNetwork,
    HypergraphConvolution,
    HGNN,
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


def test_graphormer_and_laplacian_pe() -> None:
    N, in_dim = 6, 8
    adj = np.array(
        [
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ],
        dtype=np.float32,
    )

    pe_encoder = LaplacianPositionalEncoding(k=4)
    pe = pe_encoder.compute(adj)
    assert pe.shape == (N, 4)

    x = np.random.randn(N, in_dim).astype(np.float32)
    layer = GraphormerLayer(hidden_dim=16, n_heads=2)
    h = np.random.randn(N, 16).astype(np.float32)
    out_layer = layer.forward(h)
    assert out_layer.shape == (N, 16)

    # Full Graphormer graph classification
    model = Graphormer(
        in_dim=in_dim, hidden_dim=16, out_dim=2, num_layers=2, n_heads=2, pe_dim=4
    )
    graph_out = model.forward(x, adj, pool="mean")
    assert graph_out.shape == (1, 2)

    # Node-level representation
    node_out = model.forward(x, adj, pool="none")
    assert node_out.shape == (N, 2)


def test_rgcn_multi_relational() -> None:
    N, in_dim = 5, 8
    x = np.random.randn(N, in_dim).astype(np.float32)

    # 2 relations
    edge_indices = {
        0: np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64),
        1: np.array([[2, 3, 4], [3, 4, 2]], dtype=np.int64),
    }

    rgcn_layer = RGCNLayer(in_dim=in_dim, out_dim=12, num_relations=2, num_bases=2)
    out_layer = rgcn_layer.forward(x, edge_indices)
    assert out_layer.shape == (N, 12)

    classifier = RGCNClassifier(
        in_dim=in_dim, hidden_dim=12, out_dim=3, num_relations=2, num_bases=2
    )
    probs = classifier.forward(x, edge_indices)
    assert probs.shape == (N, 3)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(N), atol=1e-5)


def test_egnn_3d_equivariance() -> None:
    N, in_dim = 4, 6
    h = np.random.randn(N, in_dim).astype(np.float32)
    x = np.random.randn(N, 3).astype(np.float32)

    # Fully connected edge index
    src, dst = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                src.append(i)
                dst.append(j)
    edges = np.array([src, dst], dtype=np.int64)
    layer = EGNNLayer(in_dim=in_dim, hidden_dim=16, out_dim=8)
    h_l, x_l = layer.forward(h, x, edges)
    assert h_l.shape == (N, 8)
    assert x_l.shape == (N, 3)

    egnn = EGNN(in_dim=in_dim, hidden_dim=16, out_dim=8, num_layers=2)
    h_out, x_out = egnn.forward(h, x, edges)
    assert h_out.shape == (N, 8)
    assert x_out.shape == (N, 3)

    # Test Rotation Equivariance: R * f(x) == f(R * x)
    # Random 3D orthogonal rotation matrix (det = 1)
    theta = np.pi / 4.0
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    x_rot = np.dot(x, R.T)
    h_rot_out, x_rot_out = egnn.forward(h, x_rot, edges)

    # Invariant features should match exactly
    np.testing.assert_allclose(h_out, h_rot_out, atol=1e-4)

    # Rotated coordinates should equal rotated output coordinates: R * x_out == x_rot_out
    expected_x_rot = np.dot(x_out, R.T)
    np.testing.assert_allclose(x_rot_out, expected_x_rot, atol=1e-4)


def test_temporal_graph_network() -> None:
    tgn = TemporalGraphNetwork(node_dim=8, edge_dim=4, memory_dim=16, time_dim=8)

    src_mem = np.random.randn(2, 16).astype(np.float32)
    dst_mem = np.random.randn(2, 16).astype(np.float32)
    dt = np.array([0.5, 1.2], dtype=np.float32)
    edge_feat = np.random.randn(2, 4).astype(np.float32)

    msgs = tgn.compute_messages(src_mem, dst_mem, dt, edge_feat)
    assert msgs.shape == (2, 16)

    new_mem = tgn.update_memory(src_mem, msgs)
    assert new_mem.shape == (2, 16)


def test_hypergraph_convolution_and_hgnn() -> None:
    N, in_dim = 6, 8
    # 3 hyperedges: e0 contains {0, 1, 2}, e1 contains {2, 3, 4}, e2 contains {4, 5, 0}
    H = np.array(
        [
            [1, 0, 1],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    X = np.random.randn(N, in_dim).astype(np.float32)
    hgc = HypergraphConvolution(in_dim=in_dim, out_dim=12)
    out_conv = hgc.forward(X, H)
    assert out_conv.shape == (N, 12)

    hgnn = HGNN(in_dim=in_dim, hidden_dim=12, out_dim=3)
    probs = hgnn.forward(X, H)
    assert probs.shape == (N, 3)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(N), atol=1e-5)
