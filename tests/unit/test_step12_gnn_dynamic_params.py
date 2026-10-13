import numpy as np

from chokkhu.models.gnn import GAT, GCN, GIN, GraphSAGE


def test_gcn_dynamic_params_and_dropout() -> None:
    np.random.seed(42)
    gcn = GCN(in_features=8, hidden_dim=16, out_features=4, num_layers=3, dropout=0.2)
    assert gcn.in_features == 8
    assert gcn.hidden_dim == 16
    assert gcn.out_features == 4
    assert gcn.num_layers == 3
    assert gcn.dropout == 0.2

    X = np.random.randn(6, 8)
    adj = np.eye(6)
    adj[0, 1] = adj[1, 0] = 1.0

    gcn.train()
    out_train = gcn.forward(X, adj)
    assert out_train.shape == (6, 4)

    gcn.eval()
    out_eval = gcn.forward(X, adj)
    assert out_eval.shape == (6, 4)

    gcn.fit(X, A=adj, epochs=2)
    preds = gcn.predict(X, adj=adj)
    assert preds.shape == (6, 4)

    params = gcn.get_params()
    assert params["num_layers"] == 3
    assert params["dropout"] == 0.2
    gcn.set_params(dropout=0.0)
    assert gcn.dropout == 0.0


def test_gat_dynamic_params_and_dropout() -> None:
    np.random.seed(42)
    gat = GAT(
        in_features=8,
        hidden_dim=12,
        out_features=4,
        num_heads=2,
        num_layers=2,
        dropout=0.1,
    )
    assert gat.num_heads == 2
    assert gat.num_layers == 2
    assert gat.dropout == 0.1

    X = np.random.randn(5, 8)
    adj = np.eye(5)

    gat.train()
    out_train = gat.forward(X, adj)
    assert out_train.shape == (5, 4)

    gat.eval()
    preds = gat.predict(X, adj=adj)
    assert preds.shape == (5, 4)

    gat.fit(X, A=adj, epochs=2)
    params = gat.get_params()
    assert params["num_heads"] == 2
    gat.set_params(num_heads=4)
    assert gat.num_heads == 4


def test_graphsage_dynamic_params_and_aggregators() -> None:
    np.random.seed(42)
    for agg in ["mean", "sum", "max"]:
        sage = GraphSAGE(
            in_features=8,
            hidden_dim=10,
            out_features=3,
            num_layers=2,
            aggregator=agg,
            dropout=0.15,
        )
        assert sage.aggregator == agg
        assert sage.dropout == 0.15

        X = np.random.randn(4, 8)
        adj = np.eye(4)

        sage.train()
        out = sage.forward(X, adj)
        assert out.shape == (4, 3)

        sage.eval()
        preds = sage.predict(X, adj=adj)
        assert preds.shape == (4, 3)

    params = sage.get_params()
    assert params["aggregator"] == "max"
    sage.set_params(aggregator="mean")
    assert sage.aggregator == "mean"


def test_gin_dynamic_params_and_dropout() -> None:
    np.random.seed(42)
    gin = GIN(
        in_features=8,
        hidden_dim=16,
        out_features=2,
        num_layers=2,
        eps=0.1,
        task="graph",
        pool_mode="sum",
        dropout=0.1,
    )
    assert gin.eps == 0.1
    assert gin.num_layers == 2
    assert gin.dropout == 0.1

    X = np.random.randn(6, 8)
    adj = np.eye(6)

    gin.train()
    out_graph = gin.forward(X, adj)
    assert out_graph.shape == (1, 2)

    gin.eval()
    preds = gin.predict(X, adj=adj)
    assert preds.shape == (1, 2)

    gin.fit(X, A=adj, epochs=2)
    params = gin.get_params()
    assert params["eps"] == 0.1
    gin.set_params(eps=0.0)
    assert gin.eps == 0.0
