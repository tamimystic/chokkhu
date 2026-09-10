"""Comprehensive Unit Tests for Chokkhu Recommendation Subsystem.

Tests:
- SVDRecommender (Funk-SVD with biases)
- SVDPlusPlus (SVD++ with implicit history)
- NMFRecommender (Multiplicative non-negative factorization)
- ImplicitALS (Alternating Least Squares)
- BayesianPersonalizedRanking (BPR-MF)
- NeuralCollaborativeFiltering (NeuMF GMF + MLP)
- WideAndDeep (Memorization + Generalization)
- DeepFM (1st order + 2nd order + Deep)
- DLRM (Dense MLP + Sparse embeddings + Dot-product interaction)
- SASRec (Self-Attention Sequential Recommendation)
- GRU4Rec (Recurrent Session Recommendation)
- Recommendation Metrics (HR@K, NDCG@K, MRR@K, Precision@K, Recall@K, MAP@K)
"""

import numpy as np

from chokkhu.models.recommendation import (
    BayesianPersonalizedRanking,
    DeepFM,
    DLRM,
    GRU4Rec,
    ImplicitALS,
    NMFRecommender,
    NeuralCollaborativeFiltering,
    SASRec,
    SVDPlusPlus,
    SVDRecommender,
    WideAndDeep,
    hit_rate_at_k,
    map_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_svd_recommender():
    users = ["u1", "u1", "u2", "u2", "u3", "u3", "u4"]
    items = ["i1", "i2", "i1", "i3", "i2", "i4", "i1"]
    ratings = [5.0, 4.0, 4.5, 1.0, 5.0, 3.5, 4.0]

    model = SVDRecommender(n_factors=8, n_epochs=15, lr=0.01, seed=42)
    model.fit(users, items, ratings)

    pred = model.predict("u1", "i1")
    assert isinstance(pred, float)
    assert 1.0 <= pred <= 6.0

    recs = model.recommend("u1", top_k=2, filter_interacted=True)
    assert isinstance(recs, list)
    for itm, score in recs:
        assert itm not in ["i1", "i2"]

    cold_pred = model.predict("unknown_user", "unknown_item")
    assert isinstance(cold_pred, float)


def test_svd_plus_plus():
    users = [1, 1, 2, 2, 3, 3]
    items = [10, 20, 10, 30, 20, 40]
    ratings = [4.0, 5.0, 3.5, 2.0, 5.0, 4.0]

    model = SVDPlusPlus(n_factors=8, n_epochs=10, lr=0.01, seed=42)
    model.fit(users, items, ratings)

    pred = model.predict(1, 10)
    assert isinstance(pred, float)
    assert 1.0 <= pred <= 6.0


def test_nmf_recommender():
    users = ["u1", "u1", "u2", "u2", "u3"]
    items = ["i1", "i2", "i2", "i3", "i1"]
    ratings = [5.0, 3.0, 4.0, 2.0, 5.0]

    model = NMFRecommender(n_factors=4, n_epochs=20, seed=42)
    model.fit(users, items, ratings)

    pred = model.predict("u1", "i1")
    assert isinstance(pred, float)
    assert pred >= 0.0

    recs = model.recommend("u1", top_k=2, filter_interacted=True)
    assert len(recs) <= 2


def test_implicit_als():
    users = ["alice", "alice", "bob", "bob", "charlie"]
    items = ["song1", "song2", "song2", "song3", "song1"]
    counts = [10.0, 5.0, 2.0, 15.0, 8.0]

    model = ImplicitALS(n_factors=8, n_epochs=5, alpha=10.0, seed=42)
    model.fit(users, items, counts)

    recs = model.recommend("alice", top_k=2)
    assert len(recs) == 2
    assert isinstance(recs[0][0], str)
    assert isinstance(recs[0][1], float)


def test_bayesian_personalized_ranking():
    users = ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"]
    items = ["i1", "i2", "i2", "i3", "i1", "i4", "i3", "i4"]

    bpr = BayesianPersonalizedRanking(n_factors=8, n_epochs=25, lr=0.05, seed=42)
    bpr.fit(users, items)

    score = bpr.predict_score("u1", "i1")
    assert isinstance(score, float)

    recs = bpr.recommend("u1", top_k=2, filter_interacted=True)
    assert len(recs) <= 2
    for itm, s in recs:
        assert itm not in ["i1", "i2"]


def test_neural_collaborative_filtering():
    users = ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"]
    items = ["i1", "i2", "i2", "i3", "i1", "i4", "i3", "i4"]

    ncf = NeuralCollaborativeFiltering(
        n_factors_gmf=8,
        n_factors_mlp=8,
        mlp_layers=[16, 8],
        lr=0.01,
        n_epochs=5,
        batch_size=4,
        seed=42,
    )
    ncf.fit(users, items)

    prob = ncf.predict_proba("u1", "i1")
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0

    recs = ncf.recommend("u1", top_k=2, filter_interacted=True)
    assert len(recs) <= 2


def test_wide_and_deep():
    rng = np.random.RandomState(42)
    N = 100
    X_dense = rng.randn(N, 4).astype(np.float32)
    X_sparse = rng.randint(0, 5, size=(N, 3)).astype(np.int32)
    y = rng.randint(0, 2, size=N).astype(np.int32)

    model = WideAndDeep(
        sparse_cardinalities=[5, 5, 5],
        embedding_dim=8,
        deep_hidden_units=[16, 8],
        lr=0.01,
        n_epochs=5,
        batch_size=16,
        seed=42,
    )
    model.fit(X_dense, X_sparse, y)

    probs = model.predict_proba(X_dense[:5], X_sparse[:5])
    assert probs.shape == (5,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    preds = model.predict(X_dense[:5], X_sparse[:5])
    assert preds.shape == (5,)
    assert set(np.unique(preds)).issubset({0, 1})


def test_deepfm():
    rng = np.random.RandomState(42)
    N = 80
    X_sparse = np.column_stack(
        [
            rng.randint(0, 10, size=N),
            rng.randint(0, 8, size=N),
            rng.randint(0, 6, size=N),
        ]
    ).astype(np.int32)
    y = rng.randint(0, 2, size=N).astype(np.int32)

    model = DeepFM(
        field_cardinalities=[10, 8, 6],
        embedding_dim=8,
        mlp_layers=[16, 8],
        lr=0.01,
        n_epochs=5,
        batch_size=16,
        seed=42,
    )
    model.fit(X_sparse, y)

    probs = model.predict_proba(X_sparse[:10])
    assert probs.shape == (10,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    preds = model.predict(X_sparse[:10])
    assert preds.shape == (10,)


def test_dlrm():
    rng = np.random.RandomState(42)
    N = 80
    X_dense = rng.randn(N, 5).astype(np.float32)
    X_sparse = np.column_stack(
        [
            rng.randint(0, 10, size=N),
            rng.randint(0, 8, size=N),
        ]
    ).astype(np.int32)
    y = rng.randint(0, 2, size=N).astype(np.int32)

    model = DLRM(
        embedding_dim=8,
        sparse_cardinalities=[10, 8],
        bottom_mlp_units=[16, 8],
        top_mlp_units=[16, 8],
        lr=0.01,
        n_epochs=5,
        batch_size=16,
        seed=42,
    )
    model.fit(X_dense, X_sparse, y)

    probs = model.predict_proba(X_dense[:8], X_sparse[:8])
    assert probs.shape == (8,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_sasrec_sequential():
    sessions = [
        ["item_A", "item_B", "item_C", "item_D"],
        ["item_B", "item_C", "item_D"],
        ["item_A", "item_C", "item_D"],
        ["item_A", "item_B", "item_D"],
    ]

    sasrec = SASRec(
        max_len=5, hidden_dim=16, n_heads=2, n_layers=1, n_epochs=5, seed=42
    )
    sasrec.fit(sessions)

    recs = sasrec.predict_next(["item_A", "item_B"], top_k=2)
    assert len(recs) == 2
    assert isinstance(recs[0][0], str)
    assert isinstance(recs[0][1], float)


def test_gru4rec_sequential():
    sessions = [
        ["click_1", "click_2", "click_3"],
        ["click_2", "click_3", "click_4"],
        ["click_1", "click_3", "click_4"],
    ]

    gru = GRU4Rec(hidden_dim=16, n_epochs=5, seed=42)
    gru.fit(sessions)

    recs = gru.predict_next(["click_1", "click_2"], top_k=2)
    assert len(recs) == 2


def test_recommendation_metrics():
    actual = ["item_A", "item_B"]
    predicted = ["item_C", "item_A", "item_D", "item_B", "item_E"]

    hr = hit_rate_at_k(actual, predicted, k=3)
    assert hr == 1.0

    hr_miss = hit_rate_at_k("item_Z", predicted, k=3)
    assert hr_miss == 0.0

    ndcg = ndcg_at_k(actual, predicted, k=5)
    assert 0.0 < ndcg <= 1.0

    mrr = mrr_at_k("item_A", predicted, k=5)
    assert mrr == 0.5

    p5 = precision_at_k(actual, predicted, k=5)
    assert p5 == 2.0 / 5.0

    r5 = recall_at_k(actual, predicted, k=5)
    assert r5 == 2.0 / 2.0

    map_score = map_at_k([actual, ["item_X"]], [predicted, ["item_X", "item_Y"]], k=3)
    assert 0.0 < map_score <= 1.0
