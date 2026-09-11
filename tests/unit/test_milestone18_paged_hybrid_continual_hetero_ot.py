import numpy as np

from chokkhu.models.nlp.paged_attention import (
    PagedKVCache,
    PagedAttention,
    ContinuousBatcher,
)
from chokkhu.models.nlp.mamba2 import Mamba2SSD, JambaHybridBlock
from chokkhu.continual.ewc import ElasticWeightConsolidation, EWC
from chokkhu.continual.replay import DarkExperienceReplay, DERPlusPlus
from chokkhu.models.gnn.hetero import HeteroGCN, SpatioTemporalGCN
from chokkhu.optimal_transport.sinkhorn import (
    SinkhornOptimalTransport,
    sinkhorn_distance,
)
from chokkhu.optimal_transport.barycenter import (
    WassersteinBarycenter,
    wasserstein_barycenter,
)
from chokkhu.transformation.symbolic_features import (
    SymbolicFeatureSynthesizer,
)


# ==========================================
# 1. PagedAttention & KV-Cache Serving Tests
# ==========================================


def test_paged_kv_cache_allocation_and_read():
    num_blocks = 16
    block_size = 4
    num_heads = 2
    head_dim = 8

    cache = PagedKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    assert len(cache.free_blocks) == num_blocks

    seq_id = "seq_1"
    cache.allocate_sequence(seq_id, num_tokens=10)
    assert len(cache.block_tables[seq_id]) == 3  # 10 tokens need 3 blocks of 4
    assert len(cache.free_blocks) == num_blocks - 3

    # Write synthetic K, V to slots
    keys = np.random.randn(10, num_heads, head_dim).astype(np.float32)
    values = np.random.randn(10, num_heads, head_dim).astype(np.float32)

    cache.seq_lengths[seq_id] = 0  # reset for token-by-token append
    for i in range(10):
        cache.append_token(seq_id, keys[i], values[i])

    k_read, v_read = cache.get_kv(seq_id)
    assert k_read.shape == (10, num_heads, head_dim)
    assert v_read.shape == (10, num_heads, head_dim)
    np.testing.assert_allclose(k_read, keys, atol=1e-6)
    np.testing.assert_allclose(v_read, values, atol=1e-6)

    usage = cache.memory_usage()
    assert usage["used_blocks"] == 3
    assert usage["active_sequences"] == 1

    cache.free_sequence(seq_id)
    assert len(cache.free_blocks) == num_blocks
    assert seq_id not in cache.block_tables


def test_paged_attention_forward():
    num_blocks = 32
    block_size = 4
    num_heads = 4
    head_dim = 16

    cache = PagedKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    attn = PagedAttention()

    seq_id = "test_seq"
    seq_len = 12
    cache.allocate_sequence(seq_id, num_tokens=0)

    keys = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
    values = np.random.randn(seq_len, num_heads, head_dim).astype(np.float32)
    for i in range(seq_len):
        cache.append_token(seq_id, keys[i], values[i])

    # Decode step: query token of shape (num_heads, head_dim)
    query = np.random.randn(num_heads, head_dim).astype(np.float32)
    out = attn.forward(query, seq_id, cache)
    assert out.shape == (num_heads, head_dim)
    assert not np.isnan(out).any()


def test_continuous_batcher_mock():
    class DummyTokenizer:
        def encode(self, text, bos=True):
            return [1, 2, 3, 4]

        def decode(self, tokens, skip_special_tokens=True):
            return "dummy text"

    class DummyModel:
        def __call__(self, x):
            # Returns logits (B, L, vocab_size)
            B, L = x.data.shape if hasattr(x, "data") else x.shape
            return np.random.randn(B, L, 50)

    batcher = ContinuousBatcher(
        model=DummyModel(),
        tokenizer=DummyTokenizer(),
        max_batch_size=4,
    )

    req_1 = batcher.add_request("prompt 1", max_new_tokens=3)
    req_2 = batcher.add_request("prompt 2", max_new_tokens=4)

    assert len(batcher.waiting_queue) == 2

    # Step 1: admits both requests
    status = batcher.step()
    assert status["active"] == 2
    assert status["waiting"] == 0

    # Step till all finished
    results = batcher.generate_all()
    assert req_1 in results
    assert req_2 in results
    assert results[req_1] == "dummy text"


# ==========================================
# 2. Mamba-2 SSD & Jamba Hybrid Block Tests
# ==========================================


def test_mamba2_ssd_forward():
    d_model = 32
    d_state = 16
    d_conv = 4
    expand = 2

    mamba2 = Mamba2SSD(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        seed=42,
    )

    batch_size = 2
    seq_len = 8
    x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)

    out = mamba2.forward(x)
    assert out.shape == (batch_size, seq_len, d_model)
    assert not np.isnan(out).any()


def test_jamba_hybrid_block():
    d_model = 32
    num_heads = 4
    num_experts = 4
    top_k = 2

    jamba = JambaHybridBlock(
        d_model=d_model,
        num_heads=num_heads,
        num_experts=num_experts,
        top_k=top_k,
        seed=42,
    )

    batch_size = 2
    seq_len = 6
    x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)

    out = jamba.forward(x)
    assert out.shape == (batch_size, seq_len, d_model)
    assert not np.isnan(out).any()


# ==========================================
# 3. Continual Learning: EWC & DER++ Tests
# ==========================================


def test_elastic_weight_consolidation():
    class DummyModel:
        def __init__(self):
            self.weights = np.array([[1.0, -0.5], [0.5, 1.5]], dtype=np.float32)

        def predict(self, x):
            return np.dot(x, self.weights)

    model = DummyModel()
    X = np.random.randn(20, 2).astype(np.float32)
    y = np.random.randint(0, 2, size=20)

    ewc = ElasticWeightConsolidation(importance=100.0)
    ewc.register_task(model, X, y, num_samples=10)

    assert len(ewc.task_memories) == 1
    # Penalty on original model should be 0.0
    pen_orig = ewc.penalty_loss(model)
    assert np.isclose(pen_orig, 0.0, atol=1e-6)

    # Shift model weights
    model.weights += 0.5
    pen_shifted = ewc.penalty_loss(model)
    assert pen_shifted > 0.0

    total = ewc.total_loss(1.5, model)
    assert total > 1.5

    # Test alias
    ewc_alias = EWC(importance=50.0)
    assert isinstance(ewc_alias, ElasticWeightConsolidation)


def test_dark_experience_replay():
    der = DarkExperienceReplay(buffer_capacity=10, alpha=0.5, beta=0.5)

    # Add experience to replay buffer
    x_sample = np.random.randn(5, 8).astype(np.float32)
    y_sample = np.array([0, 1, 0, 1, 0])
    logits_sample = np.random.randn(5, 2).astype(np.float32)

    der.add_batch(x_sample, y_sample, logits_sample)
    assert len(der) == 5

    buf_x, buf_y, buf_logits = der.sample(batch_size=3)
    assert buf_x.shape == (3, 8)
    assert buf_y.shape == (3,)
    assert buf_logits.shape == (3, 2)

    # Distillation loss
    student_logits = np.random.randn(3, 2).astype(np.float32)
    d_loss = der.distillation_loss(student_logits, buf_logits)
    assert d_loss >= 0.0

    # Total loss
    total_loss = der.compute_loss(
        current_task_loss=1.0,
        student_replay_logits=student_logits,
        teacher_replay_logits=buf_logits,
        replay_task_loss=0.5,
    )
    assert total_loss >= 1.0
    assert not np.isnan(total_loss)

    # Test alias
    der_alias = DERPlusPlus(buffer_capacity=20)
    assert isinstance(der_alias, DarkExperienceReplay)


# ==========================================
# 4. Heterogeneous & Spatio-Temporal GNN Tests
# ==========================================


def test_hetero_gcn_forward():
    in_channels_dict = {"user": 16, "item": 8}
    out_channels = 32
    relations = [("user", "buys", "item"), ("item", "bought_by", "user")]

    hetero = HeteroGCN(
        in_channels_dict=in_channels_dict,
        out_channels=out_channels,
        relations=relations,
        seed=42,
    )

    x_dict = {
        "user": np.random.randn(5, 16).astype(np.float32),
        "item": np.random.randn(4, 8).astype(np.float32),
    }

    edge_index_dict = {
        ("user", "buys", "item"): np.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
        ("item", "bought_by", "user"): np.array([[0, 1, 2, 3], [0, 1, 2, 3]]),
    }

    out_dict = hetero.forward(x_dict, edge_index_dict)
    assert out_dict["user"].shape == (5, 32)
    assert out_dict["item"].shape == (4, 32)
    assert not np.isnan(out_dict["user"]).any()
    assert not np.isnan(out_dict["item"]).any()


def test_spatio_temporal_gcn_forward():
    in_channels = 3
    out_channels = 16
    temporal_kernel_size = 3
    num_nodes = 6

    st_gcn = SpatioTemporalGCN(
        in_channels=in_channels,
        out_channels=out_channels,
        temporal_kernel_size=temporal_kernel_size,
        seed=42,
    )

    batch_size = 2
    time_steps = 10
    # Input shape: (B, T, N, in_channels)
    x = np.random.randn(batch_size, time_steps, num_nodes, in_channels).astype(
        np.float32
    )
    adj = np.eye(num_nodes) + np.random.uniform(0, 1, size=(num_nodes, num_nodes))
    adj = (adj + adj.T) / 2.0

    out = st_gcn.forward(x, adj)
    # Output shape: (B, T_out, N, out_channels)
    assert out.shape[0] == batch_size
    assert out.shape[2] == num_nodes
    assert out.shape[3] == out_channels
    assert not np.isnan(out).any()


# ==========================================
# 5. Optimal Transport & Barycenter Tests
# ==========================================


def test_sinkhorn_optimal_transport():
    n_source = 5
    n_target = 6

    # Probability distributions
    a = np.ones(n_source) / n_source
    b = np.ones(n_target) / n_target

    # Cost matrix
    x_source = np.linspace(0, 1, n_source)[:, None]
    x_target = np.linspace(0, 1, n_target)[:, None]
    diff = x_source[:, np.newaxis, :] - x_target[np.newaxis, :, :]
    C = np.sum(diff**2, axis=-1)

    ot = SinkhornOptimalTransport(reg=0.1, max_iter=100)
    ot.fit(a=a, b=b, cost_matrix=C)

    assert ot.plan_ is not None
    assert ot.plan_.shape == (n_source, n_target)
    assert ot.distance_ >= 0.0
    # Marginal constraints
    np.testing.assert_allclose(np.sum(ot.plan_, axis=1), a, atol=1e-3)
    np.testing.assert_allclose(np.sum(ot.plan_, axis=0), b, atol=1e-3)

    # Functional helper
    X_s = np.random.randn(10, 2)
    X_t = np.random.randn(12, 2)
    dist = sinkhorn_distance(X_s, X_t, reg=0.1)
    assert dist >= 0.0
    assert not np.isnan(dist)


def test_wasserstein_barycenter():
    # 2 Gaussian-like distributions on 1D grid
    n = 20
    grid = np.linspace(0, 1, n)

    d1 = np.exp(-((grid - 0.2) ** 2) / 0.02)
    d1 /= np.sum(d1)
    d2 = np.exp(-((grid - 0.8) ** 2) / 0.02)
    d2 /= np.sum(d2)

    distributions = [d1, d2]
    weights = [0.5, 0.5]

    # Cost matrix
    C = (grid[:, None] - grid[None, :]) ** 2

    wb = WassersteinBarycenter(reg=0.05, max_iter=50)
    barycenter = wb.compute(distributions, C, weights=weights)

    assert barycenter.shape == (n,)
    assert np.isclose(np.sum(barycenter), 1.0, atol=1e-4)
    assert (barycenter >= 0).all()

    # Functional helper
    bary_helper = wasserstein_barycenter(distributions, C, weights=weights, reg=0.05)
    np.testing.assert_allclose(barycenter, bary_helper, atol=1e-5)


# ==========================================
# 6. Symbolic Feature Synthesis Tests
# ==========================================


def test_symbolic_feature_synthesizer():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 4)
    # Target function y = x0 * x1 + sin(x2)
    y = X[:, 0] * X[:, 1] + np.sin(X[:, 2])

    synthesizer = SymbolicFeatureSynthesizer(
        n_features=3,
        generations=3,
        population_size=20,
        tournament_size=3,
        random_state=42,
    )

    synthesizer.fit(X, y)
    assert len(synthesizer.best_programs_) == 3

    X_transformed = synthesizer.transform(X)
    assert X_transformed.shape == (30, 3)
    assert not np.isnan(X_transformed).any()

    # Test feature names
    names = synthesizer.get_feature_names()
    assert len(names) == 3

    # Test SymbolicProgram evaluate
    prog = synthesizer.best_programs_[0]
    out_feat = prog.evaluate(X)
    assert out_feat.shape == (30,)
    assert isinstance(prog.to_formula(), str)
