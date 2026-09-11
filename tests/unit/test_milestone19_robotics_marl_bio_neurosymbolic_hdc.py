import numpy as np

from chokkhu.models.robotics import (
    DiffusionPolicy,
    RecurrentWorldModel,
    MPPITrajectoryOptimizer,
)
from chokkhu.models.marl import (
    QMIX,
    VDN,
    NashEquilibriumSolver,
)
from chokkhu.models.bio import (
    GenomicTokenizer,
    GenomicBERT,
    ProteinContactMap,
)
from chokkhu.models.neurosymbolic import (
    DifferentiableLogicEngine,
    RotatE,
    TransE,
)
from chokkhu.hdc import (
    HyperdimensionalVector,
    HDCClassifier,
)


# ==========================================
# 1. Robotics & World Models Tests
# ==========================================


def test_diffusion_policy():
    policy = DiffusionPolicy(
        action_dim=4,
        pred_horizon=6,
        obs_dim=8,
        num_timesteps=15,
        hidden_dim=32,
        seed=42,
    )

    actions = np.random.randn(2, 6, 4).astype(np.float32)
    timesteps = np.array([5, 10], dtype=np.int32)
    obs = np.random.randn(2, 8).astype(np.float32)

    # Forward diffusion
    noisy, noise = policy.q_sample(actions, timesteps)
    assert noisy.shape == (2, 6, 4)

    # Predict noise
    pred_noise = policy.predict_noise(noisy, timesteps, obs)
    assert pred_noise.shape == (2, 6, 4)
    assert not np.isnan(pred_noise).any()

    # Sampling
    samples = policy.sample(obs, n_samples=2)
    assert samples.shape == (2, 6, 4)
    assert not np.isnan(samples).any()


def test_recurrent_world_model():
    world_model = RecurrentWorldModel(
        obs_dim=8,
        action_dim=2,
        deter_dim=16,
        stoch_dim=8,
        seed=42,
    )

    h0, z0 = world_model.initial_state(batch_size=2)
    assert h0.shape == (2, 16)
    assert z0.shape == (2, 8)

    action = np.random.randn(2, 2).astype(np.float32)
    obs = np.random.randn(2, 8).astype(np.float32)

    # Prior step
    h1_prior, z1_prior, p_mean, p_std = world_model.step_prior(h0, z0, action)
    assert h1_prior.shape == (2, 16)
    assert z1_prior.shape == (2, 8)

    # Posterior step
    h1_post, z1_post, q_mean, q_std = world_model.step_posterior(h0, z0, action, obs)
    assert h1_post.shape == (2, 16)
    assert z1_post.shape == (2, 8)

    # Decoders
    obs_rec = world_model.decode_observation(h1_post, z1_post)
    reward_pred = world_model.predict_reward(h1_post, z1_post)
    assert obs_rec.shape == (2, 8)
    assert reward_pred.shape == (2, 1)

    # Imagination trajectory rollouts
    def dummy_policy(h, z):
        return np.random.randn(len(h), 2).astype(np.float32)

    rollouts = world_model.imagine_trajectory(h0, z0, dummy_policy, horizon=5)
    assert rollouts["h"].shape == (2, 5, 16)
    assert rollouts["actions"].shape == (2, 5, 2)
    assert rollouts["rewards"].shape == (2, 5, 1)


def test_mppi_trajectory_optimizer():
    optimizer = MPPITrajectoryOptimizer(
        action_dim=2,
        horizon=8,
        num_samples=20,
        temperature=1.0,
        noise_sigma=0.3,
        seed=42,
    )

    def dummy_cost(state, actions_batch):
        # Quadratic target tracking cost
        target = np.array([1.0, -1.0])
        return np.mean((actions_batch - target) ** 2, axis=(1, 2))

    init_state = np.zeros(4)
    best_action, planned_traj = optimizer.optimize(init_state, dummy_cost)
    assert best_action.shape == (2,)
    assert planned_traj.shape == (8, 2)
    assert not np.isnan(best_action).any()


# ==========================================
# 2. Multi-Agent Reinforcement Learning Tests
# ==========================================


def test_qmix_forward_and_loss():
    qmix = QMIX(n_agents=3, state_dim=12, mixing_embed_dim=16, seed=42)

    agent_qs = np.random.randn(4, 3).astype(np.float32)
    states = np.random.randn(4, 12).astype(np.float32)

    q_tot = qmix.forward(agent_qs, states)
    assert q_tot.shape == (4, 1)
    assert not np.isnan(q_tot).any()

    # TD loss
    next_qs = np.random.randn(4, 3).astype(np.float32)
    next_states = np.random.randn(4, 12).astype(np.float32)
    rewards = np.array([1.0, 0.5, -0.2, 2.0])
    dones = np.array([0, 0, 1, 0])

    loss = qmix.td_loss(agent_qs, states, rewards, next_qs, next_states, dones)
    assert loss >= 0.0
    assert not np.isnan(loss)


def test_vdn_forward_and_loss():
    vdn = VDN(n_agents=3)

    agent_qs = np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 1.0]], dtype=np.float32)
    q_tot = vdn.forward(agent_qs)
    np.testing.assert_allclose(q_tot, np.array([[6.0], [1.0]], dtype=np.float32))

    rewards = np.array([1.0, 0.0])
    next_qs = np.array([[1.5, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    dones = np.array([0, 1])

    loss = vdn.td_loss(agent_qs, rewards, next_qs, dones)
    assert loss >= 0.0


def test_nash_equilibrium_solver():
    solver = NashEquilibriumSolver(max_iter=500)

    # Rock-Paper-Scissors (Zero-Sum Game)
    rps = np.array(
        [
            [0.0, -1.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 1.0, 0.0],
        ]
    )

    p_row, p_col, val = solver.solve_zero_sum(rps)
    assert len(p_row) == 3
    assert len(p_col) == 3
    np.testing.assert_allclose(np.sum(p_row), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.sum(p_col), 1.0, atol=1e-5)
    np.testing.assert_allclose(p_row, [1 / 3, 1 / 3, 1 / 3], atol=0.15)
    assert np.isclose(val, 0.0, atol=0.1)

    # General-sum bi-matrix Prisoner's Dilemma
    A = np.array([[3.0, 0.0], [5.0, 1.0]])
    B = np.array([[3.0, 5.0], [0.0, 1.0]])
    p_a, p_b, payoffs = solver.solve_bimatrix(A, B)
    assert len(p_a) == 2
    assert len(p_b) == 2


# ==========================================
# 3. Genomic & Bio-Molecular AI Tests
# ==========================================


def test_genomic_tokenizer():
    tok = GenomicTokenizer(k=3, stride=1, is_rna=False)
    assert tok.vocab_size == 4**3 + 5

    seq = "ATGCGATCG"
    kmers = tok.tokenize(seq)
    assert len(kmers) == len(seq) - 3 + 1
    assert kmers[0] == "ATG"

    tokens = tok.encode(seq, add_special_tokens=True)
    assert tokens[0] == tok.vocab["[CLS]"]
    assert tokens[-1] == tok.vocab["[SEP]"]

    decoded = tok.decode(tokens)
    assert "ATG" in decoded

    rev_comp = GenomicTokenizer.reverse_complement("ATGC")
    assert rev_comp == "GCAT"


def test_genomic_bert():
    bert = GenomicBERT(
        vocab_size=100,
        d_model=32,
        num_heads=4,
        num_layers=2,
        max_len=32,
        num_classes=2,
        seed=42,
    )

    tokens = np.random.randint(0, 100, size=(2, 16))
    hidden, cls_logits, mlm_logits = bert.forward(tokens)

    assert hidden.shape == (2, 16, 32)
    assert cls_logits.shape == (2, 2)
    assert mlm_logits.shape == (2, 16, 100)

    # Variant effect scoring
    ref_tokens = np.random.randint(0, 100, size=(1, 10))
    alt_tokens = ref_tokens.copy()
    alt_tokens[0, 5] = (alt_tokens[0, 5] + 1) % 100

    score = bert.score_variant(ref_tokens, alt_tokens)
    assert isinstance(score, float)
    assert not np.isnan(score)


def test_protein_contact_map():
    dca = ProteinContactMap(pseudocount_weight=0.5, apc=True)

    # Synthetic MSA with N=30 sequences of length L=12
    rng = np.random.default_rng(42)
    msa = rng.integers(0, 20, size=(30, 12))

    contact_map = dca.compute_contact_map(msa)
    assert contact_map.shape == (12, 12)
    assert np.allclose(contact_map, contact_map.T)
    assert np.all(np.diag(contact_map) == 0.0)
    assert not np.isnan(contact_map).any()


# ==========================================
# 4. Neuro-Symbolic Logic & KG Embedding Tests
# ==========================================


def test_differentiable_logic_engine():
    engine_prod = DifferentiableLogicEngine(t_norm="product")
    engine_godel = DifferentiableLogicEngine(t_norm="godel")
    engine_luka = DifferentiableLogicEngine(t_norm="lukasiewicz")

    a = np.array([0.8, 0.4])
    b = np.array([0.5, 0.9])

    # Product
    assert np.allclose(engine_prod.conjunction(a, b), [0.4, 0.36])
    assert np.allclose(engine_prod.negation(a), [0.2, 0.6])

    # Godel
    assert np.allclose(engine_godel.conjunction(a, b), [0.5, 0.4])
    assert np.allclose(engine_godel.disjunction(a, b), [0.8, 0.9])

    # Lukasiewicz
    assert np.allclose(engine_luka.conjunction(a, b), [0.3, 0.3])

    # Quantifiers & Loss
    vals = np.array([[0.9, 0.7, 0.8], [0.4, 0.2, 0.6]])
    assert np.allclose(engine_prod.universal_quantifier(vals, axis=-1), [0.7, 0.2])
    assert np.allclose(engine_prod.existential_quantifier(vals, axis=-1), [0.9, 0.6])

    loss = engine_prod.satisfaction_loss(np.array([0.9, 0.8]))
    assert np.isclose(loss, 0.15, atol=1e-5)


def test_rotate_and_transe():
    # RotatE
    rotate = RotatE(
        num_entities=15, num_relations=4, embedding_dim=16, gamma=10.0, seed=42
    )
    h = np.array([0, 1, 2])
    r = np.array([0, 1, 0])
    t = np.array([1, 2, 0])

    rot_scores = rotate.score_triplets(h, r, t)
    assert rot_scores.shape == (3,)
    assert not np.isnan(rot_scores).any()

    # TransE
    transe = TransE(
        num_entities=15, num_relations=4, embedding_dim=16, margin=1.0, seed=42
    )
    trans_scores = transe.score_triplets(h, r, t)
    assert trans_scores.shape == (3,)

    loss = transe.margin_loss(
        pos_heads=h,
        pos_rels=r,
        pos_tails=t,
        neg_heads=h,
        neg_rels=r,
        neg_tails=np.array([2, 0, 1]),
    )
    assert loss >= 0.0


# ==========================================
# 5. Hyperdimensional Computing (HDC/VSA) Tests
# ==========================================


def test_hypervector_operations():
    dim = 2000
    v1 = HyperdimensionalVector.random_bipolar(dim=dim, seed=42)
    v2 = HyperdimensionalVector.random_bipolar(dim=dim, seed=43)

    # Quasi-orthogonality of random hypervectors
    sim_rand = v1.similarity(v2)
    assert abs(sim_rand) < 0.1  # Highly orthogonal in high dimensions

    # Binding preserves orthogonality to inputs
    bound = v1.bind(v2)
    assert abs(bound.similarity(v1)) < 0.1
    assert abs(bound.similarity(v2)) < 0.1

    # Unbinding (self-inverse for bipolar: (A * B) * B = A)
    unbound = bound.bind(v2)
    assert np.isclose(unbound.similarity(v1), 1.0, atol=1e-5)

    # Bundling has high similarity to constituent vectors
    bundle = HyperdimensionalVector.bundle([v1, v2], binarize=True)
    assert bundle.similarity(v1) > 0.4
    assert bundle.similarity(v2) > 0.4

    # Permutation
    perm = v1.permute(shift=3)
    assert abs(perm.similarity(v1)) < 0.1


def test_hdc_classifier():
    rng = np.random.default_rng(42)
    # Synthetic classification dataset
    X_train = rng.normal(loc=0.0, scale=1.0, size=(20, 4))
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

    hdc = HDCClassifier(dim=2000, num_levels=16, seed=42)
    hdc.fit(X_train, y_train)

    preds = hdc.predict(X_train)
    assert len(preds) == 20
    acc = hdc.score(X_train, y_train)
    assert acc >= 0.7  # HDC reaches high training accuracy
