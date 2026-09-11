"""Unit tests for Milestone 20: The Grand Frontier.

Tests covering:
- DiscreteTextDiffusion
- RobotArmKinematics
- DifferentiableParticleFluid
- StatisticalTextWatermark
- RefusalDirectionProbe
- NOTEARSCausalDiscovery
- PCAlgorithm
- PoincareBallEmbedding
- LorentzManifold
"""

import numpy as np

from chokkhu import (
    DiscreteTextDiffusion,
    RobotArmKinematics,
    DifferentiableParticleFluid,
    StatisticalTextWatermark,
    RefusalDirectionProbe,
    NOTEARSCausalDiscovery,
    PCAlgorithm,
    PoincareBallEmbedding,
    LorentzManifold,
)


def test_discrete_text_diffusion():
    vocab_size = 50
    seq_len = 8
    batch_size = 2
    model = DiscreteTextDiffusion(
        vocab_size=vocab_size,
        max_seq_len=16,
        num_timesteps=10,
        d_model=32,
        num_heads=2,
        num_layers=1,
        schedule="linear",
        seed=42,
    )

    x_0 = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    t = np.array([2, 5])

    # 1. Forward categorical corruption
    x_t = model.q_sample(x_0, t)
    assert x_t.shape == (batch_size, seq_len)
    assert np.all(x_t >= 0) and np.all(x_t <= model.mask_token_id)

    # 2. Forward pass logits
    logits = model.forward(x_t, t)
    assert logits.shape == (batch_size, seq_len, vocab_size)

    # 3. Loss computation
    loss = model.compute_loss(x_0, t)
    assert isinstance(loss, float)
    assert loss > 0.0

    # 4. Sampling
    samples = model.sample(batch_size=2, seq_len=6, num_steps=5, temperature=1.0)
    assert samples.shape == (2, 6)
    assert not np.any(samples == model.mask_token_id)
    assert np.all(samples >= 0) and np.all(samples < vocab_size)


def test_robot_arm_kinematics_2d():
    # 2-link planar arm with link lengths 1.0 and 1.0
    arm = RobotArmKinematics.planar_2d(link_lengths=[1.0, 1.0])
    assert arm.num_joints == 2

    # Forward Kinematics at theta = [0, 0] -> should be at [2.0, 0.0, 0.0]
    angles = np.array([0.0, 0.0])
    T_end, transforms = arm.forward_kinematics(angles)
    pos = arm.get_end_effector_position(angles)
    assert np.allclose(pos, [2.0, 0.0, 0.0], atol=1e-5)
    assert len(transforms) == 3

    # Forward Kinematics at theta = [pi/2, 0] -> should be at [0.0, 2.0, 0.0]
    angles_90 = np.array([np.pi / 2, 0.0])
    pos_90 = arm.get_end_effector_position(angles_90)
    assert np.allclose(pos_90, [0.0, 2.0, 0.0], atol=1e-5)

    # Jacobian computation
    J = arm.compute_jacobian(angles)
    assert J.shape == (6, 2)
    # At [0, 0], y-velocity should respond to joint rotations
    assert np.abs(J[1, 0]) > 0.0

    # Inverse Kinematics: target [1.0, 1.0, 0.0]
    target = np.array([1.0, 1.0, 0.0])
    solved_angles, converged, iters = arm.inverse_kinematics(
        target_position=target,
        max_iters=100,
        tolerance=1e-3,
    )
    final_pos = arm.get_end_effector_position(solved_angles)
    assert np.linalg.norm(target - final_pos) < 1e-2


def test_robot_arm_puma560():
    puma = RobotArmKinematics.puma560()
    assert puma.num_joints == 6
    angles = np.zeros(6)
    T_end, transforms = puma.forward_kinematics(angles)
    assert T_end.shape == (4, 4)
    assert len(transforms) == 7
    J = puma.compute_jacobian(angles)
    assert J.shape == (6, 6)


def test_differentiable_particle_fluid():
    num_particles = 20
    fluid = DifferentiableParticleFluid(
        num_particles=num_particles,
        dim=2,
        rest_density=1000.0,
        stiffness=100.0,
        viscosity=0.05,
        smoothing_length=0.2,
        particle_mass=0.01,
        dt=0.002,
        seed=42,
    )

    pos_init = np.random.RandomState(42).uniform(-0.5, 0.5, size=(num_particles, 2))
    vel_init = np.zeros((num_particles, 2))

    # 1. Density computation
    densities = fluid.compute_density(pos_init)
    assert densities.shape == (num_particles,)
    assert np.all(densities > 0.0)

    # 2. Pressure computation
    pressures = fluid.compute_pressure(densities)
    assert pressures.shape == (num_particles,)
    assert np.all(pressures >= 0.0)

    # 3. Forces computation
    forces = fluid.compute_forces(pos_init, vel_init)
    assert forces.shape == (num_particles, 2)
    assert not np.any(np.isnan(forces))

    # 4. Simulation step
    pos_next, vel_next = fluid.step(pos_init, vel_init)
    assert pos_next.shape == (num_particles, 2)
    assert vel_next.shape == (num_particles, 2)

    # 5. Multi-step simulation
    pos_hist, vel_hist = fluid.simulate(pos_init, vel_init, num_steps=5)
    assert pos_hist.shape == (6, num_particles, 2)
    assert vel_hist.shape == (6, num_particles, 2)


def test_statistical_text_watermark():
    vocab_size = 200
    watermark = StatisticalTextWatermark(
        vocab_size=vocab_size,
        gamma=0.5,
        delta=5.0,
        hash_key=42,
        window_size=1,
    )

    # 1. Green list mask
    prefix = np.array([12])
    mask = watermark.get_greenlist_mask(prefix)
    assert mask.shape == (vocab_size,)
    assert np.sum(mask) == watermark.green_list_size

    # 2. Apply watermark bias
    logits = np.zeros(vocab_size)
    biased_logits = watermark.apply_watermark_bias(logits, prefix)
    assert np.allclose(biased_logits[mask], 5.0)
    assert np.allclose(biased_logits[~mask], 0.0)

    # 3. Generate heavily watermarked sequence vs unwatermarked sequence
    T = 60
    watermarked_seq = [10]
    rng = np.random.RandomState(42)
    for _ in range(T):
        curr_prefix = np.array(watermarked_seq)
        g_mask = watermark.get_greenlist_mask(curr_prefix)
        green_candidates = np.where(g_mask)[0]
        # Always choose a green token
        tok = rng.choice(green_candidates)
        watermarked_seq.append(tok)

    det_wm = watermark.detect(np.array(watermarked_seq))
    assert det_wm["is_watermarked"] is True
    assert det_wm["z_score"] > 3.0
    assert det_wm["p_value"] < 0.01

    # Random unwatermarked sequence
    unwatermarked_seq = rng.randint(0, vocab_size, size=T + 1)
    det_unwm = watermark.detect(unwatermarked_seq)
    assert det_unwm["is_watermarked"] is False


def test_refusal_direction_probe():
    dim = 16
    probe = RefusalDirectionProbe(dim=dim)

    # Synthetic harmful activations (aligned along positive dim 0)
    harmful = np.random.randn(30, dim) * 0.1
    harmful[:, 0] += 3.0

    # Harmless activations (aligned along negative dim 0)
    harmless = np.random.randn(30, dim) * 0.1
    harmless[:, 0] -= 3.0

    probe.fit(harmful, harmless)
    assert probe.direction is not None
    assert np.isclose(np.linalg.norm(probe.direction), 1.0)
    assert probe.direction[0] > 0.9  # Primary difference along dim 0

    # Test steering
    sample_act = np.zeros((2, dim))
    steered = probe.steer(sample_act, alpha=2.0)
    assert np.allclose(steered, 2.0 * probe.direction)

    # Test ablation
    ablated = probe.ablate(harmful)
    # Projection along refusal direction should now be ~0
    ab_scores = probe.score_refusal_intent(ablated)
    assert np.allclose(ab_scores, 0.0, atol=1e-5)


def test_notears_causal_discovery():
    # Synthetic DAG: X0 -> X1 -> X2 (3 variables)
    # W_true = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    rng = np.random.RandomState(42)
    n = 200
    x0 = rng.randn(n)
    x1 = 0.8 * x0 + 0.2 * rng.randn(n)
    x2 = 0.8 * x1 + 0.2 * rng.randn(n)
    X = np.column_stack([x0, x1, x2])

    notears = NOTEARSCausalDiscovery(lambda1=0.05, max_iter=30, w_threshold=0.2)
    notears.fit(X)

    assert notears.W_est_ is not None
    assert notears.adjacency_matrix_ is not None
    assert notears.is_dag() is True
    # Verify no self-loops
    assert np.all(np.diag(notears.W_est_) == 0.0)


def test_pc_algorithm():
    # Synthetic data: X0 -> X2 <- X1 (v-structure collider)
    rng = np.random.RandomState(42)
    n = 300
    x0 = rng.randn(n)
    x1 = rng.randn(n)
    x2 = 0.7 * x0 + 0.7 * x1 + 0.2 * rng.randn(n)
    X = np.column_stack([x0, x1, x2])

    pc = PCAlgorithm(alpha=0.05)
    pc.fit(X)

    assert pc.adjacency_matrix_ is not None
    assert pc.adjacency_matrix_.shape == (3, 3)
    edges = pc.get_edges()
    assert isinstance(edges, list)


def test_poincare_ball_embedding():
    dim = 3
    poincare = PoincareBallEmbedding(dim=dim, c=1.0, seed=42)

    u = np.array([0.2, 0.1, -0.3])
    v = np.array([-0.1, 0.4, 0.2])

    # 1. Distance properties
    d_uv = poincare.distance(u, v)
    d_vu = poincare.distance(v, u)
    assert np.isclose(d_uv, d_vu)
    assert np.isclose(poincare.distance(u, u), 0.0)

    # 2. Möbius addition
    mob = poincare.mobius_add(u, v)
    assert np.linalg.norm(mob) < 1.0

    # 3. Exp and Log map inverse
    origin = np.zeros(dim)
    tangent_v = np.array([0.1, -0.2, 0.15])
    pt = poincare.exp_map(origin, tangent_v)
    rec_v = poincare.log_map(origin, pt)
    assert np.allclose(tangent_v, rec_v, atol=1e-5)

    # 4. Graph embedding fitting
    edges = [(0, 1), (0, 2), (1, 3)]
    embs = poincare.fit_graph(edges=edges, num_nodes=4, num_epochs=10, lr=0.05)
    assert embs.shape == (4, dim)
    assert np.all(np.linalg.norm(embs, axis=-1) < 1.0)


def test_lorentz_manifold():
    dim = 2
    lorentz = LorentzManifold(dim=dim, c=1.0)

    # Point on Lorentz hyperboloid: x_0 = sqrt(1 + x_1^2 + x_2^2)
    x = np.array([np.sqrt(1.0 + 0.3**2 + 0.4**2), 0.3, 0.4])
    y = np.array([np.sqrt(1.0 + 0.1**2 + 0.2**2), 0.1, 0.2])

    # Minkowski inner product
    dot_xx = lorentz.minkowski_dot(x, x)
    assert np.isclose(dot_xx, -1.0)

    # Distance
    dist = lorentz.distance(x, y)
    assert dist > 0.0
    assert np.isclose(lorentz.distance(x, x), 0.0, atol=1e-5)

    # Isomorphism: Poincare -> Lorentz -> Poincare
    u = np.array([0.2, -0.4])
    x_lor = lorentz.poincare_to_lorentz(u)
    assert np.isclose(lorentz.minkowski_dot(x_lor, x_lor), -1.0)

    u_rec = lorentz.lorentz_to_poincare(x_lor)
    assert np.allclose(u, u_rec, atol=1e-5)
