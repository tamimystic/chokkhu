"""Unit tests for Milestone 23: OptNet (Differentiable QP), Sinkhorn Divergence,
Support Vector Data Description (SVDD), Spectral Graph Clustering, and Differentiable ILP.
"""

from __future__ import annotations

import numpy as np

from chokkhu.optimization.differentiable_qp import OptNet
from chokkhu.optimal_transport.divergence import SinkhornDivergence
from chokkhu.models.anomaly.svdd import SupportVectorDataDescription
from chokkhu.clustering.spectral import SpectralGraphClusterer
from chokkhu.models.neurosymbolic.ilp import DifferentiableILP


# =====================================================================
# 1. OptNet: Primal-Dual Interior Point QP & Implicit Differentiation
# =====================================================================


def test_optnet_forward_solve():
    # Minimize 1/2 x^T Q x + p^T x s.t. G x <= h, A x = b
    # Let Q = 2 * I (2x2), p = [-2, -2]^T
    # Unconstrained min is at x = [1, 1]^T
    # Constraint: x_0 + x_1 <= 1 (G = [[1, 1]], h = [1]), x >= 0
    Q = 2.0 * np.eye(2)
    p = np.array([-2.0, -2.0])
    G = np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    h = np.array([1.0, 0.0, 0.0])

    qp = OptNet(max_iter=50, tol=1e-6)
    x_opt = qp.forward(Q=Q, p=p, G=G, h=h)

    assert x_opt.shape == (2,)
    # Optimal solution is [0.5, 0.5]
    np.testing.assert_allclose(x_opt, np.array([0.5, 0.5]), atol=1e-3)
    assert np.all(G @ x_opt <= h + 1e-4)


def test_optnet_backward_gradients():
    # Test implicit differentiation through KKT conditions
    Q = 2.0 * np.eye(2)
    p = np.array([-2.0, -2.0])
    G = np.array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    h = np.array([1.0, 0.0, 0.0])

    qp = OptNet(max_iter=50, tol=1e-6)
    x_opt = qp.forward(Q=Q, p=p, G=G, h=h)

    # Let loss L(x) = 1/2 ||x - x_target||^2
    x_target = np.array([1.0, 0.0])
    grad_output = x_opt - x_target

    grads = qp.backward(grad_output)

    assert "dQ" in grads
    assert "dp" in grads
    assert "dG" in grads
    assert "dh" in grads
    assert grads["dQ"].shape == Q.shape
    assert grads["dp"].shape == p.shape
    assert grads["dG"].shape == G.shape
    assert grads["dh"].shape == h.shape
    assert not np.isnan(grads["dp"]).any()
    assert not np.isnan(grads["dQ"]).any()


def test_optnet_equality_constraints():
    # Min 1/2 (x0^2 + x1^2) s.t. x0 + x1 = 2
    # Solution is [1, 1]
    Q = np.eye(2)
    p = np.zeros(2)
    A = np.array([[1.0, 1.0]])
    b = np.array([2.0])

    qp = OptNet()
    x_opt = qp.forward(Q=Q, p=p, A=A, b=b)

    np.testing.assert_allclose(x_opt, np.array([1.0, 1.0]), atol=1e-3)
    np.testing.assert_allclose(A @ x_opt, b, atol=1e-3)


# =====================================================================
# 2. Sinkhorn Divergence: Debiased Entropic Optimal Transport
# =====================================================================


def test_sinkhorn_divergence_properties():
    sd = SinkhornDivergence(epsilon=0.1, max_iter=200, tol=1e-6)

    np.random.seed(42)
    X = np.random.randn(15, 2)
    Y = np.random.randn(15, 2) + 1.5

    # 1. Identity: S_eps(X, X) == 0
    div_XX = sd(X, X)
    np.testing.assert_allclose(div_XX, 0.0, atol=1e-5)

    # 2. Positivity: S_eps(X, Y) >= 0
    div_XY = sd(X, Y)
    assert div_XY >= 0.0

    # 3. Symmetry: S_eps(X, Y) == S_eps(Y, X)
    div_YX = sd(Y, X)
    np.testing.assert_allclose(div_XY, div_YX, atol=1e-5)


def test_sinkhorn_divergence_weighted():
    np.random.seed(42)
    X = np.random.randn(10, 3)
    Y = np.random.randn(12, 3)

    a = np.ones(10) / 10.0
    b = np.ones(12) / 12.0

    sd = SinkhornDivergence(epsilon=0.2)
    div = sd.compute(X, Y, a=a, b=b)
    assert div >= 0.0


# =====================================================================
# 3. Support Vector Data Description (SVDD): Anomaly Detection
# =====================================================================


def test_svdd_fit_and_predict():
    np.random.seed(42)
    # Generate inliers: Gaussian blob around origin
    X_train = np.random.randn(100, 2) * 0.5

    svdd = SupportVectorDataDescription(C=0.1, kernel="rbf", gamma=0.5)
    svdd.fit(X_train)

    assert svdd.radius_sq_ > 0.0
    assert svdd.support_vectors_ is not None and len(svdd.support_vectors_) > 0

    # Predict inliers (should mostly be 0 = inlier)
    inliers = np.array([[0.0, 0.0], [0.1, -0.1], [-0.1, 0.1]])
    preds_in = svdd.predict(inliers)
    assert (preds_in == 0).all()

    # Predict extreme outliers (should be 1 = outlier)
    outliers = np.array([[5.0, 5.0], [-6.0, 4.0], [10.0, -10.0]])
    preds_out = svdd.predict(outliers)
    assert (preds_out == 1).all()


def test_svdd_decision_function():
    np.random.seed(42)
    X_train = np.random.randn(80, 3) * 0.4
    svdd = SupportVectorDataDescription(C=0.1, kernel="rbf", gamma=1.0)
    svdd.fit(X_train)

    center_point = np.zeros((1, 3))
    far_point = np.array([[8.0, 8.0, 8.0]])

    # Decision score is negative for inliers, positive for outliers
    score_center = svdd.decision_function(center_point)[0]
    score_far = svdd.decision_function(far_point)[0]

    assert score_center < score_far
    assert score_center < 0
    assert score_far > 0


# =====================================================================
# 4. SpectralGraphClusterer: Normalized Graph Laplacian Partitioning
# =====================================================================


def test_spectral_graph_clustering_two_clusters():
    np.random.seed(42)
    # Generate 2 well-separated clusters
    c1 = np.random.randn(30, 2) + np.array([-5.0, 0.0])
    c2 = np.random.randn(30, 2) + np.array([5.0, 0.0])
    X = np.vstack([c1, c2])
    true_labels = np.array([0] * 30 + [1] * 30)

    clusterer = SpectralGraphClusterer(
        n_clusters=2, affinity="rbf", gamma=0.1, laplacian_type="symmetric", seed=42
    )
    labels = clusterer.fit_predict(X)

    assert labels.shape == (60,)
    # Cluster assignment accuracy (up to label permutation)
    match1 = np.mean(labels == true_labels)
    match2 = np.mean(labels == (1 - true_labels))
    accuracy = max(match1, match2)
    assert accuracy >= 0.95


def test_spectral_precomputed_affinity():
    # 4 nodes: (0, 1) connected strongly, (2, 3) connected strongly
    A = np.array(
        [
            [1.0, 0.9, 0.05, 0.0],
            [0.9, 1.0, 0.0, 0.05],
            [0.05, 0.0, 1.0, 0.95],
            [0.0, 0.05, 0.95, 1.0],
        ]
    )

    clusterer = SpectralGraphClusterer(n_clusters=2, affinity="precomputed", seed=42)
    labels = clusterer.fit_predict(A)

    # 0 and 1 should be in the same cluster, 2 and 3 in the same cluster
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]

    ncut = clusterer.normalized_cut(A)
    assert ncut >= 0.0


# =====================================================================
# 5. DifferentiableILP: Continuous Deduction & Rule Induction
# =====================================================================


def test_differentiable_ilp_forward_deduction():
    predicates = ["parent", "ancestor"]
    constants = ["alice", "bob", "charlie"]

    dilp = DifferentiableILP(
        predicates=predicates, constants=constants, max_steps=2, learning_rate=0.1
    )

    v0 = np.zeros(dilp.num_atoms)
    # parent(alice, bob) = 1, parent(bob, charlie) = 1
    v0[dilp.atom_to_idx[("parent", "alice", "bob")]] = 1.0
    v0[dilp.atom_to_idx[("parent", "bob", "charlie")]] = 1.0

    v_final = dilp.forward_deduction(v0)
    assert v_final.shape == (dilp.num_atoms,)
    assert (v_final >= 0.0).all() and (v_final <= 1.0 + 1e-6).all()


def test_differentiable_ilp_training_and_query():
    predicates = ["parent", "ancestor"]
    constants = ["alice", "bob", "charlie"]

    dilp = DifferentiableILP(
        predicates=predicates, constants=constants, max_steps=3, learning_rate=0.2
    )

    bg_facts = [
        ("parent", "alice", "bob"),
        ("parent", "bob", "charlie"),
    ]
    pos_examples = [
        ("ancestor", "alice", "bob"),
        ("ancestor", "bob", "charlie"),
        ("ancestor", "alice", "charlie"),
    ]
    neg_examples = [
        ("ancestor", "charlie", "alice"),
        ("ancestor", "bob", "alice"),
    ]

    dilp.fit(bg_facts, pos_examples, neg_examples, epochs=10)

    conf = dilp.predict_atom(bg_facts, ("ancestor", "alice", "charlie"))
    assert isinstance(conf, float)
    assert 0.0 <= conf <= 1.0
