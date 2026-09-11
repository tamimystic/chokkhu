"""Unit tests for Frontier 4.21: Equivariant GNNs, Geometric Sheaves & Clifford Algebras."""

from __future__ import annotations

import numpy as np

from chokkhu.models.geometric.egnn import EnEquivariantGNN
from chokkhu.geometry.sheaf import (
    CellularSheaf,
    SheafNeuralNetwork,
)
from chokkhu.models.geometric.clifford import (
    CliffordMultivector,
    CliffordGANN,
    geometric_product,
)


def test_egnn_rotation_translation_equivariance():
    """Verify exact E(3) rotational and translational equivariance for 3D coordinates and scalar invariance."""
    rng = np.random.default_rng(42)
    N = 6
    in_dim = 4
    coord_dim = 3

    # Random node features and 3D positions
    H = rng.standard_normal((N, in_dim))
    X = rng.standard_normal((N, coord_dim))

    # Fully connected edge graph without self loops
    src, dst = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                src.append(i)
                dst.append(j)
    edges = np.array([src, dst], dtype=np.int64)

    egnn = EnEquivariantGNN(
        in_dim=in_dim,
        hidden_dim=16,
        out_dim=in_dim,
        coord_dim=coord_dim,
        num_layers=2,
        random_state=42,
    )

    # 1. Clean forward pass
    H_out, X_out = egnn.forward(H, X, edges)

    # 2. Construct random 3D rotation matrix Q in O(3) and translation t
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    t = rng.standard_normal((1, 3))

    X_transformed = X @ Q.T + t

    # 3. Forward pass on transformed coordinates
    H_trans_out, X_trans_out = egnn.forward(H, X_transformed, edges)

    # Invariant features must be identical
    np.testing.assert_allclose(H_trans_out, H_out, atol=1e-5)

    # Equivariant coordinates must equal R X_out + t
    expected_X_trans = X_out @ Q.T + t
    np.testing.assert_allclose(X_trans_out, expected_X_trans, atol=1e-5)


def test_egnn_arbitrary_spatial_dimensions():
    """Verify EGNN functions seamlessly in 2D and 4D spacetime."""
    edges = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
    H = np.ones((3, 2))

    # 2D space
    X_2d = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
    egnn_2d = EnEquivariantGNN(in_dim=2, hidden_dim=8, out_dim=2, coord_dim=2)
    h_2d, x_2d = egnn_2d.forward(H, X_2d, edges)
    assert x_2d.shape == (3, 2)
    assert h_2d.shape == (3, 2)

    # 4D space
    X_4d = np.ones((3, 4))
    egnn_4d = EnEquivariantGNN(in_dim=2, hidden_dim=8, out_dim=2, coord_dim=4)
    h_4d, x_4d = egnn_4d.forward(H, X_4d, edges)
    assert x_4d.shape == (3, 4)
    assert h_4d.shape == (3, 2)


def test_cellular_sheaf_laplacian_properties():
    """Verify Cellular Sheaf coboundary, Laplacian positive semi-definiteness and Dirichlet energy."""
    # 4 nodes, 4 directed edges forming a cycle
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64)
    sheaf = CellularSheaf(
        num_nodes=4,
        edges=edges,
        node_dim=2,
        edge_dim=2,
        random_state=42,
    )

    # 1. Coboundary delta shape: (E * d_e, N * d_v) = (4 * 2, 4 * 2) = (8, 8)
    delta = sheaf.compute_coboundary()
    assert delta.shape == (8, 8)

    # 2. Sheaf Laplacian Delta_F = delta^T delta
    L = sheaf.compute_laplacian()
    assert L.shape == (8, 8)
    np.testing.assert_allclose(L, L.T, atol=1e-6)

    # 3. Positive semi-definiteness: all eigenvalues >= -1e-10
    eigenvalues = np.linalg.eigvalsh(L)
    assert np.all(eigenvalues >= -1e-10)

    # 4. Dirichlet energy: 1/2 x^T L x == 1/2 ||delta x||_2^2
    rng = np.random.default_rng(42)
    x = rng.standard_normal(8)
    energy = sheaf.dirichlet_energy(x)
    expected_energy = float(0.5 * np.dot(x, L @ x))
    np.testing.assert_allclose(energy, expected_energy, atol=1e-6)


def test_sheaf_neural_network_forward():
    """Verify SheafNeuralNetwork forward propagation over graph stalks."""
    edges = np.array([[0, 1, 2, 0], [1, 2, 0, 2]], dtype=np.int64)
    sheaf = CellularSheaf(num_nodes=3, edges=edges, node_dim=2, edge_dim=2)

    snn = SheafNeuralNetwork(
        node_dim=2,
        edge_dim=2,
        hidden_dim=16,
        out_dim=3,
        num_layers=2,
        alpha=0.3,
        random_state=42,
    )

    # Node features: (3, 1) or (3, 2, 1)
    X = np.array([[1.0], [2.0], [3.0]])
    out = snn.forward(X, sheaf)
    assert out.shape == (3, 3)
    assert not np.isnan(out).any()


def test_clifford_geometric_product_cayley_axioms():
    """Verify fundamental geometric algebra axioms and basis multiplication in Cl(3, 0, 0)."""
    # 1. Vector basis elements square to +1
    e1 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
    e2 = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float64)
    e3 = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float64)

    e1_sq = geometric_product(e1, e1)
    np.testing.assert_allclose(e1_sq, [1, 0, 0, 0, 0, 0, 0, 0])

    e2_sq = geometric_product(e2, e2)
    np.testing.assert_allclose(e2_sq, [1, 0, 0, 0, 0, 0, 0, 0])

    e3_sq = geometric_product(e3, e3)
    np.testing.assert_allclose(e3_sq, [1, 0, 0, 0, 0, 0, 0, 0])

    # 2. Anticommutation: e1 e2 = e12 = - e2 e1
    e12 = geometric_product(e1, e2)
    e21 = geometric_product(e2, e1)
    np.testing.assert_allclose(e12, [0, 0, 0, 0, 1, 0, 0, 0])
    np.testing.assert_allclose(e21, [0, 0, 0, 0, -1, 0, 0, 0])

    # 3. Bivectors square to -1: e12^2 = -1
    e12_sq = geometric_product(e12, e12)
    np.testing.assert_allclose(e12_sq, [-1, 0, 0, 0, 0, 0, 0, 0])

    # 4. Pseudoscalar squares to -1: e123^2 = -1
    e123 = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float64)
    e123_sq = geometric_product(e123, e123)
    np.testing.assert_allclose(e123_sq, [-1, 0, 0, 0, 0, 0, 0, 0])


def test_clifford_rotor_sandwich_rotation():
    """Verify 3D rotor R = exp(-theta/2 * B) accurately rotates 3D vector by 90 degrees around z-axis."""
    # Vector pointing along x-axis: v = [1, 0, 0]
    v = CliffordMultivector.from_vector(np.array([1.0, 0.0, 0.0]))

    # Rotor for 90 deg (pi/2) rotation around z-axis ([0, 0, 1])
    rotor_z = CliffordMultivector.rotor(
        axis=np.array([0.0, 0.0, 1.0]), angle_rad=np.pi / 2.0
    )

    # Sandwich product: v' = R * v * ~R
    v_rot = v.sandwich(rotor_z)

    # Expected rotated vector is along y-axis: [0, 1, 0]
    np.testing.assert_allclose(v_rot.vector, [0.0, 1.0, 0.0], atol=1e-5)
    # Scalar, bivector and pseudoscalar components must be zero
    np.testing.assert_allclose(v_rot.scalar, 0.0, atol=1e-5)
    np.testing.assert_allclose(v_rot.bivector, [0.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(v_rot.pseudoscalar, 0.0, atol=1e-5)


def test_clifford_gann_forward():
    """Verify Clifford Geometric Algebra Neural Network forward pass."""
    gann = CliffordGANN(
        in_channels=1,
        hidden_channels=8,
        out_channels=1,
        num_layers=2,
        random_state=42,
    )

    mv_data = np.random.randn(5, 1, 8)
    out = gann.forward(mv_data)
    assert out.shape == (5, 1, 8)
    assert not np.isnan(out).any()
