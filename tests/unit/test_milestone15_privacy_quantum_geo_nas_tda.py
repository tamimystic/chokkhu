"""Unit test suite for Milestone 15: Privacy, Quantum ML, Geospatial, NAS, and TDA."""

import numpy as np

from chokkhu.privacy import (
    LaplaceMechanism,
    GaussianMechanism,
    DP_SGD,
    FederatedClient,
    FederatedServer,
)
from chokkhu.quantum import (
    QuantumCircuit,
    VariationalQuantumClassifier,
    QuantumKernel,
)
from chokkhu.geospatial import (
    SpatialWeights,
    morans_i,
    local_morans_i,
    SpatialAutoregression,
    GeographicallyWeightedRegression,
    OrdinaryKriging,
)
from chokkhu.automl import DARTS
from chokkhu.tda import (
    VietorisRipsComplex,
    PersistenceDiagram,
    PersistenceLandscape,
    bottleneck_distance,
)


def test_differential_privacy_mechanisms():
    """Test Laplace and Gaussian DP mechanisms and DP-SGD optimizer."""
    # 1. Laplace Mechanism
    lap = LaplaceMechanism(epsilon=1.0, sensitivity=0.5, random_state=42)
    val = 10.0
    perturbed_lap = lap.perturb(val)
    assert isinstance(perturbed_lap, float)
    assert abs(perturbed_lap - val) < 5.0  # reasonable noise bound
    assert lap.get_privacy_budget() == (1.0, 0.0)

    # 2. Gaussian Mechanism
    gauss = GaussianMechanism(epsilon=1.0, delta=1e-5, sensitivity=0.5, random_state=42)
    perturbed_gauss = gauss.perturb(val)
    assert isinstance(perturbed_gauss, float)
    assert gauss.get_privacy_budget() == (1.0, 1e-5)

    # 3. DP-SGD Optimizer
    optimizer = DP_SGD(lr=0.01, l2_norm_clip=1.0, noise_multiplier=0.5, random_state=42)
    params = {"W": np.ones((5, 2)), "b": np.zeros(2)}

    # Generate 4 mock per-sample gradients
    per_sample_grads = [
        {"W": np.random.randn(5, 2) * 2.0, "b": np.random.randn(2)} for _ in range(4)
    ]
    updated_params = optimizer.step(params, per_sample_grads)
    assert updated_params["W"].shape == (5, 2)

    # Privacy spent accounting
    spent = optimizer.compute_privacy_spent(
        total_samples=100, batch_size=4, target_delta=1e-5
    )
    assert "epsilon" in spent
    assert spent["epsilon"] > 0.0


def test_federated_learning_fedavg_and_fedprox():
    """Test FederatedServer and FederatedClient with FedAvg and FedProx."""
    np.random.seed(42)

    # Generate heterogeneous synthetic data for 3 clients
    client1_X = np.random.randn(20, 3)
    client1_y = client1_X[:, 0] * 2.0 + client1_X[:, 1] * -1.5 + 0.5

    client2_X = np.random.randn(25, 3)
    client2_y = client2_X[:, 0] * 1.8 + client2_X[:, 1] * -1.2 + 0.4

    c1 = FederatedClient(
        "client_1", client1_X, client1_y, mu=0.0, lr=0.05, random_state=42
    )
    c2 = FederatedClient(
        "client_2", client2_X, client2_y, mu=0.1, lr=0.05, random_state=42
    )  # FedProx

    initial_weights = {"W": np.zeros((3, 1)), "b": np.zeros((1,))}
    server = FederatedServer(
        initial_weights=initial_weights,
        clients=[c1, c2],
        strategy="fedavg",
        random_state=42,
    )

    # Run 3 federated rounds
    for _ in range(3):
        res = server.train_round(fraction_fit=1.0, local_epochs=3, batch_size=8)
        assert res["participating_clients"] == 2

    global_w = server.get_weights()
    assert global_w["W"].shape == (3, 1)

    preds = server.predict(np.random.randn(5, 3))
    assert preds.shape == (5, 1)


def test_quantum_circuit_simulation():
    """Test QuantumCircuit state vector evolution, gates, and Bell state."""
    # 1. Single qubit rotations and Hadamard
    qc = QuantumCircuit(n_qubits=1)
    qc.h(0)
    probs = qc.probabilities()
    np.testing.assert_allclose(probs, [0.5, 0.5], atol=1e-5)

    # Pauli gates
    qc.reset()
    qc.x(0)
    np.testing.assert_allclose(qc.probabilities(), [0.0, 1.0], atol=1e-5)

    # 2. Bell State Creation: |Phi+> = (|00> + |11>) / sqrt(2)
    qc2 = QuantumCircuit(n_qubits=2)
    qc2.h(0).cnot(0, 1)
    probs2 = qc2.probabilities()
    np.testing.assert_allclose(probs2, [0.5, 0.0, 0.0, 0.5], atol=1e-5)
    assert abs(qc2.expectation_z(0)) < 1e-5


def test_variational_quantum_classifier_and_kernel():
    """Test VariationalQuantumClassifier with Parameter Shift Rule and QuantumKernel."""
    np.random.seed(42)
    X = np.random.uniform(-1.0, 1.0, size=(16, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # 1. VQC
    vqc = VariationalQuantumClassifier(n_qubits=2, n_layers=1, lr=0.1, random_state=42)
    vqc.fit(X, y, epochs=5, batch_size=8)
    preds = vqc.predict(X[:4])
    probs = vqc.predict_proba(X[:4])
    assert len(preds) == 4
    assert probs.shape == (4, 2)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

    # 2. Quantum Kernel
    qk = QuantumKernel(n_qubits=2)
    sim = qk.evaluate(X[0], X[0])
    assert abs(sim - 1.0) < 1e-5  # self similarity is 1.0

    K = qk.compute_matrix(X[:5])
    assert K.shape == (5, 5)
    np.testing.assert_allclose(K, K.T, atol=1e-5)  # symmetric Gram matrix


def test_geospatial_statistics_and_sar():
    """Test SpatialWeights, Moran's I autocorrelation, and Spatial Autoregression."""
    np.random.seed(42)
    coords = np.random.uniform(0, 10, size=(20, 2))

    # 1. Spatial Weights
    sw = SpatialWeights(coords, method="knn", k=4, row_standardize=True)
    W = sw.to_matrix()
    assert W.shape == (20, 20)
    np.testing.assert_allclose(np.sum(W, axis=1), 1.0, atol=1e-5)

    # 2. Moran's I
    values = coords[:, 0] * 2.0 + np.random.randn(20) * 0.1  # spatially correlated
    m_res = morans_i(values, sw, permutations=49, random_state=42)
    assert "I" in m_res
    assert "p_value" in m_res

    local_i = local_morans_i(values, sw)
    assert len(local_i) == 20

    # 3. Spatial Autoregression (Lag Model)
    X = np.random.randn(20, 2)
    y = 0.5 * np.dot(W, values) + np.dot(X, [1.0, -1.0]) + np.random.randn(20) * 0.1
    sar = SpatialAutoregression(model_type="lag")
    sar.fit(X, y, sw)
    sar_preds = sar.predict(X[:5])
    assert len(sar_preds) == 5


def test_gwr_and_ordinary_kriging():
    """Test Geographically Weighted Regression (GWR) and Ordinary Kriging."""
    np.random.seed(42)
    coords = np.random.uniform(0, 10, size=(15, 2))
    X = np.random.randn(15, 2)
    y = coords[:, 0] * 0.5 + X[:, 0] * 2.0 + np.random.randn(15) * 0.1

    # 1. GWR
    gwr = GeographicallyWeightedRegression(bandwidth=3.0, kernel="gaussian")
    gwr.fit(coords, X, y)
    preds = gwr.predict(coords[:3], X[:3])
    assert len(preds) == 3

    # 2. Ordinary Kriging
    z = np.sin(coords[:, 0]) + np.cos(coords[:, 1])
    krig = OrdinaryKriging(
        variogram_model="spherical", nugget=0.05, sill=1.0, range_val=5.0
    )
    krig.fit(coords, z)
    z_interp, var_est = krig.predict(np.array([[5.0, 5.0], [2.0, 8.0]]))
    assert len(z_interp) == 2
    assert len(var_est) == 2
    assert np.all(var_est >= 0.0)


def test_darts_neural_architecture_search():
    """Test DARTS Differentiable Neural Architecture Search."""
    np.random.seed(42)
    X = np.random.randn(30, 4)
    y = (X[:, 0] > 0).astype(int)

    darts = DARTS(
        input_dim=4,
        num_classes=2,
        num_intermediate_nodes=2,
        hidden_dim=16,
        random_state=42,
    )
    darts.fit(X, y, epochs=3, batch_size=10)

    genotype = darts.genotype()
    assert len(genotype) == 2
    assert all(op in DARTS.OPERATIONS for op in genotype)

    preds = darts.predict(X[:4])
    assert len(preds) == 4


def test_topological_data_analysis_and_rips():
    """Test VietorisRipsComplex, Persistent Homology, Diagrams, and Landscapes."""
    np.random.seed(42)
    # Circle point cloud (has 1-hole / H1 feature)
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])

    # 1. Vietoris-Rips Complex
    rips = VietorisRipsComplex(max_edge_length=3.0, max_dimension=1)
    diagrams = rips.fit_transform(circle)
    assert 0 in diagrams
    assert 1 in diagrams
    assert len(diagrams[0]) > 0

    # 2. Persistence Diagram & Entropy
    diag = PersistenceDiagram(diagrams[0], dimension=0)
    lt = diag.lifetimes()
    assert len(lt) > 0
    assert diag.total_persistence() >= 0.0
    assert diag.persistent_entropy() >= 0.0

    # 3. Persistence Landscape
    pl = PersistenceLandscape(num_landscapes=2, resolution=20)
    landscape_feat = pl.transform(diagrams[0])
    assert len(landscape_feat) == 40

    # 4. Bottleneck distance
    d1 = np.array([[0.0, 1.0], [0.2, 0.8]])
    d2 = np.array([[0.0, 1.1], [0.1, 0.7]])
    b_dist = bottleneck_distance(d1, d2)
    assert b_dist >= 0.0


def test_edge_cases_and_numerical_stability():
    """Test corner cases: extreme rotations, zero variance, empty diagrams, single sample inputs."""
    # 1. Quantum large angles
    qc = QuantumCircuit(n_qubits=2)
    qc.rx(100.0 * np.pi, 0).ry(-50.0 * np.pi, 1)
    probs = qc.probabilities()
    assert abs(float(np.sum(probs)) - 1.0) < 1e-8

    # 2. TDA empty diagram transform
    pl = PersistenceLandscape(num_landscapes=3, resolution=10)
    feat = pl.transform(np.empty((0, 2)))
    assert len(feat) == 30
    assert (feat == 0.0).all()

    # 3. Geospatial identical points and zero variance
    coords = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]])
    sw = SpatialWeights(coords, method="knn", k=2, row_standardize=True)
    m_res = morans_i(np.ones(4), sw, permutations=10)
    assert not np.isnan(m_res["I"])

    # 4. DARTS 1-sample batch
    darts = DARTS(input_dim=2, num_classes=2, num_intermediate_nodes=1, hidden_dim=4)
    darts.fit(np.array([[0.5, -0.5]]), np.array([0]), epochs=1, batch_size=1)
    assert len(darts.genotype()) == 1
