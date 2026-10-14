import numpy as np

from chokkhu.hdc import HDCClassifier
from chokkhu.quantum import QuantumKernel, VariationalQuantumClassifier


def test_hdc_classifier_dynamic_params_and_proba() -> None:
    np.random.seed(42)
    hdc = HDCClassifier(dim=1000, num_levels=16, seed=42)
    assert hdc.dim == 1000
    assert hdc.num_levels == 16

    X = np.random.randn(20, 4)
    y = np.array([0] * 10 + [1] * 10)

    hdc.fit(X, y)
    preds = hdc.predict(X[:4])
    assert preds.shape == (4,)

    proba = hdc.predict_proba(X[:4])
    assert proba.shape == (4, 2)
    np.testing.assert_allclose(np.sum(proba, axis=1), np.ones(4), atol=1e-5)
    assert np.all(proba >= 0.0)

    params = hdc.get_params()
    assert params["dim"] == 1000
    assert params["num_levels"] == 16
    hdc.set_params(dim=2000)
    assert hdc.dim == 2000


def test_vqc_dynamic_params_and_proba() -> None:
    np.random.seed(42)
    vqc = VariationalQuantumClassifier(n_qubits=2, n_layers=1, lr=0.05, random_state=42)
    assert vqc.n_qubits == 2
    assert vqc.n_layers == 1

    X = np.array([[0.5, -0.2], [-0.5, 0.3], [0.8, -0.7], [-0.9, 0.4]])
    y = np.array([1, 0, 1, 0])

    vqc.fit(X, y, epochs=2, batch_size=2)
    preds = vqc.predict(X)
    assert preds.shape == (4,)

    proba = vqc.predict_proba(X)
    assert proba.shape == (4, 2)
    np.testing.assert_allclose(np.sum(proba, axis=1), np.ones(4), atol=1e-5)

    params = vqc.get_params()
    assert params["n_qubits"] == 2
    assert params["n_layers"] == 1
    vqc.set_params(n_layers=2)
    assert vqc.n_layers == 2


def test_quantum_kernel_dynamic_params_and_gram_matrix() -> None:
    qk = QuantumKernel(n_qubits=2)
    params = qk.get_params()
    assert params["n_qubits"] == 2
    qk.set_params(n_qubits=3)
    assert qk.n_qubits == 3

    X = np.array([[0.1, 0.2, -0.1], [-0.2, 0.3, 0.4]])
    K = qk.compute_matrix(X)
    assert K.shape == (2, 2)
    np.testing.assert_allclose(np.diag(K), np.ones(2), atol=1e-5)
    assert np.isclose(K[0, 1], K[1, 0])
