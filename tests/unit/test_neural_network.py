import numpy as np
import chokkhu as ck
from chokkhu.models.ml.neural_network import NeuralNetwork


def test_neural_network_classification():
    np.random.seed(42)
    X = np.random.randn(80, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Train via ck.train
    model = ck.train(
        model="neural_network",
        X_train=X,
        y_train=y,
        task="classification",
        layers=[16, 8],
        epochs=30,
        learning_rate=0.05,
        random_state=42,
        verbose=False,
    )
    assert isinstance(model, NeuralNetwork)

    preds = model.predict(X[:5])
    assert len(preds) == 5
    assert set(preds).issubset({0, 1})

    probs = model.predict_proba(X[:5])
    assert probs.shape == (5, 2)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(5), atol=1e-5)


def test_neural_network_regression():
    np.random.seed(42)
    X = np.random.randn(80, 3)
    y = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5 * X[:, 2]

    model = NeuralNetwork(
        layers=[16, 8],
        task="regression",
        epochs=40,
        learning_rate=0.01,
        random_state=42,
    )
    model.fit(X, y)

    preds = model.predict(X[:5])
    assert len(preds) == 5
    assert isinstance(preds[0], (float, np.floating, int, np.integer))
