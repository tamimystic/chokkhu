from __future__ import annotations

import numpy as np
from chokkhu.core.tensor import Tensor
from chokkhu.models.dl import (
    Adam,
    CrossEntropyLoss,
    Linear,
    ReLU,
    Sequential,
)


def test_tensor_basic_autograd():
    # Scalar computation: z = (x * y) + y^2
    x = Tensor(3.0, requires_grad=True)
    y = Tensor(4.0, requires_grad=True)
    z = (x * y) + (y**2)

    assert float(z.data) == 28.0
    z.backward()

    # dz/dx = y = 4
    # dz/dy = x + 2y = 3 + 8 = 11
    assert float(x.grad) == 4.0
    assert float(y.grad) == 11.0


def test_tensor_matrix_multiplication():
    # Matrix computation: Y = X @ W
    X = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    W = Tensor([[0.5, -0.5], [1.0, 2.0]], requires_grad=True)
    Y = X @ W
    loss = Y.sum()
    loss.backward()

    expected_dX = np.dot(np.ones((2, 2)), W.data.T)
    expected_dW = np.dot(X.data.T, np.ones((2, 2)))

    np.testing.assert_allclose(X.grad, expected_dX)
    np.testing.assert_allclose(W.grad, expected_dW)


def test_tensor_activations_and_losses():
    x = Tensor([-2.0, 0.0, 2.0], requires_grad=True)
    r = x.relu()
    assert np.all(r.data == [0.0, 0.0, 2.0])
    r.sum().backward()
    assert np.all(x.grad == [0.0, 0.0, 1.0])

    # CrossEntropy Loss
    logits = Tensor([[2.0, 1.0, 0.1], [0.1, 3.0, 0.2]], requires_grad=True)
    targets = Tensor([0, 1])
    criterion = CrossEntropyLoss()
    ce_loss = criterion(logits, targets)
    assert float(ce_loss.data) > 0.0
    ce_loss.backward()
    assert logits.grad is not None


def test_sequential_dense_training():
    np.random.seed(42)
    X = np.random.randn(50, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = Sequential(task="classification")
    model.add(Linear(4, 16))
    model.add(ReLU())
    model.add(Linear(16, 2))

    optimizer = Adam(model.parameters(), lr=0.05)
    model.fit(X, y, epochs=15, batch_size=16, optimizer=optimizer, verbose=False)

    preds = model.predict(X[:5])
    assert len(preds) == 5
    probs = model.predict_proba(X[:5])
    assert probs.shape == (5, 2)
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(5), atol=1e-5)
