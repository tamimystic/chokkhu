"""Comprehensive numerical gradient check (gradcheck) test suite for Chokkhu Autograd."""

import numpy as np
from chokkhu.core.tensor import Tensor, concat
from chokkhu.models.dl.layers import Linear
from chokkhu.models.nlp.attention import MultiHeadAttention
from chokkhu.models.nlp.embeddings import RotaryPositionEmbedding
from chokkhu.models.vision.architectures.vit import VisionTransformer


def numerical_gradcheck(func, inputs, eps=1e-5, rtol=1e-3, atol=1e-4):
    """Numerically verify gradients using central finite differences."""
    out = func(*inputs)
    out.sum().backward()

    for i, inp in enumerate(inputs):
        if not inp.requires_grad:
            continue
        analytical_grad = inp.grad.copy()
        numerical_grad = np.zeros_like(inp.data)

        it = np.nditer(inp.data, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig_val = inp.data[idx]

            inp.data[idx] = orig_val + eps
            out_pos = func(*inputs)
            loss_pos = np.sum(out_pos.data)

            inp.data[idx] = orig_val - eps
            out_neg = func(*inputs)
            loss_neg = np.sum(out_neg.data)

            inp.data[idx] = orig_val
            numerical_grad[idx] = (loss_pos - loss_neg) / (2.0 * eps)
            it.iternext()

        np.testing.assert_allclose(
            analytical_grad,
            numerical_grad,
            rtol=rtol,
            atol=atol,
            err_msg=f"Gradcheck failed for input {i}",
        )


def test_gradcheck_arithmetic():
    x = Tensor(np.random.randn(3, 4), requires_grad=True)
    y = Tensor(np.random.randn(3, 4), requires_grad=True)

    numerical_gradcheck(lambda a, b: a + b, [x, y])
    x.zero_grad()
    y.zero_grad()
    numerical_gradcheck(lambda a, b: a - b, [x, y])
    x.zero_grad()
    y.zero_grad()
    numerical_gradcheck(lambda a, b: a * b, [x, y])
    x.zero_grad()
    y.zero_grad()
    numerical_gradcheck(lambda a, b: a / (b**2 + 0.5), [x, y])


def test_gradcheck_matmul():
    a = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    b = Tensor(np.random.randn(2, 4, 5), requires_grad=True)
    numerical_gradcheck(lambda x, y: x @ y, [a, b])


def test_gradcheck_activations():
    x = Tensor(
        np.array([[-1.5, -0.5, 0.5, 1.5], [2.0, -1.0, 0.2, -0.8]]), requires_grad=True
    )
    numerical_gradcheck(lambda t: t.sigmoid(), [x])
    x.zero_grad()
    numerical_gradcheck(lambda t: t.tanh(), [x])
    x.zero_grad()
    numerical_gradcheck(lambda t: t.gelu(), [x])
    x.zero_grad()
    numerical_gradcheck(lambda t: t.softmax(axis=-1), [x])


def test_gradcheck_slice_concat():
    x = Tensor(np.random.randn(4, 6), requires_grad=True)
    numerical_gradcheck(lambda t: t[1:3, 2:5], [x])
    x.zero_grad()

    y = Tensor(np.random.randn(4, 4), requires_grad=True)
    numerical_gradcheck(lambda a, b: concat([a, b], axis=1), [x, y])


def test_gradcheck_linear_layer():
    linear = Linear(4, 3)
    x = Tensor(np.random.randn(2, 4), requires_grad=True)
    numerical_gradcheck(lambda t: linear(t), [x])


def test_gradcheck_attention():
    mha = MultiHeadAttention(embed_dim=8, num_heads=2)
    x = Tensor(np.random.randn(2, 3, 8), requires_grad=True)
    numerical_gradcheck(lambda t: mha(t), [x])


def test_gradcheck_rope():
    rope = RotaryPositionEmbedding(dim=4, max_seq_len=16)
    x = Tensor(np.random.randn(2, 2, 4, 4), requires_grad=True)
    numerical_gradcheck(lambda t: rope.apply_rope(t, seq_len=4), [x])


def test_gradcheck_vit():
    vit = VisionTransformer(
        img_size=8,
        patch_size=4,
        in_channels=3,
        num_classes=2,
        embed_dim=8,
        depth=1,
        num_heads=2,
    )
    x = Tensor(np.random.randn(1, 3, 8, 8), requires_grad=True)
    numerical_gradcheck(lambda t: vit(t), [x])
