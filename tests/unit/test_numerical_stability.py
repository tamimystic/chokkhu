"""Comprehensive numerical stability, precision, and edge-case stress tests for Chokkhu."""

from __future__ import annotations

import numpy as np
from chokkhu.core.tensor import Tensor
from chokkhu.models.dl.layers import BatchNorm1d, LayerNorm, Parameter
from chokkhu.models.dl.losses import (
    BinaryCrossEntropyLoss,
    CrossEntropyLoss,
    HuberLoss,
    MAELoss,
    MSELoss,
)
from chokkhu.models.dl.optimizers import SGD, Adam, AdamW, RMSProp
from chokkhu.models.nlp.transformer_blocks import RMSNorm
from chokkhu.models.vision.losses import (
    CIoULoss,
    DIoULoss,
    DiceLoss,
    FocalLoss,
    GIoULoss,
    TverskyLoss,
)


def test_tensor_log_exp_sqrt_stability():
    # Test log with zeros and near-zero values
    x = Tensor([0.0, 1e-30, 1.0, 10.0], requires_grad=True)
    y = x.log()
    assert np.all(np.isfinite(y.data))
    y.sum().backward()
    assert np.all(np.isfinite(x.grad))

    # Test exp with extreme values (overflow protection)
    x = Tensor([-1000.0, -500.0, 0.0, 500.0, 1000.0], requires_grad=True)
    y = x.exp()
    assert np.all(np.isfinite(y.data))
    y.sum().backward()
    assert np.all(np.isfinite(x.grad))

    # Test sqrt with zero and positive values
    x = Tensor([0.0, 0.01, 4.0, 16.0], requires_grad=True)
    y = x.sqrt()
    assert np.all(np.isfinite(y.data))
    y.sum().backward()
    assert np.all(np.isfinite(x.grad))


def test_tensor_div_pow_zero_division():
    # Division by zero protection
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([0.0, 0.0, 2.0], requires_grad=True)
    c = a / b
    assert np.all(np.isfinite(c.data))
    c.sum().backward()
    assert np.all(np.isfinite(a.grad))
    assert np.all(np.isfinite(b.grad))

    # Power with zero base and fractional exponent
    x = Tensor([0.0, 1.0, 4.0], requires_grad=True)
    p = x**0.5
    assert np.all(np.isfinite(p.data))
    p.sum().backward()
    assert np.all(np.isfinite(x.grad))


def test_tensor_trigonometric_and_clip():
    x = Tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
    y_sin = x.sin()
    y_cos = x.cos()
    y_clip = x.clip(min_val=-1.0, max_val=1.0)

    assert np.allclose(y_sin.data, np.sin(x.data))
    assert np.allclose(y_cos.data, np.cos(x.data))
    assert np.all(y_clip.data >= -1.0) and np.all(y_clip.data <= 1.0)

    (y_sin + y_cos + y_clip).sum().backward()
    assert np.all(np.isfinite(x.grad))


def test_cross_entropy_extreme_logits():
    loss_fn = CrossEntropyLoss()

    # Extreme logits (scale of 1000)
    logits = Tensor([[1000.0, -1000.0, 0.0], [-500.0, 500.0, 0.0]], requires_grad=True)
    targets = Tensor([0, 1])

    loss = loss_fn(logits, targets)
    assert np.isfinite(loss.data)
    assert loss.data < 1.0

    loss.backward()
    assert np.all(np.isfinite(logits.grad))

    # All identical logits
    logits2 = Tensor([[10.0, 10.0, 10.0]], requires_grad=True)
    targets2 = Tensor([2])
    loss2 = loss_fn(logits2, targets2)
    assert np.allclose(loss2.data, np.log(3.0))
    loss2.backward()
    assert np.all(np.isfinite(logits2.grad))


def test_binary_cross_entropy_boundaries():
    loss_fn_logits = BinaryCrossEntropyLoss(from_logits=True)
    loss_fn_probs = BinaryCrossEntropyLoss(from_logits=False)

    # From logits with extreme values
    pred_logits = Tensor([-100.0, 100.0, 0.0], requires_grad=True)
    y_true = Tensor([0.0, 1.0, 1.0])

    loss1 = loss_fn_logits(pred_logits, y_true)
    assert np.isfinite(loss1.data)
    loss1.backward()
    assert np.all(np.isfinite(pred_logits.grad))

    # From probabilities with boundary values (0.0 and 1.0)
    pred_probs = Tensor([0.0, 1.0, 0.5], requires_grad=True)
    loss2 = loss_fn_probs(pred_probs, y_true)
    assert np.isfinite(loss2.data)
    loss2.backward()
    assert np.all(np.isfinite(pred_probs.grad))


def test_regression_losses():
    mse = MSELoss()
    mae = MAELoss()
    huber = HuberLoss(delta=1.0)

    y_pred = Tensor([0.0, 10.0, -5.0], requires_grad=True)
    y_true = Tensor([0.0, 0.0, 0.0])

    l_mse = mse(y_pred, y_true)
    l_mae = mae(y_pred, y_true)
    l_huber = huber(y_pred, y_true)

    assert (
        np.isfinite(l_mse.data)
        and np.isfinite(l_mae.data)
        and np.isfinite(l_huber.data)
    )

    (l_mse + l_mae + l_huber).backward()
    assert np.all(np.isfinite(y_pred.grad))


def test_segmentation_losses_edge_cases():
    dice = DiceLoss()
    tversky = TverskyLoss()
    focal = FocalLoss()

    # All zeros (empty masks)
    pred_empty = Tensor(np.zeros((2, 10)), requires_grad=True)
    true_empty = Tensor(np.zeros((2, 10)))

    l_dice = dice(pred_empty, true_empty)
    l_tversky = tversky(pred_empty, true_empty)
    l_focal = focal(pred_empty, true_empty)

    assert np.isfinite(l_dice.data)
    assert np.isfinite(l_tversky.data)
    assert np.isfinite(l_focal.data)


def test_box_losses_edge_cases():
    giou = GIoULoss()
    diou = DIoULoss()
    ciou = CIoULoss()

    # Identical bounding boxes
    boxes1 = Tensor([[10.0, 10.0, 50.0, 50.0]])
    boxes2 = Tensor([[10.0, 10.0, 50.0, 50.0]])

    assert np.isclose(giou(boxes1, boxes2).data, 0.0, atol=1e-5)
    assert np.isclose(diou(boxes1, boxes2).data, 0.0, atol=1e-5)
    assert np.isclose(ciou(boxes1, boxes2).data, 0.0, atol=1e-5)

    # Disjoint far-apart bounding boxes
    boxes_far = Tensor([[1000.0, 1000.0, 1050.0, 1050.0]])
    l_giou = giou(boxes1, boxes_far)
    l_diou = diou(boxes1, boxes_far)
    l_ciou = ciou(boxes1, boxes_far)

    assert np.isfinite(l_giou.data)
    assert np.isfinite(l_diou.data)
    assert np.isfinite(l_ciou.data)


def test_normalization_layers_zero_variance():
    # Constant input vector where variance = 0
    x_const = Tensor(np.full((4, 8), 5.0), requires_grad=True)

    ln = LayerNorm(normalized_shape=8)
    out_ln = ln(x_const)
    assert np.all(np.isfinite(out_ln.data))
    out_ln.sum().backward()
    assert np.all(np.isfinite(x_const.grad))

    x_const2 = Tensor(np.full((4, 8), 7.0), requires_grad=True)
    rms = RMSNorm(dim=8)
    out_rms = rms(x_const2)
    assert np.all(np.isfinite(out_rms.data))
    out_rms.sum().backward()
    assert np.all(np.isfinite(x_const2.grad))

    bn = BatchNorm1d(num_features=8)
    bn.train()
    out_bn = bn(x_const)
    assert np.all(np.isfinite(out_bn.data))


def test_optimizers_stability():
    w = Parameter(np.random.randn(4, 4))
    b = Parameter(np.zeros((1, 4)))

    optimizers = [
        SGD([w, b], lr=0.01, momentum=0.9, nesterov=True),
        Adam([w, b], lr=0.001),
        AdamW([w, b], lr=0.001, weight_decay=0.01),
        RMSProp([w, b], lr=0.001),
    ]

    for opt in optimizers:
        w.grad = np.zeros_like(w.data)
        b.grad = np.zeros_like(b.data)
        opt.step()
        assert np.all(np.isfinite(w.data))
        assert np.all(np.isfinite(b.data))

        # Large gradient step
        w.grad = np.ones_like(w.data) * 1e4
        b.grad = np.ones_like(b.data) * 1e4
        opt.step()
        assert np.all(np.isfinite(w.data))
        assert np.all(np.isfinite(b.data))
