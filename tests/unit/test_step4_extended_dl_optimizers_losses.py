import numpy as np

from chokkhu.core.tensor import Tensor
from chokkhu.models.dl import (
    Adadelta,
    KLDivLoss,
    NAdam,
    Parameter,
    SmoothL1Loss,
)


def test_adadelta_optimizer():
    param = Parameter(np.array([2.0, -1.0], dtype=np.float64))
    opt = Adadelta([param], lr=1.0)
    opt.zero_grad()
    param.grad = np.array([0.5, -0.2], dtype=np.float64)
    opt.step()
    assert not np.allclose(param.data, [2.0, -1.0])


def test_nadam_optimizer():
    param = Parameter(np.array([1.5, 3.0], dtype=np.float64))
    opt = NAdam([param], lr=0.01)
    opt.zero_grad()
    param.grad = np.array([0.1, -0.3], dtype=np.float64)
    opt.step()
    assert not np.allclose(param.data, [1.5, 3.0])


def test_kldiv_loss():
    loss_fn = KLDivLoss(reduction="mean")
    p = Tensor(np.array([[0.6, 0.4]], dtype=np.float64), requires_grad=True)
    q = Tensor(np.array([[0.5, 0.5]], dtype=np.float64))
    loss = loss_fn(p, q)
    assert float(loss.data) >= 0.0
    loss.backward()
    assert p.grad is not None


def test_smooth_l1_loss():
    loss_fn = SmoothL1Loss(delta=1.0)
    pred = Tensor(np.array([1.0, 2.5], dtype=np.float64), requires_grad=True)
    target = Tensor(np.array([1.2, 0.5], dtype=np.float64))
    loss = loss_fn(pred, target)
    assert float(loss.data) >= 0.0
    loss.backward()
    assert pred.grad is not None
