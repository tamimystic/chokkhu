"""Unit tests for Sovereign Neural Explainable AI (Integrated Gradients, SmoothGrad, DeepLIFT)."""

import numpy as np

from chokkhu.core.tensor import Tensor
from chokkhu.explainability import (
    IntegratedGradients,
    SmoothGrad,
    DeepLIFT,
    explain,
    ExplanationResult,
)
from chokkhu.models.dl.layers import Linear, Module
from chokkhu.models.dl.activations import ReLU


class SimpleNet(Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = Linear(4, 8)
        self.act = ReLU()
        self.fc2 = Linear(8, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


def test_integrated_gradients_linear_and_nonlinear() -> None:
    # 1. Linear function test: f(x) = 2*x0 + 3*x1 - x2 + 5
    def linear_model(x: np.ndarray) -> np.ndarray:
        weights = np.array([2.0, 3.0, -1.0, 0.5])
        return np.dot(x, weights) + 5.0

    ig = IntegratedGradients(linear_model, steps=20)
    x = np.array([1.0, 2.0, 3.0, 4.0])
    baseline = np.zeros(4)

    attr = ig.attribute(x)
    assert attr.shape == (4,)

    # Completeness axiom: sum(attr) should equal f(x) - f(0)
    expected_diff = linear_model(x) - linear_model(baseline)
    np.testing.assert_allclose(np.sum(attr), expected_diff, rtol=1e-3)

    # 2. Neural network module
    net = SimpleNet()
    ig_net = IntegratedGradients(net, steps=20)
    attr_net = ig_net.attribute(x)
    assert attr_net.shape == (4,)


def test_smoothgrad_saliency() -> None:
    net = SimpleNet()
    sg = SmoothGrad(net, num_samples=15, noise_level=0.1)
    x = np.array([0.5, -1.0, 2.0, 0.3])

    saliency = sg.attribute(x)
    assert saliency.shape == (4,)
    assert not np.isnan(saliency).any()


def test_deeplift_conservation() -> None:
    net = SimpleNet()
    x = np.array([1.5, 0.5, -0.5, 2.0])
    baseline = np.zeros(4)

    dl = DeepLIFT(net, baseline=baseline)
    attr = dl.attribute(x)
    assert attr.shape == (4,)

    # Conservation axiom: sum of attributions equals delta y
    y_target = float(net(Tensor(x.reshape(1, -1))).data.flatten()[0])
    y_base = float(net(Tensor(baseline.reshape(1, -1))).data.flatten()[0])
    delta_y = y_target - y_base

    np.testing.assert_allclose(np.sum(attr), delta_y, rtol=1e-3)


def test_explain_unified_with_neural_methods() -> None:
    net = SimpleNet()
    X = np.random.randn(5, 4)
    feature_names = ["feat_0", "feat_1", "feat_2", "feat_3"]

    # 1. Integrated Gradients
    res_ig = explain(
        net, X, method="integrated_gradients", feature_names=feature_names, n_samples=10
    )
    assert isinstance(res_ig, ExplanationResult)
    df_ig = res_ig.to_dataframe()
    assert len(df_ig) == 4
    assert "importance" in df_ig.columns
    assert "std" in df_ig.columns
    assert "CHOKKHU EXPLAINABILITY REPORT" in res_ig.summary()

    # 2. SmoothGrad
    res_sg = explain(
        net, X, method="smoothgrad", feature_names=feature_names, n_samples=10
    )
    assert isinstance(res_sg, ExplanationResult)
    df_sg = res_sg.to_dataframe()
    assert len(df_sg) == 4

    # 3. DeepLIFT
    res_dl = explain(net, X, method="deeplift", feature_names=feature_names)
    assert isinstance(res_dl, ExplanationResult)
    df_dl = res_dl.to_dataframe()
    assert len(df_dl) == 4
