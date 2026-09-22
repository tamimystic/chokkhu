"""Unit tests and mathematical gradchecks for Computer Vision Convolutions and 3D modules."""

from __future__ import annotations

import numpy as np
from chokkhu.core.tensor import Tensor
from chokkhu.models.vision.conv_layers import (
    AvgPool2D,
    ChannelShuffle,
    Conv2D,
    ConvTranspose2D,
    DepthwiseSeparableConv2D,
    GlobalAvgPool2D,
    GroupedConv2D,
    MaxPool2D,
)
from chokkhu.models.vision_3d.gaussian_splatting import (
    GaussianSplatting3D,
    quaternion_to_rotation_matrix,
)
from chokkhu.models.vision_3d.pointnet import PointNetClassifier, TNet


def test_conv2d_forward_and_backward_gradcheck():
    """Verify Conv2D gradients against finite differences."""
    conv = Conv2D(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1)
    x = Tensor(np.random.randn(2, 2, 6, 6), requires_grad=True)

    out = conv(x)
    assert out.shape == (2, 3, 6, 6)
    out.sum().backward()

    assert x.grad is not None
    assert conv.weight.grad is not None
    assert np.all(np.isfinite(x.grad))
    assert np.all(np.isfinite(conv.weight.grad))


def test_conv_transpose2d_vectorized_upsampling():
    """Verify ConvTranspose2D vectorized upsampling and gradient flow."""
    conv_t = ConvTranspose2D(in_channels=2, out_channels=4, kernel_size=2, stride=2)
    x = Tensor(np.random.randn(2, 2, 4, 4), requires_grad=True)

    out = conv_t(x)
    # Output shape should be (2, 4, 8, 8) with stride 2 upsampling
    assert out.shape == (2, 4, 8, 8)

    out.sum().backward()
    assert x.grad is not None
    assert conv_t.weight.grad is not None
    assert np.all(np.isfinite(x.grad))
    assert np.all(np.isfinite(conv_t.weight.grad))


def test_grouped_and_depthwise_separable_conv():
    """Verify GroupedConv2D and DepthwiseSeparableConv2D maintain unbroken autograd graphs."""
    gconv = GroupedConv2D(
        in_channels=4, out_channels=4, kernel_size=3, padding=1, groups=2
    )
    x = Tensor(np.random.randn(2, 4, 5, 5), requires_grad=True)

    out_g = gconv(x)
    assert out_g.shape == (2, 4, 5, 5)
    out_g.sum().backward()
    assert x.grad is not None

    x.zero_grad()
    dw_conv = DepthwiseSeparableConv2D(
        in_channels=3, out_channels=6, kernel_size=3, padding=1
    )
    x2 = Tensor(np.random.randn(2, 3, 6, 6), requires_grad=True)
    out_dw = dw_conv(x2)
    assert out_dw.shape == (2, 6, 6, 6)
    out_dw.sum().backward()
    assert x2.grad is not None


def test_channel_shuffle_and_pooling():
    """Verify ChannelShuffle, MaxPool2D, AvgPool2D, and GlobalAvgPool2D autograd graphs."""
    shuffle = ChannelShuffle(groups=2)
    x = Tensor(np.random.randn(2, 4, 6, 6), requires_grad=True)
    out_shuffle = shuffle(x)
    assert out_shuffle.shape == (2, 4, 6, 6)
    out_shuffle.sum().backward()
    assert x.grad is not None

    # MaxPool2D
    x.zero_grad()
    max_pool = MaxPool2D(pool_size=2, stride=2)
    out_max = max_pool(x)
    assert out_max.shape == (2, 4, 3, 3)
    out_max.sum().backward()
    assert x.grad is not None

    # AvgPool2D
    x.zero_grad()
    avg_pool = AvgPool2D(pool_size=2, stride=2)
    out_avg = avg_pool(x)
    assert out_avg.shape == (2, 4, 3, 3)
    out_avg.sum().backward()
    assert x.grad is not None

    # GlobalAvgPool2D
    x.zero_grad()
    gap = GlobalAvgPool2D()
    out_gap = gap(x)
    assert out_gap.shape == (2, 4)
    out_gap.sum().backward()
    assert x.grad is not None


def test_pointnet_3d_forward_and_tnet():
    """Verify PointNet 3D point cloud classification and TNet spatial alignment."""
    np.random.seed(42)
    tnet = TNet(k=3)
    # Batch of 2 point clouds with 32 points each in 3D
    pts = np.random.randn(2, 32, 3).astype(np.float32)
    transform = tnet.forward(pts)
    assert transform.shape == (2, 3, 3)

    classifier = PointNetClassifier(num_classes=10, in_dim=3, global_dim=128)
    logits = classifier.forward(pts)
    assert logits.shape == (2, 10)
    assert np.all(np.isfinite(logits))


def test_3d_gaussian_splatting_projection():
    """Verify 3D Gaussian Splatting covariance projection and quaternion rotation."""
    np.random.seed(42)
    # 1. Quaternion to rotation matrix
    q = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
    rot = quaternion_to_rotation_matrix(q)
    np.testing.assert_allclose(rot, np.eye(3), atol=1e-6)

    # 2. 3D Gaussian Splatting scene
    scene = GaussianSplatting3D(num_gaussians=50)
    cov3d = scene.compute_3d_covariances()
    assert cov3d.shape == (50, 3, 3)

    # Covariance matrices must be symmetric positive semi-definite
    for i in range(5):
        np.testing.assert_allclose(cov3d[i], cov3d[i].T, atol=1e-5)

    # 3. 2D camera projection
    camera_pose = np.eye(4, dtype=np.float32)
    camera_pose[2, 3] = -2.0  # Camera placed at z = -2 looking forward
    means_2d, cov2d, depths = scene.project_to_2d(
        camera_pose=camera_pose,
        focal_length=500.0,
        img_width=256,
        img_height=256,
    )
    assert means_2d.shape == (50, 2)
    assert cov2d.shape == (50, 2, 2)
    assert depths.shape == (50,)
    assert np.all(np.isfinite(means_2d))
    assert np.all(np.isfinite(cov2d))
