"""Unit tests for 3D Computer Vision, Point Clouds & Neural Radiance (NeRF & 3DGS)."""

import numpy as np
import pytest

from chokkhu.models.vision_3d import (
    TNet,
    PointNetClassifier,
    PointNetSegmenter,
    PointNet2Classifier,
    SetAbstractionModule,
    farthest_point_sampling,
    ball_query,
    PositionalEncoder,
    NeRFMLP,
    volume_render,
    generate_camera_rays,
    GaussianSplatting3D,
    quaternion_to_rotation_matrix,
)


def test_tnet_forward() -> None:
    np.random.seed(42)
    b, n, k = 2, 64, 3
    x = np.random.randn(b, n, k).astype(np.float32)

    tnet = TNet(k=k)
    trans = tnet.forward(x)

    assert trans.shape == (b, k, k)
    # T-Net initialized near identity
    for bi in range(b):
        assert np.allclose(trans[bi], np.eye(k), atol=0.1)


def test_pointnet_classifier() -> None:
    np.random.seed(42)
    b, n = 2, 128
    points = np.random.randn(b, n, 3).astype(np.float32)

    clf = PointNetClassifier(num_classes=10, in_dim=3)
    logits = clf(points)

    assert logits.shape == (b, 10)

    # Single point cloud (2D input)
    single_pts = points[0]
    single_logits = clf(single_pts)
    assert single_logits.shape == (1, 10)


def test_pointnet_segmenter() -> None:
    np.random.seed(42)
    b, n = 2, 64
    num_classes = 16
    points = np.random.randn(b, n, 3).astype(np.float32)

    segmenter = PointNetSegmenter(num_classes=num_classes, in_dim=3)
    seg_logits = segmenter(points)

    assert seg_logits.shape == (b, n, num_classes)


def test_farthest_point_sampling() -> None:
    np.random.seed(42)
    b, n = 2, 100
    points = np.random.randn(b, n, 3).astype(np.float32)

    sampled_idx = farthest_point_sampling(points, num_samples=20)
    assert sampled_idx.shape == (b, 20)
    # All sampled indices within a batch should be unique
    assert len(np.unique(sampled_idx[0])) == 20

    # 2D input
    sampled_2d = farthest_point_sampling(points[0], num_samples=10)
    assert sampled_2d.shape == (10,)


def test_ball_query() -> None:
    np.random.seed(42)
    b, n, s = 2, 100, 10
    points = np.random.randn(b, n, 3).astype(np.float32)
    centroids = points[:, :s, :].copy()

    group_idx = ball_query(
        radius=1.5, max_samples=16, points=points, centroids=centroids
    )
    assert group_idx.shape == (b, s, 16)


def test_set_abstraction_module() -> None:
    np.random.seed(42)
    b, n = 2, 128
    points = np.random.randn(b, n, 3).astype(np.float32)

    sa = SetAbstractionModule(
        num_points=32, radius=0.5, max_samples=16, in_dim=3, out_dim=64
    )
    new_pts, new_feat = sa.forward(points)

    assert new_pts.shape == (b, 32, 3)
    assert new_feat.shape == (b, 32, 64)


def test_pointnet2_classifier() -> None:
    np.random.seed(42)
    b, n = 2, 600
    points = np.random.randn(b, n, 3).astype(np.float32)

    clf2 = PointNet2Classifier(num_classes=10)
    logits = clf2(points)

    assert logits.shape == (b, 10)


def test_positional_encoder() -> None:
    encoder = PositionalEncoder(num_freqs=4, include_input=True)
    coords = np.array([[0.5, 0.2, -0.3]], dtype=np.float32)
    encoded = encoder.encode(coords)

    # 3 + (3 * 4 * 2) = 27 features
    assert encoded.shape == (1, 27)
    assert encoder.out_dim(3) == 27


def test_nerf_mlp() -> None:
    np.random.seed(42)
    n_points = 50
    positions = np.random.randn(n_points, 3).astype(np.float32)
    view_dirs = np.random.randn(n_points, 3).astype(np.float32)
    view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=-1, keepdims=True)

    nerf = NeRFMLP(pos_freqs=4, dir_freqs=2, hidden_dim=64)
    rgb, sigma = nerf(positions, view_dirs)

    assert rgb.shape == (n_points, 3)
    assert sigma.shape == (n_points, 1)
    assert np.all(rgb >= 0.0) and np.all(rgb <= 1.0)
    assert np.all(sigma >= 0.0)


def test_volume_render() -> None:
    np.random.seed(42)
    n_rays = 16
    n_samples = 32

    rgb = np.random.uniform(0.0, 1.0, size=(n_rays, n_samples, 3)).astype(np.float32)
    sigma = np.random.uniform(0.1, 2.0, size=(n_rays, n_samples, 1)).astype(np.float32)
    z_vals = (
        np.linspace(2.0, 6.0, n_samples)
        .reshape(1, -1)
        .repeat(n_rays, axis=0)
        .astype(np.float32)
    )
    rays_d = np.ones((n_rays, 3), dtype=np.float32) / np.sqrt(3.0)

    comp_rgb, depth_map, acc_map = volume_render(rgb, sigma, z_vals, rays_d)

    assert comp_rgb.shape == (n_rays, 3)
    assert depth_map.shape == (n_rays,)
    assert acc_map.shape == (n_rays,)
    assert np.all(comp_rgb >= 0.0)
    assert np.all(depth_map >= 2.0) and np.all(depth_map <= 6.0)


def test_generate_camera_rays() -> None:
    c2w: np.ndarray = np.eye(4, dtype=np.float32)
    c2w[:3, 3] = np.array([0.0, 0.0, 4.0])

    rays_o, rays_d = generate_camera_rays(
        height=16, width=16, focal_length=20.0, camera_pose=c2w
    )

    assert rays_o.shape == (256, 3)
    assert rays_d.shape == (256, 3)
    # Unit direction vectors
    norms = np.linalg.norm(rays_d, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_quaternion_to_rotation_matrix() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    r = quaternion_to_rotation_matrix(q)
    assert r.shape == (3, 3)
    assert np.allclose(r, np.eye(3), atol=1e-5)

    # 90-degree rotation around z-axis: q = (cos(pi/4), 0, 0, sin(pi/4))
    q_z90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], dtype=np.float32)
    r_z90 = quaternion_to_rotation_matrix(q_z90)
    # R * R.T = I
    assert np.allclose(r_z90 @ r_z90.T, np.eye(3), atol=1e-5)
    assert pytest.approx(float(np.linalg.det(r_z90)), 1e-4) == 1.0


def test_gaussian_splatting_3d() -> None:
    np.random.seed(42)
    gs = GaussianSplatting3D(num_gaussians=50)

    # Covariance computation
    covs = gs.compute_3d_covariances()
    assert covs.shape == (50, 3, 3)

    # 2D projection
    camera_pose: np.ndarray = np.eye(4, dtype=np.float32)
    camera_pose[:3, 3] = np.array([0.0, 0.0, 3.0])

    means_2d, covs_2d, depths = gs.project_to_2d(
        camera_pose, focal_length=50.0, img_width=32, img_height=32
    )
    assert means_2d.shape == (50, 2)
    assert covs_2d.shape == (50, 2, 2)
    assert depths.shape == (50,)

    # Rendering
    img = gs.render(camera_pose, focal_length=50.0, img_width=32, img_height=32)
    assert img.shape == (32, 32, 3)
    assert np.all(img >= 0.0) and np.all(img <= 1.0)
