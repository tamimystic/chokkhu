"""PointNet++ Hierarchical Feature Learning on Point Sets in Metric Spaces.

Pure NumPy implementation of Qi et al. (2017) "PointNet++":
- Farthest Point Sampling (FPS)
- Ball Query Radius Neighborhood Grouping
- Set Abstraction (SA) Modules
"""

from typing import Optional, Tuple
import numpy as np


def farthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    r"""Farthest Point Sampling (FPS) on 3D Point Cloud.

    Parameters
    ----------
    points : array-like of shape (B, N, 3) or (N, 3)
        Input point coordinates.
    num_samples : int
        Number of centroid points to sample (:math:`M \le N`).

    Returns
    -------
    centroids_idx : ndarray of shape (B, num_samples) or (num_samples,)
        Indices of sampled points.
    """
    pts = np.asarray(points, dtype=np.float32)
    is_2d = pts.ndim == 2
    if is_2d:
        pts = pts[None, ...]
    b, n, _ = pts.shape

    centroids = np.zeros((b, num_samples), dtype=np.int64)
    distance = np.full((b, n), 1e10, dtype=np.float32)

    for i in range(b):
        farthest = 0
        for j in range(num_samples):
            centroids[i, j] = farthest
            centroid_pt = pts[i, farthest, :][None, :]
            dist = np.sum((pts[i] - centroid_pt) ** 2, axis=-1)
            distance[i] = np.minimum(distance[i], dist)
            farthest = int(np.argmax(distance[i]))

    return centroids[0] if is_2d else centroids


def ball_query(
    radius: float,
    max_samples: int,
    points: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Ball Query radius neighborhood search around sampled centroids.

    Parameters
    ----------
    radius : float
        Search radius sphere.
    max_samples : int
        Maximum points to collect per query ball.
    points : array-like of shape (B, N, 3)
        All point coordinates.
    centroids : array-like of shape (B, S, 3)
        Sampled centroid coordinates.

    Returns
    -------
    group_idx : ndarray of shape (B, S, max_samples)
        Indices of neighbor points inside sphere.
    """
    pts = np.asarray(points, dtype=np.float32)
    cnts = np.asarray(centroids, dtype=np.float32)

    b, n, _ = pts.shape
    _, s, _ = cnts.shape

    group_idx = np.zeros((b, s, max_samples), dtype=np.int64)

    diff = cnts[:, :, None, :] - pts[:, None, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    r2 = radius**2

    for bi in range(b):
        for si in range(s):
            in_ball = np.where(dist_sq[bi, si] <= r2)[0]
            if len(in_ball) == 0:
                group_idx[bi, si, :] = 0
            elif len(in_ball) >= max_samples:
                group_idx[bi, si, :] = in_ball[:max_samples]
            else:
                group_idx[bi, si, : len(in_ball)] = in_ball
                group_idx[bi, si, len(in_ball) :] = in_ball[0]

    return group_idx


class SetAbstractionModule:
    """PointNet++ Set Abstraction (SA) Module.

    Downsamples points via FPS, groups neighbors via Ball Query, and extracts localized features via mini-PointNet.
    """

    def __init__(
        self,
        num_points: int,
        radius: float,
        max_samples: int,
        in_dim: int,
        out_dim: int,
    ) -> None:
        self.num_points: int = num_points
        self.radius: float = radius
        self.max_samples: int = max_samples
        self.in_dim: int = in_dim
        self.out_dim: int = out_dim

        self.w1: np.ndarray = np.random.randn(in_dim, 64).astype(np.float32) * np.sqrt(
            2.0 / in_dim
        )
        self.b1: np.ndarray = np.zeros(64, dtype=np.float32)
        self.w2: np.ndarray = np.random.randn(64, out_dim).astype(np.float32) * np.sqrt(
            2.0 / 64
        )
        self.b2: np.ndarray = np.zeros(out_dim, dtype=np.float32)

    def forward(
        self,
        points: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through SA module."""
        pts = np.asarray(points, dtype=np.float32)
        b, n, _ = pts.shape

        fps_idx = farthest_point_sampling(pts, self.num_points)
        new_points = np.zeros((b, self.num_points, 3), dtype=np.float32)
        for bi in range(b):
            new_points[bi] = pts[bi, fps_idx[bi]]

        group_idx = ball_query(self.radius, self.max_samples, pts, new_points)

        grouped_pts = np.zeros(
            (b, self.num_points, self.max_samples, self.in_dim), dtype=np.float32
        )
        for bi in range(b):
            for si in range(self.num_points):
                neighbors = pts[bi, group_idx[bi, si]]
                rel_coords = neighbors - new_points[bi, si][None, :]
                if features is not None:
                    feat_neighbors = features[bi, group_idx[bi, si]]
                    grouped_pts[bi, si] = np.concatenate(
                        [rel_coords, feat_neighbors], axis=-1
                    )
                else:
                    grouped_pts[bi, si] = rel_coords

        h = np.maximum(0.0, grouped_pts @ self.w1 + self.b1)
        h = np.maximum(0.0, h @ self.w2 + self.b2)

        new_features = np.max(h, axis=2)

        return new_points, new_features


class PointNet2Classifier:
    """PointNet++ Hierarchical Classification Network."""

    def __init__(self, num_classes: int = 40) -> None:
        self.num_classes: int = num_classes
        self.sa1: SetAbstractionModule = SetAbstractionModule(
            num_points=512, radius=0.2, max_samples=32, in_dim=3, out_dim=128
        )
        self.sa2: SetAbstractionModule = SetAbstractionModule(
            num_points=128, radius=0.4, max_samples=64, in_dim=128 + 3, out_dim=256
        )

        self.fc1_w: np.ndarray = np.random.randn(256, 128).astype(np.float32) * np.sqrt(
            2.0 / 256
        )
        self.fc1_b: np.ndarray = np.zeros(128, dtype=np.float32)
        self.fc2_w: np.ndarray = np.random.randn(128, num_classes).astype(
            np.float32
        ) * np.sqrt(2.0 / 128)
        self.fc2_b: np.ndarray = np.zeros(num_classes, dtype=np.float32)

    def forward(self, points: np.ndarray) -> np.ndarray:
        """Forward pass: (B, N, 3) -> (B, num_classes)."""
        x = np.asarray(points, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, ...]

        l1_pts, l1_feat = self.sa1.forward(x, features=None)
        _, l2_feat = self.sa2.forward(l1_pts, features=l1_feat)

        global_feat = np.max(l2_feat, axis=1)

        h = np.maximum(0.0, global_feat @ self.fc1_w + self.fc1_b)
        logits = h @ self.fc2_w + self.fc2_b
        return logits

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return self.forward(points)
