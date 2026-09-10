"""PointNet 3D Point Cloud Classification and Segmentation Architecture.

Pure NumPy implementation of Qi et al. (2017) "PointNet: Deep Learning on Point Sets":
- T-Net Spatial Transformation Network with Orthogonality Regularization
- PointNetClassifier: Global feature aggregation via symmetric max-pooling
- PointNetSegmenter: Per-point feature concatenation for part/semantic segmentation
"""

import numpy as np


class TNet:
    """Spatial Transformer Network for Point Clouds (T-Net).

    Learns an affine transformation matrix :math:`(K \times K)` to align input points or features.
    """

    def __init__(self, k: int = 3) -> None:
        self.k: int = k
        # MLP layers: k -> 64 -> 128 -> 1024
        self.w1: np.ndarray = np.random.randn(k, 64).astype(np.float32) * np.sqrt(
            2.0 / k
        )
        self.b1: np.ndarray = np.zeros(64, dtype=np.float32)

        self.w2: np.ndarray = np.random.randn(64, 128).astype(np.float32) * np.sqrt(
            2.0 / 64
        )
        self.b2: np.ndarray = np.zeros(128, dtype=np.float32)

        self.w3: np.ndarray = np.random.randn(128, 1024).astype(np.float32) * np.sqrt(
            2.0 / 128
        )
        self.b3: np.ndarray = np.zeros(1024, dtype=np.float32)

        # FC layers: 1024 -> 512 -> 256 -> k*k
        self.fc1_w: np.ndarray = np.random.randn(1024, 512).astype(
            np.float32
        ) * np.sqrt(2.0 / 1024)
        self.fc1_b: np.ndarray = np.zeros(512, dtype=np.float32)

        self.fc2_w: np.ndarray = np.random.randn(512, 256).astype(np.float32) * np.sqrt(
            2.0 / 512
        )
        self.fc2_b: np.ndarray = np.zeros(256, dtype=np.float32)

        # Initialize to identity transform
        self.fc3_w: np.ndarray = np.zeros((256, k * k), dtype=np.float32)
        self.fc3_b: np.ndarray = np.eye(k, dtype=np.float32).ravel()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through T-Net: (B, N, K) -> (B, K, K) transform matrix."""
        b, n, k = x.shape
        h = np.maximum(0.0, x @ self.w1 + self.b1)
        h = np.maximum(0.0, h @ self.w2 + self.b2)
        h = np.maximum(0.0, h @ self.w3 + self.b3)

        global_feat = np.max(h, axis=1)

        h_fc = np.maximum(0.0, global_feat @ self.fc1_w + self.fc1_b)
        h_fc = np.maximum(0.0, h_fc @ self.fc2_w + self.fc2_b)
        transform = h_fc @ self.fc3_w + self.fc3_b

        return transform.reshape(b, k, k)


class PointNetClassifier:
    r"""PointNet 3D Point Cloud Classification Network.

    .. math::
        f(x_1, \dots, x_n) = \gamma\left( \max_{i=1,\dots,n} h(x_i) \right)

    Parameters
    ----------
    num_classes : int, default=40
        Number of output classification categories (e.g. ModelNet40).
    in_dim : int, default=3
        Input point coordinate dimensionality (usually 3 for (x, y, z)).
    global_dim : int, default=1024
        Dimensionality of the pooled global feature vector.
    """

    def __init__(
        self,
        num_classes: int = 40,
        in_dim: int = 3,
        global_dim: int = 1024,
    ) -> None:
        self.num_classes: int = num_classes
        self.in_dim: int = in_dim
        self.global_dim: int = global_dim

        self.input_tnet: TNet = TNet(k=in_dim)
        self.feature_tnet: TNet = TNet(k=64)

        # Shared MLPs: in_dim -> 64 -> 64
        self.mlp1_w: np.ndarray = np.random.randn(in_dim, 64).astype(
            np.float32
        ) * np.sqrt(2.0 / in_dim)
        self.mlp1_b: np.ndarray = np.zeros(64, dtype=np.float32)
        self.mlp2_w: np.ndarray = np.random.randn(64, 64).astype(np.float32) * np.sqrt(
            2.0 / 64
        )
        self.mlp2_b: np.ndarray = np.zeros(64, dtype=np.float32)

        # Shared MLPs: 64 -> 64 -> 128 -> global_dim
        self.mlp3_w: np.ndarray = np.random.randn(64, 64).astype(np.float32) * np.sqrt(
            2.0 / 64
        )
        self.mlp3_b: np.ndarray = np.zeros(64, dtype=np.float32)
        self.mlp4_w: np.ndarray = np.random.randn(64, 128).astype(np.float32) * np.sqrt(
            2.0 / 64
        )
        self.mlp4_b: np.ndarray = np.zeros(128, dtype=np.float32)
        self.mlp5_w: np.ndarray = np.random.randn(128, global_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / 128)
        self.mlp5_b: np.ndarray = np.zeros(global_dim, dtype=np.float32)

        # Classification Head: global_dim -> 512 -> 256 -> num_classes
        self.fc1_w: np.ndarray = np.random.randn(global_dim, 512).astype(
            np.float32
        ) * np.sqrt(2.0 / global_dim)
        self.fc1_b: np.ndarray = np.zeros(512, dtype=np.float32)
        self.fc2_w: np.ndarray = np.random.randn(512, 256).astype(np.float32) * np.sqrt(
            2.0 / 512
        )
        self.fc2_b: np.ndarray = np.zeros(256, dtype=np.float32)
        self.fc3_w: np.ndarray = np.random.randn(256, num_classes).astype(
            np.float32
        ) * np.sqrt(2.0 / 256)
        self.fc3_b: np.ndarray = np.zeros(num_classes, dtype=np.float32)

    def forward(self, points: np.ndarray) -> np.ndarray:
        """Forward pass for point cloud classification: (B, N, 3) -> (B, num_classes)."""
        x = np.asarray(points, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, ...]

        trans_input = self.input_tnet.forward(x)
        x_trans = np.matmul(x, trans_input)

        h = np.maximum(0.0, x_trans @ self.mlp1_w + self.mlp1_b)
        h = np.maximum(0.0, h @ self.mlp2_w + self.mlp2_b)

        trans_feat = self.feature_tnet.forward(h)
        h_trans = np.matmul(h, trans_feat)

        h2 = np.maximum(0.0, h_trans @ self.mlp3_w + self.mlp3_b)
        h2 = np.maximum(0.0, h2 @ self.mlp4_w + self.mlp4_b)
        h2 = np.maximum(0.0, h2 @ self.mlp5_w + self.mlp5_b)

        global_feat = np.max(h2, axis=1)

        out = np.maximum(0.0, global_feat @ self.fc1_w + self.fc1_b)
        out = np.maximum(0.0, out @ self.fc2_w + self.fc2_b)
        logits = out @ self.fc3_w + self.fc3_b

        return logits

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return self.forward(points)


class PointNetSegmenter:
    """PointNet Point Cloud Part / Semantic Segmentation Network.

    Concatenates per-point local features with global max-pooled features to predict per-point classes.
    """

    def __init__(
        self,
        num_classes: int = 50,
        in_dim: int = 3,
    ) -> None:
        self.num_classes: int = num_classes
        self.in_dim: int = in_dim

        self.w1: np.ndarray = np.random.randn(in_dim, 64).astype(np.float32) * np.sqrt(
            2.0 / in_dim
        )
        self.b1: np.ndarray = np.zeros(64, dtype=np.float32)
        self.w2: np.ndarray = np.random.randn(64, 128).astype(np.float32) * np.sqrt(
            2.0 / 64
        )
        self.b2: np.ndarray = np.zeros(128, dtype=np.float32)
        self.w3: np.ndarray = np.random.randn(128, 1024).astype(np.float32) * np.sqrt(
            2.0 / 128
        )
        self.b3: np.ndarray = np.zeros(1024, dtype=np.float32)

        self.seg1_w: np.ndarray = np.random.randn(1088, 512).astype(
            np.float32
        ) * np.sqrt(2.0 / 1088)
        self.seg1_b: np.ndarray = np.zeros(512, dtype=np.float32)
        self.seg2_w: np.ndarray = np.random.randn(512, 256).astype(
            np.float32
        ) * np.sqrt(2.0 / 512)
        self.seg2_b: np.ndarray = np.zeros(256, dtype=np.float32)
        self.seg3_w: np.ndarray = np.random.randn(256, 128).astype(
            np.float32
        ) * np.sqrt(2.0 / 256)
        self.seg3_b: np.ndarray = np.zeros(128, dtype=np.float32)
        self.seg4_w: np.ndarray = np.random.randn(128, num_classes).astype(
            np.float32
        ) * np.sqrt(2.0 / 128)
        self.seg4_b: np.ndarray = np.zeros(num_classes, dtype=np.float32)

    def forward(self, points: np.ndarray) -> np.ndarray:
        """Forward pass for point segmentation: (B, N, 3) -> (B, N, num_classes)."""
        x = np.asarray(points, dtype=np.float32)
        if x.ndim == 2:
            x = x[None, ...]
        b, n, _ = x.shape

        local_feat = np.maximum(0.0, x @ self.w1 + self.b1)
        h = np.maximum(0.0, local_feat @ self.w2 + self.b2)
        h = np.maximum(0.0, h @ self.w3 + self.b3)

        global_feat = np.max(h, axis=1, keepdims=True)
        global_tiled = np.repeat(global_feat, n, axis=1)

        concat_feat = np.concatenate([local_feat, global_tiled], axis=-1)

        out = np.maximum(0.0, concat_feat @ self.seg1_w + self.seg1_b)
        out = np.maximum(0.0, out @ self.seg2_w + self.seg2_b)
        out = np.maximum(0.0, out @ self.seg3_w + self.seg3_b)
        logits = out @ self.seg4_w + self.seg4_b

        return logits

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return self.forward(points)
