"""3D Computer Vision, Point Clouds & Neural Radiance (NeRF & 3DGS).

Pure NumPy implementations of:
- PointNet: PointNetClassifier, PointNetSegmenter, TNet
- PointNet++: PointNet2Classifier, SetAbstractionModule, farthest_point_sampling, ball_query
- NeRF: PositionalEncoder, NeRFMLP, volume_render, generate_camera_rays
- 3D Gaussian Splatting: GaussianSplatting3D, quaternion_to_rotation_matrix
"""

from .pointnet import (
    TNet,
    PointNetClassifier,
    PointNetSegmenter,
)
from .pointnet2 import (
    PointNet2Classifier,
    SetAbstractionModule,
    farthest_point_sampling,
    ball_query,
)
from .nerf import (
    PositionalEncoder,
    NeRFMLP,
    volume_render,
    generate_camera_rays,
)
from .gaussian_splatting import (
    GaussianSplatting3D,
    quaternion_to_rotation_matrix,
)

__all__ = [
    "TNet",
    "PointNetClassifier",
    "PointNetSegmenter",
    "PointNet2Classifier",
    "SetAbstractionModule",
    "farthest_point_sampling",
    "ball_query",
    "PositionalEncoder",
    "NeRFMLP",
    "volume_render",
    "generate_camera_rays",
    "GaussianSplatting3D",
    "quaternion_to_rotation_matrix",
]
