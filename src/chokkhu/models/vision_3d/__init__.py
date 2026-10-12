"""3D Computer Vision, Point Clouds & Neural Radiance (NeRF & 3DGS).

Pure NumPy implementations of:
- PointNet: PointNetClassifier, PointNetSegmenter, TNet
- PointNet++: PointNet2Classifier, SetAbstractionModule, farthest_point_sampling, ball_query
- NeRF: PositionalEncoder, NeRFMLP, volume_render, generate_camera_rays
- 3D Gaussian Splatting: GaussianSplatting3D, quaternion_to_rotation_matrix
"""

from .gaussian_splatting import (
    GaussianSplatting3D,
    quaternion_to_rotation_matrix,
)
from .nerf import (
    NeRFMLP,
    PositionalEncoder,
    generate_camera_rays,
    volume_render,
)
from .pointnet import (
    PointNetClassifier,
    PointNetSegmenter,
    TNet,
)
from .pointnet2 import (
    PointNet2Classifier,
    SetAbstractionModule,
    ball_query,
    farthest_point_sampling,
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
