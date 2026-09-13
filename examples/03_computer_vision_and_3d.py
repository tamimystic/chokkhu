"""Example 3: 2D Computer Vision (ViT, SAM, YOLO) & 3D Point Clouds / NeRF."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chokkhu.core.tensor import Tensor
from chokkhu.models.vision import (
    SegmentAnythingModel,
    VisionTransformer,
    YOLOHead,
)
from chokkhu.models.vision_3d import PointNetClassifier, volume_render


def main():
    print("=" * 70)
    print("  CHOKKHU EXAMPLE 3: 2D VISION, DETECTION, SAM & 3D POINT CLOUDS")
    print("=" * 70)

    # 1. Vision Transformer (ViT)
    print("[1] Vision Transformer (ViT) Forward Pass:")
    vit = VisionTransformer(
        img_size=32,
        patch_size=8,
        in_channels=3,
        num_classes=10,
        embed_dim=64,
        depth=2,
        num_heads=4,
    )
    img = Tensor(np.random.randn(2, 3, 32, 32))
    logits = vit(img)
    print(f"  Input image batch shape: {img.shape}")
    print(f"  ViT class logits shape : {logits.shape}")

    # 2. Segment Anything Model (SAM) Zero-Shot Mask Projection
    print("\n[2] Segment Anything Model (SAM):")
    sam = SegmentAnythingModel(
        embed_dim=64,
        num_heads=4,
        num_mask_tokens=3,
    )
    image_embeddings = np.random.randn(1, 64, 16, 16).astype(np.float32)
    masks, iou_scores = sam.forward(image_embeddings)
    print(f"  SAM predicted masks shape : {masks.shape}")
    print(f"  SAM predicted IoU scores  : {np.round(iou_scores, 3)}")

    # 3. YOLO Object Detection Head Decoding
    print("\n[3] YOLO Object Detection Anchor Head:")
    yolo = YOLOHead(in_channels=16, num_classes=4, num_anchors=3)
    feat_map = Tensor(np.random.randn(1, 16, 8, 8))
    detections = yolo(feat_map)
    print(f"  YOLO decoded detection shape: {detections.shape}")

    # 4. 3D Point Cloud Classification (PointNet)
    print("\n[4] 3D Point Cloud Processing (PointNet):")
    pointnet = PointNetClassifier(num_classes=5)
    point_cloud = np.random.randn(2, 128, 3).astype(np.float32)
    pred_3d = pointnet.forward(point_cloud)
    print(f"  Point cloud batch shape: {point_cloud.shape}")
    print(f"  Predicted 3D classes   : {pred_3d.shape}")

    # 5. NeRF Volumetric Ray Rendering
    print("\n[5] NeRF Volumetric Ray Rendering:")
    densities = np.random.uniform(0.1, 1.0, size=(10, 32)).astype(np.float32)
    colors = np.random.uniform(0.0, 1.0, size=(10, 32, 3)).astype(np.float32)
    z_vals = np.repeat(np.linspace(1.0, 5.0, 32)[None, :], 10, axis=0).astype(
        np.float32
    )
    rays_d = np.ones((10, 3), dtype=np.float32)
    rendered_rgb, depth_map, acc_map = volume_render(colors, densities, z_vals, rays_d)
    print(f"  Volumetric rendered image pixels shape: {rendered_rgb.shape}")
    print("=" * 70)


if __name__ == "__main__":
    main()
