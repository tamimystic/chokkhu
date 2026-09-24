# 3D Computer Vision, Point Clouds & NeRF / 3DGS API Reference

The `chokkhu.models.vision.vision_3d` module implements 3D geometric deep learning architectures, point cloud processing, Neural Radiance Fields (NeRF), and 3D Gaussian Splatting from scratch.

---

## 1. Point Cloud Architectures

### PointNet (`PointNetClassifier` / `PointNetSegmenter`)
Processes unordered 3D point sets $P \in \mathbb{R}^{N \times 3}$ using permutation-invariant symmetric pooling:

$$f(x_1, x_2, \dots, x_n) = \gamma \left( \max_{i=1 \dots n} \{ h(x_i) \} \right)$$

```python
import numpy as np
from chokkhu.models.vision.vision_3d import PointNetClassifier

# Batch of 4 point clouds with 1024 points each (B, N, 3)
point_clouds = np.random.randn(4, 1024, 3).astype(np.float32)

model = PointNetClassifier(num_classes=10)
logits = model(point_clouds)
print(f"Classification Logits Shape: {logits.shape}")  # (4, 10)
```

### PointNet++ (`PointNet2Classifier`)
Hierarchical feature learning on point sets using Farthest Point Sampling (FPS) and Ball Query grouping.

---

## 2. Neural Radiance Fields (NeRF)

NeRF models a continuous 3D volumetric scene representation mapping 5D coordinates $(x, y, z, \theta, \phi)$ to volume density $\sigma$ and emitted RGB radiance $c$:

$$F_\Theta: (x, y, z, \theta, \phi) \to (\text{RGB}, \sigma)$$

### Volumetric Rendering (`volume_render`)
Estimates camera ray radiance by numerically integrating color along sampled points:

$$\hat{C}(r) = \sum_{i=1}^N T_i (1 - \exp(-\sigma_i \delta_i)) c_i, \quad T_i = \exp\left( -\sum_{j=1}^{i-1} \sigma_j \delta_j \right)$$

```python
from chokkhu.models.vision.vision_3d import NeRFMLP, volume_render
import numpy as np

nerf = NeRFMLP(pos_dim=60, dir_dim=24, hidden_dim=128)
# Sample points along ray: rgb (N, 3), sigma (N, 1), z_vals (N,)
rgb = np.random.uniform(0, 1, size=(64, 3))
sigma = np.random.uniform(0, 2, size=(64, 1))
z_vals = np.linspace(2.0, 6.0, 64)

comp_rgb, depth, weights = volume_render(rgb, sigma, z_vals)
print(f"Rendered Pixel RGB: {comp_rgb}")
```

---

## 3. 3D Gaussian Splatting (`GaussianSplatting3D`)

Represents scenes as explicit 3D Gaussian distributions with covariance matrix $\Sigma$ projected onto 2D image coordinates:

$$\Sigma = R S S^T R^T, \quad \Sigma' = J W \Sigma W^T J^T$$

Where $R$ is rotation (quaternion), $S$ is 3D scaling vector, $W$ is viewing transformation, and $J$ is the Jacobian of the projective transformation.
