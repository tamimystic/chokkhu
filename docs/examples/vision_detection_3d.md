# 2D Object Detection & 3D Point Cloud Classification Tutorial

In this tutorial, you will train 2D and 3D vision models using Chokkhu's pure NumPy vision subsystem.

---

## 1. 3D Point Cloud Classification with PointNet

```python
import numpy as np
import chokkhu as ck
from chokkhu.models.vision.vision_3d import PointNetClassifier

# 1. Generate synthetic point cloud batch: 8 objects, 1024 (x,y,z) points each
batch_size = 8
n_points = 1024
point_clouds = np.random.randn(batch_size, n_points, 3).astype(np.float32)

# 2. Initialize PointNet for 10-class 3D shape classification
pointnet = PointNetClassifier(num_classes=10)

# 3. Forward pass
logits = pointnet(point_clouds)
predicted_classes = np.argmax(logits, axis=-1)

print(f"PointNet Classification Logits Shape: {logits.shape}")
print(f"Predicted Object Classes: {predicted_classes}")
```

---

## 2. 2D YOLO Object Detection Loss & Anchor Evaluation

```python
from chokkhu.models.vision.losses import FocalLoss
import numpy as np

# Focal Loss for hard negative mining in dense detection
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

# Simulated predicted classification probabilities vs ground truth
pred_probs = np.array([[0.95, 0.05], [0.30, 0.70], [0.80, 0.20]])
targets = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

loss = focal_loss(pred_probs, targets)
print(f"Focal Loss: {loss:.4f}")
```
