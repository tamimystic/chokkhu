import numpy as np
from chokkhu.models.vision import (
    LeNet5,
    GradCAM,
    compute_iou,
    non_maximum_suppression,
)


def test_iou_computation():
    box1 = np.array([0, 0, 10, 10])
    box2 = np.array([5, 5, 15, 15])
    iou = compute_iou(box1, box2)

    assert 0.14 < iou < 0.15


def test_non_maximum_suppression():
    boxes = np.array(
        [
            [0, 0, 10, 10],
            [1, 1, 10, 10],
            [50, 50, 70, 70],
        ]
    )
    scores = np.array([0.9, 0.8, 0.95])
    keep = non_maximum_suppression(boxes, scores, iou_threshold=0.5)

    assert 2 in keep
    assert 0 in keep
    assert 1 not in keep


def test_gradcam_generation():
    np.random.seed(42)
    model = LeNet5(num_classes=2, in_channels=1)
    gradcam = GradCAM(model, target_layer_idx=3)

    sample_img = np.random.randn(1, 28, 28)
    heatmap = gradcam.generate(sample_img, target_class=0)

    assert heatmap.shape == (28, 28)
    assert np.all(heatmap >= 0.0)
    assert np.all(heatmap <= 1.0)
