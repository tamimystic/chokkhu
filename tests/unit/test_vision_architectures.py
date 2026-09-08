from __future__ import annotations

import numpy as np
import chokkhu as ck
from chokkhu.core.tensor import Tensor
from chokkhu.models.vision import (
    AlexNet,
    LeNet5,
    MobileNetV1,
    ResNet18,
    UNet,
    VGG11,
)


def test_lenet5_forward_pass():
    np.random.seed(42)
    model = LeNet5(num_classes=10, in_channels=1)
    X = np.random.randn(4, 1, 28, 28)
    preds = model.predict(X)
    assert len(preds) == 4


def test_vgg11_and_alexnet_forward_pass():
    np.random.seed(42)
    vgg = VGG11(num_classes=5, in_channels=3)
    X = np.random.randn(2, 3, 32, 32)
    preds = vgg.predict(X)
    assert len(preds) == 2

    alex = AlexNet(num_classes=5, in_channels=3)
    X_alex = np.random.randn(2, 3, 224, 224)
    preds_alex = alex.predict(X_alex)
    assert len(preds_alex) == 2


def test_resnet18_forward_pass():
    np.random.seed(42)
    model = ResNet18(num_classes=5, in_channels=3)
    X = np.random.randn(2, 3, 32, 32)
    preds = model.predict(X)
    assert len(preds) == 2


def test_mobilenet_forward_pass():
    np.random.seed(42)
    model = MobileNetV1(num_classes=3, in_channels=3)
    X = np.random.randn(2, 3, 32, 32)
    preds = model.predict(X)
    assert len(preds) == 2


def test_unet_segmentation_forward():
    np.random.seed(42)
    unet = UNet(in_channels=3, out_channels=1)
    x = Tensor(np.random.randn(2, 3, 32, 32), requires_grad=True)
    out = unet(x)
    assert out.shape == (2, 1, 32, 32)


def test_train_vision_via_top_level_api():
    np.random.seed(42)
    X = np.random.randn(10, 1, 28, 28)
    y = np.random.choice([0, 1], size=10)

    # Train LeNet via chokkhu.train
    model = ck.train(
        model="lenet", X_train=X, y_train=y, epochs=2, batch_size=5, verbose=False
    )
    preds = model.predict(X[:2])
    assert len(preds) == 2
