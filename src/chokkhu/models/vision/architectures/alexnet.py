from __future__ import annotations

from ...dl.layers import Dropout, Flatten, Linear
from ...dl.activations import ReLU
from ...dl.sequential import Sequential
from ..conv_layers import Conv2D, MaxPool2D


def AlexNet(num_classes: int = 1000, in_channels: int = 3) -> Sequential:
    """AlexNet Architecture (2012)."""
    model = Sequential(task="classification")
    model.add(Conv2D(in_channels, 64, kernel_size=11, stride=4, padding=2))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=3, stride=2))
    model.add(Conv2D(64, 192, kernel_size=5, padding=2))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=3, stride=2))
    model.add(Conv2D(192, 384, kernel_size=3, padding=1))
    model.add(ReLU())
    model.add(Conv2D(384, 256, kernel_size=3, padding=1))
    model.add(ReLU())
    model.add(Conv2D(256, 256, kernel_size=3, padding=1))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=3, stride=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Linear(256 * 6 * 6, 4096))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Linear(4096, 4096))
    model.add(ReLU())
    model.add(Linear(4096, num_classes))
    return model


def ZFNet(num_classes: int = 1000, in_channels: int = 3) -> Sequential:
    """ZFNet Architecture (Zeiler & Fergus, 2013)."""
    model = Sequential(task="classification")
    model.add(Conv2D(in_channels, 96, kernel_size=7, stride=2, padding=1))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=3, stride=2))
    model.add(Conv2D(96, 256, kernel_size=5, stride=2, padding=0))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=3, stride=2))
    model.add(Conv2D(256, 384, kernel_size=3, padding=1))
    model.add(ReLU())
    model.add(Conv2D(384, 384, kernel_size=3, padding=1))
    model.add(ReLU())
    model.add(Conv2D(384, 256, kernel_size=3, padding=1))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=3, stride=2))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Linear(256 * 5 * 5, 4096))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Linear(4096, 4096))
    model.add(ReLU())
    model.add(Linear(4096, num_classes))
    return model
