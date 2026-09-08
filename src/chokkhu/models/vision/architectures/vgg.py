from __future__ import annotations

from ...dl.layers import Dropout, Flatten, Linear
from ...dl.activations import ReLU
from ...dl.sequential import Sequential
from ..conv_layers import Conv2D, MaxPool2D


def VGG11(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    """VGG-11 Architecture from Scratch."""
    model = Sequential(task="classification")
    # Block 1
    model.add(Conv2D(in_channels, 64, kernel_size=3, padding="same"))
    model.add(ReLU())
    model.add(MaxPool2D(2, 2))
    # Block 2
    model.add(Conv2D(64, 128, kernel_size=3, padding="same"))
    model.add(ReLU())
    model.add(MaxPool2D(2, 2))
    # Block 3
    model.add(Conv2D(128, 256, kernel_size=3, padding="same"))
    model.add(ReLU())
    model.add(Conv2D(256, 256, kernel_size=3, padding="same"))
    model.add(ReLU())
    model.add(MaxPool2D(2, 2))
    # Classifier
    model.add(Flatten())
    model.add(Linear(256 * 4 * 4, 512))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Linear(512, num_classes))
    return model


def VGG16(num_classes: int = 10, in_channels: int = 3) -> Sequential:
    """VGG-16 Architecture from Scratch."""
    model = Sequential(task="classification")
    # Block 1
    model.add(Conv2D(in_channels, 64, kernel_size=3, padding="same"))
    model.add(ReLU())
    model.add(Conv2D(64, 64, kernel_size=3, padding="same"))
    model.add(ReLU())
    model.add(MaxPool2D(2, 2))
    # Block 2
    model.add(Conv2D(64, 128, kernel_size=3, padding="same"))
    model.add(ReLU())
    model.add(Conv2D(128, 128, kernel_size=3, padding="same"))
    model.add(ReLU())
    model.add(MaxPool2D(2, 2))
    # Block 3
    model.add(Conv2D(128, 256, kernel_size=3, padding="same"))
    model.add(ReLU())
    model.add(Conv2D(256, 256, kernel_size=3, padding="same"))
    model.add(ReLU())
    model.add(Conv2D(256, 256, kernel_size=3, padding="same"))
    model.add(ReLU())
    model.add(MaxPool2D(2, 2))
    # Classifier
    model.add(Flatten())
    model.add(Linear(256 * 4 * 4, 512))
    model.add(ReLU())
    model.add(Dropout(0.5))
    model.add(Linear(512, num_classes))
    return model
