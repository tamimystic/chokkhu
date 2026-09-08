from __future__ import annotations

from ...dl.layers import Flatten, Linear
from ...dl.activations import ReLU
from ...dl.sequential import Sequential
from ..conv_layers import Conv2D, MaxPool2D


def LeNet5(num_classes: int = 10, in_channels: int = 1) -> Sequential:
    """LeNet-5 Classic Architecture."""
    model = Sequential(task="classification")
    model.add(
        Conv2D(in_channels=in_channels, out_channels=6, kernel_size=5, padding="same")
    )
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=2, stride=2))
    model.add(Conv2D(in_channels=6, out_channels=16, kernel_size=5, padding="valid"))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=2, stride=2))
    model.add(Flatten())
    model.add(Linear(in_features=16 * 5 * 5, out_features=120))
    model.add(ReLU())
    model.add(Linear(in_features=120, out_features=84))
    model.add(ReLU())
    model.add(Linear(in_features=84, out_features=num_classes))
    return model
