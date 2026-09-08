from __future__ import annotations

from typing import Optional
import cv2
import numpy as np
from chokkhu.core.tensor import Tensor
from ..dl.sequential import Sequential


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) from scratch."""

    def __init__(self, model: Sequential, target_layer_idx: int = -4) -> None:
        self.model = model
        self.target_layer_idx = target_layer_idx

    def generate(
        self, input_image: np.ndarray, target_class: Optional[int] = None
    ) -> np.ndarray:
        # input_image shape: (1, C, H, W) or (C, H, W)
        if input_image.ndim == 3:
            input_image = input_image[np.newaxis, :, :, :]

        x_t = Tensor(input_image, requires_grad=True)

        activations = None
        current = x_t
        for i, layer in enumerate(self.model.layers_list):
            current = layer(current)
            if (
                i == len(self.model.layers_list) + self.target_layer_idx
                or i == self.target_layer_idx
            ):
                activations = current

        out = current.data
        if target_class is None:
            target_class = int(np.argmax(out[0]))

        grad = np.zeros_like(current.data)
        grad[0, target_class] = 1.0
        current.backward(grad)

        if activations is not None and activations.grad is not None:
            weights = np.mean(activations.grad, axis=(2, 3), keepdims=True)
            cam = np.sum(weights * activations.data, axis=1, keepdims=True)
            cam = np.maximum(0.0, cam)[0, 0]
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
            H, W = input_image.shape[2], input_image.shape[3]
            cam_resized = cv2.resize(cam, (W, H))
            return cam_resized
        else:
            return np.ones(
                (input_image.shape[2], input_image.shape[3]), dtype=np.float64
            )
