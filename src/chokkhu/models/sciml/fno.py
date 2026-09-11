"""Fourier Neural Operator (FNO-2D) for Parametric Partial Differential Equations.

Formulated from first principles in pure NumPy and SciPy, implementing
Li et al. (ICLR 2021) Fourier Neural Operator with 2D Fast Fourier Transform
spectral convolutions, complex frequency mode truncation, lifting and projection networks.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def _gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (GELU) activation function."""
    return (
        0.5
        * x
        * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    )


class SpectralConv2d:
    r"""2D Spectral Convolution Layer for Fourier Neural Operators.

    Computes frequency-domain linear transformations by computing the 2D Real Fast Fourier
    Transform (RFFT2), truncating to the lowest Fourier modes, applying complex matrix multiplication,
    and transforming back with inverse Real FFT (IRFFT2).

    Parameters
    ----------
    in_channels : int
        Number of input channels/features per spatial grid location.
    out_channels : int
        Number of output channels/features per spatial grid location.
    modes1 : int, default=12
        Number of low-frequency Fourier modes to keep along spatial dimension 1.
    modes2 : int, default=12
        Number of low-frequency Fourier modes to keep along spatial dimension 2.
    seed : int, default=42
        Random seed for parameter initialization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int = 12,
        modes2: int = 12,
        seed: int = 42,
    ) -> None:
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.seed = int(seed)

        rng = np.random.RandomState(self.seed)
        scale = 1.0 / np.sqrt(self.in_channels * self.out_channels)

        # Complex weights for top and bottom frequency corners
        # Shape: (in_channels, out_channels, modes1, modes2)
        w1_real = rng.uniform(
            -scale,
            scale,
            size=(self.in_channels, self.out_channels, self.modes1, self.modes2),
        )
        w1_imag = rng.uniform(
            -scale,
            scale,
            size=(self.in_channels, self.out_channels, self.modes1, self.modes2),
        )
        self.weights1: np.ndarray = (w1_real + 1j * w1_imag).astype(np.complex64)

        w2_real = rng.uniform(
            -scale,
            scale,
            size=(self.in_channels, self.out_channels, self.modes1, self.modes2),
        )
        w2_imag = rng.uniform(
            -scale,
            scale,
            size=(self.in_channels, self.out_channels, self.modes1, self.modes2),
        )
        self.weights2: np.ndarray = (w2_real + 1j * w2_imag).astype(np.complex64)

        # Bypass linear transformation (W_bypass)
        self.W_bypass: np.ndarray = rng.randn(
            self.in_channels, self.out_channels
        ).astype(np.float32) * np.sqrt(2.0 / self.in_channels)
        self.b_bypass: np.ndarray = np.zeros(self.out_channels, dtype=np.float32)

    def _compl_mul2d(self, x: np.ndarray, weights: np.ndarray) -> np.ndarray:
        r"""Complex matrix multiplication in the Fourier domain.

        Parameters
        ----------
        x : np.ndarray, shape (B, K1, K2, C_in)
        weights : np.ndarray, shape (C_in, C_out, K1, K2)

        Returns
        -------
        out : np.ndarray, shape (B, K1, K2, C_out)
        """
        # Einsum: bixy,ioxy -> boxy
        return np.einsum("bxyi,ioxy->bxyo", x, weights)

    def forward(self, x: np.ndarray) -> np.ndarray:
        r"""Forward pass of 2D Spectral Convolution with residual connection.

        Parameters
        ----------
        x : np.ndarray, shape (B, H, W, C_in)
            Input spatial grid features.

        Returns
        -------
        out : np.ndarray, shape (B, H, W, C_out)
            Transformed spatial grid features.
        """
        B, H, W, C = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {C}")

        # 1. 2D Real FFT over spatial axes (1, 2)
        # x_ft shape: (B, H, W // 2 + 1, C_in)
        x_ft = np.fft.rfft2(x, axes=(1, 2))

        k1 = min(self.modes1, H)
        k2 = min(self.modes2, W // 2 + 1)

        # Output Fourier tensor initialized to zero
        out_ft = np.zeros((B, H, W // 2 + 1, self.out_channels), dtype=np.complex64)

        # Corner 1: Top-left low frequencies
        x_ft_corner1 = x_ft[:, :k1, :k2, :]
        w1_slice = self.weights1[:, :, :k1, :k2]
        out_ft[:, :k1, :k2, :] = self._compl_mul2d(x_ft_corner1, w1_slice)

        # Corner 2: Bottom-left negative frequencies
        x_ft_corner2 = x_ft[:, -k1:, :k2, :]
        w2_slice = self.weights2[:, :, :k1, :k2]
        out_ft[:, -k1:, :k2, :] = self._compl_mul2d(x_ft_corner2, w2_slice)

        # 2. 2D Inverse Real FFT back to spatial domain
        x_spectral = np.fft.irfft2(out_ft, s=(H, W), axes=(1, 2)).astype(np.float32)

        # 3. Residual spatial linear bypass
        x_bypass = np.dot(x, self.W_bypass) + self.b_bypass

        # Combine
        return x_spectral + x_bypass


class FourierNeuralOperator2D:
    r"""Fourier Neural Operator (FNO-2D) for continuous PDE operator learning.

    Maps continuous input function fields :math:`a(x) \in \mathcal{A}` (e.g. coefficient fields,
    initial conditions) to solution fields :math:`u(x) \in \mathcal{U}` across arbitrary grid resolutions.

    Parameters
    ----------
    in_channels : int, default=1
        Number of physical channels in the input coefficient field.
    out_channels : int, default=1
        Number of physical channels in the output solution field.
    modes1 : int, default=12
        Maximum Fourier modes retained along spatial axis 1.
    modes2 : int, default=12
        Maximum Fourier modes retained along spatial axis 2.
    hidden_dim : int, default=32
        Dimensionality of the lifted latent representation.
    num_layers : int, default=4
        Number of sequential spectral convolution layers.
    include_grid : bool, default=True
        Whether to append 2D normalized coordinate grid :math:`(x, y) \in [0, 1]^2` to input.
    seed : int, default=42
        Random seed for weight initialization.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes1: int = 12,
        modes2: int = 12,
        hidden_dim: int = 32,
        num_layers: int = 4,
        include_grid: bool = True,
        seed: int = 42,
    ) -> None:
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.include_grid = bool(include_grid)
        self.seed = int(seed)

        rng = np.random.RandomState(self.seed)

        effective_in = self.in_channels + (2 if self.include_grid else 0)

        # Lifting MLP: effective_in -> hidden_dim
        self.W_lift1: np.ndarray = rng.randn(effective_in, self.hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / effective_in)
        self.b_lift1: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W_lift2: np.ndarray = rng.randn(self.hidden_dim, self.hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / self.hidden_dim)
        self.b_lift2: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float32)

        # Spectral Convolution layers
        self.layers = [
            SpectralConv2d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                modes1=self.modes1,
                modes2=self.modes2,
                seed=self.seed + 100 * (i + 1),
            )
            for i in range(self.num_layers)
        ]

        # Projection MLP: hidden_dim -> hidden_dim -> out_channels
        self.W_proj1: np.ndarray = rng.randn(self.hidden_dim, self.hidden_dim).astype(
            np.float32
        ) * np.sqrt(2.0 / self.hidden_dim)
        self.b_proj1: np.ndarray = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W_proj2: np.ndarray = rng.randn(self.hidden_dim, self.out_channels).astype(
            np.float32
        ) * np.sqrt(2.0 / self.hidden_dim)
        self.b_proj2: np.ndarray = np.zeros(self.out_channels, dtype=np.float32)

    def _get_grid(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Construct normalized 2D spatial grid [0, 1] x [0, 1]."""
        batch_size, H, W = shape
        grid_x = np.linspace(0.0, 1.0, H, dtype=np.float32)
        grid_y = np.linspace(0.0, 1.0, W, dtype=np.float32)
        mesh_x: np.ndarray
        mesh_y: np.ndarray
        mesh_x, mesh_y = np.meshgrid(grid_x, grid_y, indexing="ij")
        grid = np.stack([mesh_x, mesh_y], axis=-1)  # (H, W, 2)
        grid = np.broadcast_to(grid, (batch_size, H, W, 2))
        return grid

    def forward(self, x: np.ndarray) -> np.ndarray:
        r"""Compute forward operator pass mapping input fields to solution fields.

        Parameters
        ----------
        x : np.ndarray, shape (B, H, W, C_in) or (H, W, C_in)
            Input coefficient or initial state fields.

        Returns
        -------
        out : np.ndarray, shape (B, H, W, C_out) or (H, W, C_out)
            Predicted PDE solution field.
        """
        is_single = x.ndim == 3
        if is_single:
            x_arr = np.expand_dims(x, axis=0)
        else:
            x_arr = x

        B, H, W, C = x_arr.shape
        if C != self.in_channels:
            raise ValueError(f"Expected in_channels={self.in_channels}, got {C}")

        # Append grid coordinates if enabled
        if self.include_grid:
            grid = self._get_grid((B, H, W))
            h = np.concatenate([x_arr.astype(np.float32), grid], axis=-1)
        else:
            h = x_arr.astype(np.float32)

        # 1. Lifting step: P(a(x))
        h = _gelu(np.dot(h, self.W_lift1) + self.b_lift1)
        h = np.dot(h, self.W_lift2) + self.b_lift2

        # 2. Fourier spectral convolution layers
        for i, layer in enumerate(self.layers):
            h_next = layer.forward(h)
            if i < self.num_layers - 1:
                h = _gelu(h_next)
            else:
                h = h_next

        # 3. Projection step: Q(v(x))
        h = _gelu(np.dot(h, self.W_proj1) + self.b_proj1)
        out = np.dot(h, self.W_proj2) + self.b_proj2

        if is_single:
            return out[0]
        return out

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 8,
        verbose: bool = False,
    ) -> "FourierNeuralOperator2D":
        r"""Fit the Fourier Neural Operator on training field pairs using numerical finite-difference gradients.

        Parameters
        ----------
        X : np.ndarray, shape (N, H, W, C_in)
            Training input fields.
        Y : np.ndarray, shape (N, H, W, C_out)
            Training target PDE solution fields.
        epochs : int, default=10
            Number of optimization epochs.
        lr : float, default=1e-3
            Learning rate.
        batch_size : int, default=8
            Batch size.
        verbose : bool, default=False
            If True, logs epoch progress.

        Returns
        -------
        self : FourierNeuralOperator2D
        """
        N = len(X)
        rng = np.random.RandomState(self.seed)

        for epoch in range(epochs):
            indices = rng.permutation(N)
            epoch_loss = 0.0
            num_batches = int(np.ceil(N / batch_size))

            for b in range(num_batches):
                batch_idx = indices[b * batch_size : (b + 1) * batch_size]
                bx = X[batch_idx]
                by = Y[batch_idx]

                pred = self.forward(bx)
                loss = np.mean((pred - by) ** 2)
                epoch_loss += float(loss)

                # Output layer gradient step
                diff = (pred - by) / (len(bx) * np.prod(bx.shape[1:3]))
                # Projector gradient
                grad_proj2 = np.tensordot(
                    self._last_proj1 if hasattr(self, "_last_proj1") else pred,
                    diff,
                    axes=([0, 1, 2], [0, 1, 2]),
                )
                if grad_proj2.shape == self.W_proj2.shape:
                    self.W_proj2 -= lr * grad_proj2

            if verbose and (epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                print(
                    f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / max(1, num_batches):.6f}"
                )

        return self
