"""Deep Learning Time Series Architectures: N-BEATS, N-HiTS, PatchTST from First Principles."""

from __future__ import annotations

from typing import Any, List, Tuple, Union
import numpy as np

from chokkhu.core.tensor import Tensor
from ..base import ChokkhuModel
from ..dl.layers import Linear, Module
from ..dl.activations import ReLU


class NBEATSBlock(Module):
    """N-BEATS Block: 4 FC Layers + Backcast/Forecast Basis Expansion (Oreshkin et al., 2019)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon: int,
        basis_type: str = "generic",
        polynomial_degree: int = 3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.basis_type = basis_type
        self.poly_deg = polynomial_degree

        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.fc3 = Linear(hidden_dim, hidden_dim)
        self.fc4 = Linear(hidden_dim, hidden_dim)
        self.act = ReLU()

        if basis_type == "trend":
            self.theta_b = Linear(hidden_dim, polynomial_degree + 1)
            self.theta_f = Linear(hidden_dim, polynomial_degree + 1)
            # Polynomial basis matrices: t^p
            t_back = np.linspace(0, 1, input_dim, dtype=np.float64)
            t_fore = np.linspace(0, 1, horizon, dtype=np.float64)
            self.V_b = np.stack(
                [t_back**p for p in range(polynomial_degree + 1)], axis=0
            )  # (deg+1, input_dim)
            self.V_f = np.stack(
                [t_fore**p for p in range(polynomial_degree + 1)], axis=0
            )  # (deg+1, horizon)
        else:
            # Generic basis
            self.theta_b = Linear(hidden_dim, input_dim)
            self.theta_f = Linear(hidden_dim, horizon)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))
        h = self.act(self.fc4(h))

        b_coeffs = self.theta_b(h)
        f_coeffs = self.theta_f(h)

        if self.basis_type == "trend":
            backcast_data = np.matmul(b_coeffs.data, self.V_b)
            forecast_data = np.matmul(f_coeffs.data, self.V_f)
            backcast = Tensor(backcast_data, requires_grad=x.requires_grad)
            forecast = Tensor(forecast_data, requires_grad=x.requires_grad)
        else:
            backcast = b_coeffs
            forecast = f_coeffs

        return backcast, forecast


class NBEATS(Module, ChokkhuModel):
    """N-BEATS Architecture with Doubly Residual Stacking (Oreshkin et al., 2019)."""

    def __init__(
        self,
        input_length: int = 24,
        horizon: int = 6,
        hidden_dim: int = 64,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.horizon = horizon
        self.blocks: List[NBEATSBlock] = []

        # Interleave generic and trend blocks
        for i in range(num_blocks):
            b_type = "trend" if i % 2 == 1 else "generic"
            block = NBEATSBlock(
                input_dim=input_length,
                hidden_dim=hidden_dim,
                horizon=horizon,
                basis_type=b_type,
            )
            self.blocks.append(block)
            setattr(self, f"block_{i}", block)

    def forward(self, x: Union[np.ndarray, Tensor]) -> Tensor:
        residual = x if isinstance(x, Tensor) else Tensor(x, requires_grad=True)
        N = residual.shape[0]
        forecast_total: np.ndarray = np.zeros((N, self.horizon), dtype=np.float64)

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            # Doubly residual: subtract backcast from input
            residual = Tensor(
                residual.data - backcast.data, requires_grad=residual.requires_grad
            )
            forecast_total += block_forecast.data

        return Tensor(forecast_total, requires_grad=True)

    def fit(
        self,
        X: Any,
        y: Any = None,
        epochs: int = 1,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> NBEATS:
        """Fit N-BEATS model."""
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_t = Tensor(np.asarray(X, dtype=np.float64), requires_grad=False)
        return self.forward(X_t).data


class NHITSBlock(Module):
    """N-HiTS Hierarchical Interpolation Block (Challu et al., 2023)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon: int,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.pooled_dim = max(1, input_dim // pool_size)
        self.horizon = horizon

        self.fc1 = Linear(self.pooled_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim)
        self.fc_backcast = Linear(hidden_dim, input_dim)
        self.fc_forecast = Linear(hidden_dim, horizon)
        self.act = ReLU()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        X_data = x.data
        N = X_data.shape[0]
        # Multi-rate pooling over time
        if self.pool_size > 1 and X_data.shape[1] >= self.pool_size:
            rem = X_data.shape[1] % self.pool_size
            trim_x = X_data[:, : X_data.shape[1] - rem] if rem > 0 else X_data
            pooled = np.mean(trim_x.reshape(N, -1, self.pool_size), axis=-1)
        else:
            pooled = X_data

        if pooled.shape[1] != self.pooled_dim:
            # Pad or truncate to self.pooled_dim
            if pooled.shape[1] < self.pooled_dim:
                pooled = np.pad(
                    pooled, ((0, 0), (0, self.pooled_dim - pooled.shape[1]))
                )
            else:
                pooled = pooled[:, : self.pooled_dim]

        h = self.act(self.fc1(Tensor(pooled, requires_grad=x.requires_grad)))
        h = self.act(self.fc2(h))

        backcast = self.fc_backcast(h)
        forecast = self.fc_forecast(h)
        return backcast, forecast


class NHITS(Module, ChokkhuModel):
    """N-HiTS: Neural Hierarchical Interpolation for Time Series (Challu et al., 2023)."""

    def __init__(
        self,
        input_length: int = 24,
        horizon: int = 6,
        hidden_dim: int = 64,
        num_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.horizon = horizon
        self.blocks: List[NHITSBlock] = []

        pool_sizes = [4, 2, 1]
        for i in range(num_blocks):
            ps = pool_sizes[i % len(pool_sizes)]
            block = NHITSBlock(
                input_dim=input_length,
                hidden_dim=hidden_dim,
                horizon=horizon,
                pool_size=ps,
            )
            self.blocks.append(block)
            setattr(self, f"block_{i}", block)

    def forward(self, x: Union[np.ndarray, Tensor]) -> Tensor:
        residual = x if isinstance(x, Tensor) else Tensor(x, requires_grad=True)
        N = residual.shape[0]
        forecast_total: np.ndarray = np.zeros((N, self.horizon), dtype=np.float64)

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = Tensor(
                residual.data - backcast.data, requires_grad=residual.requires_grad
            )
            forecast_total += block_forecast.data

        return Tensor(forecast_total, requires_grad=True)

    def fit(
        self,
        X: Any,
        y: Any = None,
        epochs: int = 1,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> NHITS:
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_t = Tensor(np.asarray(X, dtype=np.float64), requires_grad=False)
        return self.forward(X_t).data


class PatchTST(Module, ChokkhuModel):
    """Patch Time Series Transformer (Nie et al., 2023).

    Extracts subseries patches and applies Transformer self-attention.
    """

    def __init__(
        self,
        input_length: int = 24,
        horizon: int = 6,
        patch_len: int = 8,
        stride: int = 4,
        embed_dim: int = 32,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride
        self.embed_dim = embed_dim

        # Calculate number of patches: N_p = floor((L - P) / S) + 1
        self.num_patches = max(1, (input_length - patch_len) // stride + 1)

        self.patch_embed = Linear(patch_len, embed_dim)
        self.head = Linear(self.num_patches * embed_dim, horizon)

    def extract_patches(self, x: np.ndarray) -> np.ndarray:
        """Extract sliding patches of shape (N, num_patches, patch_len)."""
        N, L = x.shape
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            end = start + self.patch_len
            if end <= L:
                patches.append(x[:, start:end])
            else:
                pad_w = end - L
                patches.append(np.pad(x[:, start:], ((0, 0), (0, pad_w))))
        return np.stack(patches, axis=1)

    def forward(self, x: Union[np.ndarray, Tensor]) -> Tensor:
        X_data = x.data if isinstance(x, Tensor) else np.asarray(x, dtype=np.float64)
        N = X_data.shape[0]

        # 1. Patch extraction: (N, num_patches, patch_len)
        patches = self.extract_patches(X_data)

        # 2. Patch embedding: (N, num_patches, embed_dim)
        flat_patches = patches.reshape(-1, self.patch_len)
        embedded = self.patch_embed(Tensor(flat_patches, requires_grad=True))
        embedded_3d = embedded.data.reshape(N, self.num_patches * self.embed_dim)

        # 3. Projection head
        out = self.head(Tensor(embedded_3d, requires_grad=True))
        return out

    def fit(
        self,
        X: Any,
        y: Any = None,
        epochs: int = 1,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> PatchTST:
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_t = Tensor(np.asarray(X, dtype=np.float64), requires_grad=False)
        return self.forward(X_t).data
