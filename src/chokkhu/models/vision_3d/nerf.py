"""Neural Radiance Fields (NeRF) 3D Scene Synthesis and Volume Rendering.

Pure NumPy implementation of Mildenhall et al. (2020) "NeRF: Representing Scenes as Neural Radiance Fields":
- Positional Encoding with high-frequency sinusoidal harmonic basis
- NeRFMLP 8-layer Coordinate Network with skip connections
- Ray Marching and Numerical Quadrature Volume Rendering
"""

from typing import Optional, Tuple
import numpy as np


class PositionalEncoder:
    r"""Sinusoidal Harmonic Positional Encoder.

    .. math::
        \gamma(p) = \left( \sin(2^0 \pi p), \cos(2^0 \pi p), \dots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p) \right)

    Parameters
    ----------
    num_freqs : int, default=10
        Number of octave frequency bands :math:`L`.
    include_input : bool, default=True
        Whether to include raw coordinate inputs in output vector.
    """

    def __init__(self, num_freqs: int = 10, include_input: bool = True) -> None:
        self.num_freqs: int = num_freqs
        self.include_input: bool = include_input
        self.freq_bands: np.ndarray = 2.0 ** np.arange(num_freqs, dtype=np.float32)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Apply harmonic positional encoding to coordinate array."""
        arr = np.asarray(x, dtype=np.float32)
        encoded: list[np.ndarray] = []

        if self.include_input:
            encoded.append(arr)

        for freq in self.freq_bands:
            encoded.append(np.sin(arr * freq * np.pi))
            encoded.append(np.cos(arr * freq * np.pi))

        return np.concatenate(encoded, axis=-1)

    def out_dim(self, in_dim: int) -> int:
        """Return output feature dimension after encoding."""
        return (
            in_dim + (in_dim * self.num_freqs * 2)
            if self.include_input
            else in_dim * self.num_freqs * 2
        )


class NeRFMLP:
    r"""NeRF Coordinate-based Scene Representation MLP.

    Maps 3D coordinate :math:`(x, y, z)` and viewing direction :math:`(\theta, \phi)`
    to density :math:`\sigma` and RGB color :math:`c`.
    """

    def __init__(
        self,
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        self.pos_encoder: PositionalEncoder = PositionalEncoder(
            num_freqs=pos_freqs, include_input=True
        )
        self.dir_encoder: PositionalEncoder = PositionalEncoder(
            num_freqs=dir_freqs, include_input=True
        )

        in_pos_dim = self.pos_encoder.out_dim(3)
        in_dir_dim = self.dir_encoder.out_dim(3)

        self.w0: np.ndarray = (
            np.random.randn(in_pos_dim, hidden_dim).astype(np.float32) * 0.05
        )
        self.b0: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)

        self.w1: np.ndarray = (
            np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.05
        )
        self.b1: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)

        self.w2: np.ndarray = (
            np.random.randn(hidden_dim + in_pos_dim, hidden_dim).astype(np.float32)
            * 0.05
        )
        self.b2: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)

        self.sigma_w: np.ndarray = (
            np.random.randn(hidden_dim, 1).astype(np.float32) * 0.05
        )
        self.sigma_b: np.ndarray = np.zeros(1, dtype=np.float32)

        self.w_feat: np.ndarray = (
            np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * 0.05
        )
        self.b_feat: np.ndarray = np.zeros(hidden_dim, dtype=np.float32)

        self.w_rgb: np.ndarray = (
            np.random.randn(hidden_dim + in_dir_dim, 3).astype(np.float32) * 0.05
        )
        self.b_rgb: np.ndarray = np.zeros(3, dtype=np.float32)

    def forward(
        self,
        positions: np.ndarray,
        view_dirs: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate scene density and color."""
        pos = np.asarray(positions, dtype=np.float32)
        pos_enc = self.pos_encoder.encode(pos)

        h = np.maximum(0.0, pos_enc @ self.w0 + self.b0)
        h = np.maximum(0.0, h @ self.w1 + self.b1)
        h_skip = np.concatenate([h, pos_enc], axis=-1)
        h = np.maximum(0.0, h_skip @ self.w2 + self.b2)

        sigma = np.maximum(0.0, h @ self.sigma_w + self.sigma_b)

        h_feat = np.maximum(0.0, h @ self.w_feat + self.b_feat)
        if view_dirs is not None:
            dir_enc = self.dir_encoder.encode(np.asarray(view_dirs, dtype=np.float32))
            h_color = np.concatenate([h_feat, dir_enc], axis=-1)
        else:
            dummy_dir = np.zeros(
                (h.shape[0], self.dir_encoder.out_dim(3)), dtype=np.float32
            )
            h_color = np.concatenate([h_feat, dummy_dir], axis=-1)

        rgb_raw = h_color @ self.w_rgb + self.b_rgb
        rgb = 1.0 / (1.0 + np.exp(-np.clip(rgb_raw, -15.0, 15.0)))

        return rgb, sigma

    def __call__(
        self,
        positions: np.ndarray,
        view_dirs: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.forward(positions, view_dirs)


def volume_render(
    rgb: np.ndarray,
    sigma: np.ndarray,
    z_vals: np.ndarray,
    rays_d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Numerical Quadrature Volume Rendering along rays."""
    rgb_arr = np.asarray(rgb, dtype=np.float32)
    sig_arr = np.asarray(sigma, dtype=np.float32)
    if sig_arr.ndim == 3:
        sig_arr = sig_arr.squeeze(-1)
    z_arr = np.asarray(z_vals, dtype=np.float32)

    dists = z_arr[:, 1:] - z_arr[:, :-1]
    dist_last = np.full((dists.shape[0], 1), 1e10, dtype=np.float32)
    dists = np.concatenate([dists, dist_last], axis=-1)

    norm_d = np.linalg.norm(rays_d, axis=-1, keepdims=True)
    dists = dists * norm_d

    alpha = 1.0 - np.exp(-sig_arr * dists)

    cumprod = np.cumprod(1.0 - alpha + 1e-10, axis=-1)
    t_mat = np.ones_like(alpha)
    t_mat[:, 1:] = cumprod[:, :-1]

    weights = alpha * t_mat

    comp_rgb = np.sum(weights[..., None] * rgb_arr, axis=1)
    depth_map = np.sum(weights * z_arr, axis=-1)
    acc_map = np.sum(weights, axis=-1)

    return comp_rgb, depth_map, acc_map


def generate_camera_rays(
    height: int,
    width: int,
    focal_length: float,
    camera_pose: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate camera ray origins and directions for a pinhole camera."""
    i: np.ndarray
    j: np.ndarray
    i, j = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
        indexing="xy",
    )
    dirs = np.stack(
        [
            (i - width * 0.5) / focal_length,
            -(j - height * 0.5) / focal_length,
            -np.ones_like(i),
        ],
        axis=-1,
    )

    c2w = np.asarray(camera_pose, dtype=np.float32)
    rot = c2w[:3, :3]
    trans = c2w[:3, 3]

    rays_d = np.sum(dirs[..., None, :] * rot, axis=-1)
    rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-10)

    rays_o = np.broadcast_to(trans, rays_d.shape)

    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
