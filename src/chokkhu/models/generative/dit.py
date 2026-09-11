"""Diffusion Transformers (DiT) with Adaptive LayerNorm (adaLN-Zero) Modulation.

Formulated from first principles in pure NumPy and SciPy following Peebles & Xie (ICCV 2023).
Patchifies 2D continuous feature maps into visual tokens, performs modulated self-attention,
and implements reverse diffusion sampling.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def modulate(x: np.ndarray, shift: np.ndarray, scale: np.ndarray) -> np.ndarray:
    r"""Modulate normalized activations via affine transform: x * (1 + scale) + shift."""
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class AdaLNZero:
    r"""Adaptive Layer Normalization (adaLN-Zero) Modulation Layer.

    Regresses 6 dimension-wise modulation vectors from conditioning embedding:
    (\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2) = \text{Linear}(\text{SiLU}(c))

    where \alpha_1, \alpha_2 are initialized to zero so each DiT block acts as an identity function initially.
    """

    def __init__(self, cond_dim: int, hidden_dim: int, seed: int = 42) -> None:
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.rng = np.random.RandomState(seed)

        # Weight initialized with small values, bias for alpha set to 0
        self.W = self.rng.randn(cond_dim, 6 * hidden_dim) * 0.02
        self.b = np.zeros(6 * hidden_dim)

    def forward(
        self, cond_emb: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute modulation vectors."""
        # Activation: SiLU
        h = cond_emb / (1.0 + np.exp(-cond_emb))
        out = np.dot(h, self.W) + self.b  # (B, 6 * hidden_dim)

        d = self.hidden_dim
        gamma1 = out[:, 0 * d : 1 * d]
        beta1 = out[:, 1 * d : 2 * d]
        alpha1 = out[:, 2 * d : 3 * d]
        gamma2 = out[:, 3 * d : 4 * d]
        beta2 = out[:, 4 * d : 5 * d]
        alpha2 = out[:, 5 * d : 6 * d]

        return gamma1, beta1, alpha1, gamma2, beta2, alpha2


class DiTBlock:
    r"""Diffusion Transformer Block with adaLN-Zero Modulation and Multi-Head Attention.

    Parameters
    ----------
    hidden_dim : int
        Transformer hidden dimension.
    num_heads : int
        Number of self-attention heads.
    cond_dim : int
        Conditioning vector dimension (combining timestep + class/text embedding).
    seed : int, default=42
        Random seed.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        cond_dim: int = 64,
        seed: int = 42,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.cond_dim = cond_dim
        self.rng = np.random.RandomState(seed)

        self.adaln = AdaLNZero(cond_dim, hidden_dim, seed=seed)

        # Multi-head attention projections
        self.W_qkv = self.rng.randn(hidden_dim, 3 * hidden_dim) * 0.02
        self.b_qkv = np.zeros(3 * hidden_dim)
        self.W_out = self.rng.randn(hidden_dim, hidden_dim) * 0.02
        self.b_out = np.zeros(hidden_dim)

        # MLP projection
        mlp_dim = hidden_dim * 4
        self.W_mlp1 = self.rng.randn(hidden_dim, mlp_dim) * 0.02
        self.b_mlp1 = np.zeros(mlp_dim)
        self.W_mlp2 = self.rng.randn(mlp_dim, hidden_dim) * 0.02
        self.b_mlp2 = np.zeros(hidden_dim)

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def _attention(self, x: np.ndarray) -> np.ndarray:
        B, T, C = x.shape
        qkv = np.dot(x, self.W_qkv) + self.b_qkv
        q, k, v = np.split(qkv, 3, axis=-1)

        # Reshape to (B, num_heads, T, head_dim)
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / (
            np.sum(attn_weights, axis=-1, keepdims=True) + 1e-12
        )

        out = np.matmul(attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, C)
        return np.dot(out, self.W_out) + self.b_out

    def _mlp(self, x: np.ndarray) -> np.ndarray:
        h = np.dot(x, self.W_mlp1) + self.b_mlp1
        # GELU activation approximation
        h_gelu = 0.5 * h * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h + 0.044715 * h**3)))
        return np.dot(h_gelu, self.W_mlp2) + self.b_mlp2

    def forward(self, x: np.ndarray, cond_emb: np.ndarray) -> np.ndarray:
        """Forward pass of DiT block with adaLN-Zero modulation."""
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaln.forward(cond_emb)

        # 1. Modulated self-attention
        norm_x1 = self._layer_norm(x)
        mod_x1 = modulate(norm_x1, beta1, gamma1)
        attn_out = self._attention(mod_x1)
        x = x + alpha1[:, None, :] * attn_out

        # 2. Modulated MLP
        norm_x2 = self._layer_norm(x)
        mod_x2 = modulate(norm_x2, beta2, gamma2)
        mlp_out = self._mlp(mod_x2)
        x = x + alpha2[:, None, :] * mlp_out

        return x


class DiffusionTransformer:
    r"""Scaled Patch-Level Diffusion Transformer (DiT).

    Operates on 2D image/latent grids by converting them into patch tokens,
    conditioning on diffusion timesteps via sinusoidal positional encodings and adaLN-Zero.

    Parameters
    ----------
    input_size : int, default=16
        Spatial width and height of square input grid.
    in_channels : int, default=4
        Number of input channels (e.g. VAE latent channels).
    patch_size : int, default=2
        Spatial patch size $p \times p$.
    hidden_dim : int, default=64
        Transformer latent embedding width.
    depth : int, default=4
        Number of stacked DiT transformer blocks.
    num_heads : int, default=4
        Number of attention heads per block.
    num_timesteps : int, default=100
        Total forward diffusion noise timesteps $T$.
    seed : int, default=42
        Random seed.
    """

    def __init__(
        self,
        input_size: int = 16,
        in_channels: int = 4,
        patch_size: int = 2,
        hidden_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        num_timesteps: int = 100,
        seed: int = 42,
    ) -> None:
        self.input_size = input_size
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_timesteps = num_timesteps
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

        self.num_patches = (input_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size

        # 1. Patch projection
        self.W_patch = self.rng.randn(self.patch_dim, hidden_dim) * 0.02
        self.b_patch = np.zeros(hidden_dim)

        # 2. Positional embeddings
        self.pos_emb = self.rng.randn(self.num_patches, hidden_dim) * 0.02

        # 3. Timestep embedding MLP
        self.time_dim = hidden_dim
        self.W_time1 = self.rng.randn(self.time_dim, self.time_dim) * 0.02
        self.b_time1 = np.zeros(self.time_dim)
        self.W_time2 = self.rng.randn(self.time_dim, self.time_dim) * 0.02
        self.b_time2 = np.zeros(self.time_dim)

        # 4. DiT transformer blocks
        self.blocks = [
            DiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                cond_dim=hidden_dim,
                seed=seed + i,
            )
            for i in range(depth)
        ]

        # 5. Final projection back to patch dimensions
        self.W_final = self.rng.randn(hidden_dim, self.patch_dim) * 0.02
        self.b_final = np.zeros(self.patch_dim)

        # 6. Linear noise schedule
        self.betas = np.linspace(1e-4, 0.02, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)

    def _timestep_embedding(self, timesteps: np.ndarray) -> np.ndarray:
        """Compute sinusoidal timestep positional embeddings."""
        half_dim = self.time_dim // 2
        freqs = np.exp(-np.log(10000.0) * np.arange(half_dim) / half_dim)
        args = timesteps[:, None] * freqs[None, :]
        sin_emb = np.sin(args)
        cos_emb = np.cos(args)
        emb = np.concatenate([sin_emb, cos_emb], axis=-1)

        # MLP projection
        h = np.dot(emb, self.W_time1) + self.b_time1
        h = h / (1.0 + np.exp(-h))  # SiLU
        return np.dot(h, self.W_time2) + self.b_time2

    def _patchify(self, x: np.ndarray) -> np.ndarray:
        """Convert image grid (B, C, H, W) into patch sequence (B, N, patch_dim)."""
        B, C, H, W = x.shape
        p = self.patch_size
        gh, gw = H // p, W // p
        # Reshape to (B, C, gh, p, gw, p) -> transpose to (B, gh, gw, C, p, p)
        x_reshaped = x.reshape(B, C, gh, p, gw, p).transpose(0, 2, 4, 1, 3, 5)
        return x_reshaped.reshape(B, gh * gw, self.patch_dim)

    def _unpatchify(self, patches: np.ndarray) -> np.ndarray:
        """Convert patch sequence (B, N, patch_dim) back to image grid (B, C, H, W)."""
        B, N, _ = patches.shape
        p = self.patch_size
        C = self.in_channels
        gh = int(np.sqrt(N))
        gw = gh
        H, W = gh * p, gw * p
        x = patches.reshape(B, gh, gw, C, p, p).transpose(0, 3, 1, 4, 2, 5)
        return x.reshape(B, C, H, W)

    def forward(self, x: np.ndarray, timesteps: np.ndarray) -> np.ndarray:
        """Predict noise residual epsilon_theta(x_t, t).

        Parameters
        ----------
        x : np.ndarray of shape (B, C, H, W)
            Noisy latent image grid at timestep t.
        timesteps : np.ndarray of shape (B,)
            Integer diffusion timesteps in [0, T-1].

        Returns
        -------
        pred_noise : np.ndarray of shape (B, C, H, W)
            Predicted noise vector matching shape of input x.
        """
        # 1. Patchify input
        patches = self._patchify(x)  # (B, N, patch_dim)
        h = np.dot(patches, self.W_patch) + self.b_patch + self.pos_emb[None, :, :]

        # 2. Conditioning embedding
        cond_emb = self._timestep_embedding(timesteps)

        # 3. Stacked DiT blocks
        for block in self.blocks:
            h = block.forward(h, cond_emb)

        # 4. Final projection and unpatchify
        pred_patches = np.dot(h, self.W_final) + self.b_final
        return self._unpatchify(pred_patches)

    def q_sample(self, x_0: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward diffusion noise injection: q(x_t | x_0) = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * eps."""
        noise = self.rng.randn(*x_0.shape)
        alpha_t = self.alphas_cumprod[t]
        while alpha_t.ndim < x_0.ndim:
            alpha_t = alpha_t[..., None]

        mean = np.sqrt(alpha_t) * x_0
        std = np.sqrt(1.0 - alpha_t)
        x_t = mean + std * noise
        return x_t, noise

    def sample(
        self, shape: Tuple[int, ...], n_steps: Optional[int] = None
    ) -> np.ndarray:
        """Reverse diffusion sampling from pure Gaussian noise."""
        steps = (
            self.num_timesteps if n_steps is None else min(n_steps, self.num_timesteps)
        )
        x = self.rng.randn(*shape)

        for step in reversed(range(steps)):
            t: np.ndarray = np.full(shape[0], step, dtype=int)
            pred_noise = self.forward(x, t)

            alpha_t = self.alphas[step]
            alpha_cumprod_t = self.alphas_cumprod[step]
            beta_t = self.betas[step]

            if step > 0:
                z = self.rng.randn(*shape)
            else:
                z = np.zeros_like(x)

            # DDPM reverse step: x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1 - alpha_cumprod_t) * eps) + sigma_t * z
            mean = (1.0 / np.sqrt(alpha_t)) * (
                x - (beta_t / np.sqrt(1.0 - alpha_cumprod_t + 1e-12)) * pred_noise
            )
            sigma = np.sqrt(beta_t)
            x = mean + sigma * z

        return x
