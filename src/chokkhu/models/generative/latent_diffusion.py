from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


class LatentCrossAttentionBlock:
    """Multi-Head Cross-Attention Block conditioning latent states on text/context."""

    def __init__(
        self,
        latent_dim: int = 32,
        context_dim: int = 64,
        num_heads: int = 4,
        seed: int = 42,
    ) -> None:
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads

        rng = np.random.default_rng(seed)
        scale_q = np.sqrt(2.0 / latent_dim)
        scale_kv = np.sqrt(2.0 / context_dim)

        self.w_q = rng.normal(0, scale_q, size=(latent_dim, latent_dim)).astype(
            np.float32
        )
        self.w_k = rng.normal(0, scale_kv, size=(context_dim, latent_dim)).astype(
            np.float32
        )
        self.w_v = rng.normal(0, scale_kv, size=(context_dim, latent_dim)).astype(
            np.float32
        )
        self.w_proj = rng.normal(0, scale_q, size=(latent_dim, latent_dim)).astype(
            np.float32
        )

    def forward(self, z: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Cross-attention: z is (B, N_z, latent_dim), context is (B, N_ctx, context_dim)."""
        B, N_z, D = z.shape
        _, N_ctx, _ = context.shape

        q = np.matmul(z, self.w_q)
        k = np.matmul(context, self.w_k)
        v = np.matmul(context, self.w_v)

        Q = q.reshape(B, N_z, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = k.reshape(B, N_ctx, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = v.reshape(B, N_ctx, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-12)

        out = np.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, N_z, D)
        return np.matmul(out, self.w_proj)


class LatentDiffusionModel:
    """Latent Diffusion Model (LDM) with Text Cross-Attention in pure NumPy.

    Operates in compressed VAE latent spaces, performing conditional denoising
    guided by text/context prompt representations.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        context_dim: int = 64,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        num_heads: int = 4,
        seed: int = 42,
    ) -> None:
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.num_timesteps = num_timesteps

        # Noise schedule
        self.betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)

        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / latent_dim)

        # Time-step embedding
        self.time_dim = 32
        self.w_time = rng.normal(
            0, np.sqrt(2.0 / self.time_dim), size=(self.time_dim, latent_dim)
        ).astype(np.float32)

        # Denoiser MLP layers
        self.w1 = rng.normal(0, scale, size=(latent_dim, latent_dim * 2)).astype(
            np.float32
        )
        self.w2 = rng.normal(
            0, np.sqrt(2.0 / (latent_dim * 2)), size=(latent_dim * 2, latent_dim)
        ).astype(np.float32)

        # Cross-Attention block
        self.cross_attn = LatentCrossAttentionBlock(
            latent_dim=latent_dim,
            context_dim=context_dim,
            num_heads=num_heads,
            seed=seed,
        )

    def _time_embedding(self, timesteps: np.ndarray) -> np.ndarray:
        """Sinusoidal positional encoding for diffusion timesteps."""
        t = np.asarray(timesteps, dtype=np.float32).flatten()
        half_dim = self.time_dim // 2
        emb_scale = np.log(10000.0) / (half_dim - 1)
        freqs = np.exp(-np.arange(half_dim, dtype=np.float32) * emb_scale)
        args = t[:, np.newaxis] * freqs[np.newaxis, :]
        emb = np.concatenate([np.sin(args), np.cos(args)], axis=-1)
        return np.matmul(emb, self.w_time)

    def q_sample(
        self,
        z_0: np.ndarray,
        t: np.ndarray,
        noise: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward diffusion: adds calibrated Gaussian noise to clean latents z_0."""
        z = np.asarray(z_0, dtype=np.float32)
        if noise is None:
            noise = np.random.randn(*z.shape).astype(np.float32)

        t_idx = np.clip(np.asarray(t, dtype=int), 0, self.num_timesteps - 1)
        alpha_bar = self.alphas_cumprod[t_idx]

        # Reshape alpha_bar for broadcasting
        while alpha_bar.ndim < z.ndim:
            alpha_bar = alpha_bar[..., np.newaxis]

        z_t = np.sqrt(alpha_bar) * z + np.sqrt(1.0 - alpha_bar) * noise
        return z_t, noise

    def predict_noise(
        self,
        z_t: np.ndarray,
        t: np.ndarray,
        context: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predicts noise epsilon added at timestep t conditioned on context."""
        z = np.asarray(z_t, dtype=np.float32)
        is_1d = z.ndim == 2
        if is_1d:
            # (B, D) -> (B, 1, D)
            z = z[:, np.newaxis, :]

        B, N_z, D = z.shape
        t_emb = self._time_embedding(t)  # (B, D)
        t_emb_exp = t_emb[:, np.newaxis, :]  # (B, 1, D)

        # 1. Self-denoising step + time condition
        h = z + t_emb_exp
        h_mlp = np.maximum(0.0, np.matmul(h, self.w1))
        h = h + np.matmul(h_mlp, self.w2)

        # 2. Cross-Attention context condition
        if context is not None:
            ctx = np.asarray(context, dtype=np.float32)
            if ctx.ndim == 2:
                ctx = ctx[:, np.newaxis, :]
            h_attn = self.cross_attn.forward(h, ctx)
            h = h + h_attn

        if is_1d:
            return h[:, 0, :]
        return h

    def sample(
        self,
        shape: Tuple[int, ...],
        context: Optional[np.ndarray] = None,
        n_steps: int = 20,
    ) -> np.ndarray:
        """Generates clean latent samples from pure Gaussian noise using DDIM sampler."""
        z = np.random.randn(*shape).astype(np.float32)
        B = shape[0]

        timesteps = np.linspace(self.num_timesteps - 1, 0, n_steps).astype(int)

        for i, t in enumerate(timesteps):
            t_batch: np.ndarray = np.full(B, t, dtype=int)
            pred_noise = self.predict_noise(z, t_batch, context=context)

            alpha_bar_t = self.alphas_cumprod[t]
            prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            alpha_bar_prev = self.alphas_cumprod[prev_t]

            # Predicted clean latent z_0
            pred_z0 = (z - np.sqrt(1.0 - alpha_bar_t) * pred_noise) / (
                np.sqrt(alpha_bar_t) + 1e-12
            )

            # DDIM deterministic update direction
            dir_z = np.sqrt(1.0 - alpha_bar_prev) * pred_noise
            z = np.sqrt(alpha_bar_prev) * pred_z0 + dir_z

        return z
