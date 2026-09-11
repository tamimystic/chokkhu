"""Guided Diffusion Models and Differentiable Inpainting Engines.

Formulated from first principles using Classifier-Free Guidance (CFG) score interpolation
and RePaint time-travel masked diffusion inpainting in pure NumPy.
"""

from typing import Any, Optional

import numpy as np


class ClassifierFreeGuidance:
    r"""Classifier-Free Guidance (CFG) Score Interpolator.

    Extrapolates conditional and unconditional diffusion score estimates:
        ilde{\epsilon}_	heta(x_t, c) = \epsilon_	heta(x_t, \emptyset) + s * (\epsilon_	heta(x_t, c) - \epsilon_	heta(x_t, \emptyset))

    Parameters
    ----------
    guidance_scale : float, default=3.0
        Guidance scale multiplier s. (s=1.0 is standard conditional, s > 1.0 boosts conditioning).
    """

    def __init__(self, guidance_scale: float = 3.0) -> None:
        self.guidance_scale = float(guidance_scale)

    def combine_scores(
        self,
        uncond_score: np.ndarray,
        cond_score: np.ndarray,
        scale: Optional[float] = None,
    ) -> np.ndarray:
        """Combine unconditional and conditional noise estimates."""
        s = self.guidance_scale if scale is None else float(scale)
        return uncond_score + s * (cond_score - uncond_score)


class DiffusionInpainter:
    """RePaint Masked Diffusion Inpainting and Inverse Problem Restoration.

    Synchronizes forward diffusion on unmasked clean pixels with reverse denoising
    in corrupted/masked regions with boundary harmonization time-travel steps.

    Parameters
    ----------
    ddpm_model : Any
        Base DDPM or diffusion model instance providing noise schedules (alphas_cumprod, etc.).
    num_resample_steps : int, default=1
        Time-travel resampling passes per step for boundary consistency.
    seed : int, default=42
        Random seed.
    """

    def __init__(
        self,
        ddpm_model: Any,
        num_resample_steps: int = 1,
        seed: int = 42,
    ) -> None:
        self.model = ddpm_model
        self.num_resample_steps = max(1, int(num_resample_steps))
        self.seed = int(seed)
        self.rng = np.random.RandomState(self.seed)

    def inpaint(
        self,
        image_clean: np.ndarray,
        mask: np.ndarray,
        num_steps: Optional[int] = None,
    ) -> np.ndarray:
        """Inpaint masked regions of input data.

        Parameters
        ----------
        image_clean : np.ndarray
            Clean ground-truth input data with unmasked regions intact.
        mask : np.ndarray
            Binary mask where 1.0 denotes known preserved regions and 0.0 denotes inpaint holes.
        num_steps : Optional[int], default=None
            Number of reverse diffusion timesteps.

        Returns
        -------
        inpainted_image : np.ndarray
            Harmoniously completed image / sequence.
        """
        x_orig = np.asarray(image_clean, dtype=float)
        m = np.asarray(mask, dtype=float)

        total_timesteps = getattr(self.model, "num_timesteps", 50)
        n_steps = (
            total_timesteps
            if num_steps is None
            else min(total_timesteps, int(num_steps))
        )

        alphas_cumprod = getattr(
            self.model, "alphas_cumprod", np.linspace(0.99, 0.01, total_timesteps)
        )

        # Start from pure Gaussian noise
        x_t = self.rng.randn(*x_orig.shape)
        timesteps = np.linspace(total_timesteps - 1, 0, n_steps, dtype=int)

        for s_idx, t_val in enumerate(timesteps):
            alpha_t = alphas_cumprod[t_val]
            alpha_prev = (
                alphas_cumprod[timesteps[s_idx + 1]]
                if s_idx + 1 < len(timesteps)
                else 1.0
            )

            for _ in range(self.num_resample_steps):
                # 1. Forward known region noise injection: q(x_{t-1} | x_0)
                noise_known = self.rng.randn(*x_orig.shape)
                x_known_prev = (
                    np.sqrt(alpha_prev) * x_orig
                    + np.sqrt(1.0 - alpha_prev) * noise_known
                )

                # 2. Reverse unknown region denoising: predict epsilon
                t_arr = (
                    np.full((x_orig.shape[0],), t_val, dtype=int)
                    if x_orig.ndim > 1
                    else np.array([t_val])
                )
                if hasattr(self.model, "predict_noise"):
                    pred_eps = self.model.predict_noise(x_t, t_arr)
                elif hasattr(self.model, "forward"):
                    pred_eps = self.model.forward(x_t, t_arr)
                else:
                    pred_eps = np.zeros_like(x_t)

                # Denoising mean estimate: x_0_hat = (x_t - sqrt(1 - alpha_t) * eps) / sqrt(alpha_t)
                x_0_hat = (x_t - np.sqrt(1.0 - alpha_t) * pred_eps) / (
                    np.sqrt(alpha_t) + 1e-12
                )

                # Step to t-1
                noise_unknown = (
                    self.rng.randn(*x_orig.shape)
                    if s_idx + 1 < len(timesteps)
                    else np.zeros_like(x_orig)
                )
                beta_t = 1.0 - (alpha_t / (alpha_prev + 1e-12))
                x_unknown_prev = (
                    np.sqrt(alpha_prev) * x_0_hat
                    + np.sqrt(np.maximum(0.0, beta_t)) * noise_unknown
                )

                # 3. Fuse known and unknown regions
                x_prev = m * x_known_prev + (1.0 - m) * x_unknown_prev

                # 4. Time-travel resampling forward if needed
                if self.num_resample_steps > 1 and s_idx + 1 < len(timesteps):
                    # Add noise back from t-1 to t
                    x_t = np.sqrt(1.0 - beta_t) * x_prev + np.sqrt(
                        beta_t
                    ) * self.rng.randn(*x_orig.shape)
                else:
                    x_t = x_prev

        # Final blend
        final_output = m * x_orig + (1.0 - m) * x_t
        return final_output
