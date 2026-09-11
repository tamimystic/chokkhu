"""CycleGAN: Unpaired Image-to-Image Translation with Cycle Consistency in pure NumPy."""

from __future__ import annotations

from typing import Dict
import numpy as np


class CycleGANGenerator:
    """ResNet-based Generator for CycleGAN translation (A -> B or B -> A)."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 16,
        seed: int = 42,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels

        rng = np.random.default_rng(seed)
        scale_in = np.sqrt(2.0 / (in_channels * 9))
        scale_mid = np.sqrt(2.0 / (base_channels * 9))
        scale_out = np.sqrt(2.0 / (base_channels * 9))

        # Initial stem
        self.w_stem = rng.normal(
            0, scale_in, size=(base_channels, in_channels, 3, 3)
        ).astype(np.float32)

        # ResNet residual transformation block
        self.w_res1 = rng.normal(
            0, scale_mid, size=(base_channels, base_channels, 3, 3)
        ).astype(np.float32)
        self.w_res2 = rng.normal(
            0, scale_mid, size=(base_channels, base_channels, 3, 3)
        ).astype(np.float32)

        # Output projection with tanh activation
        self.w_out = rng.normal(
            0, scale_out, size=(out_channels, base_channels, 3, 3)
        ).astype(np.float32)

    def _conv(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        B, _, H, W = x.shape
        C_out = weight.shape[0]
        x_pad = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
        out = np.zeros((B, C_out, H, W), dtype=np.float32)

        for i in range(H):
            for j in range(W):
                patch = x_pad[:, :, i : i + 3, j : j + 3]
                out[:, :, i, j] = np.tensordot(
                    patch, weight, axes=((1, 2, 3), (1, 2, 3))
                )
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Translates image across domains (B, in_channels, H, W) -> (B, out_channels, H, W)."""
        # Stem
        h = np.maximum(0.0, self._conv(x, self.w_stem))

        # Residual block with skip connection
        h_res = np.maximum(0.0, self._conv(h, self.w_res1))
        h_res = self._conv(h_res, self.w_res2)
        h = h + h_res

        # Output image in [-1, 1] range
        out = np.tanh(self._conv(h, self.w_out))
        return out


class PatchGANDiscriminator:
    """70x70 PatchGAN Discriminator evaluating realism of local image patches."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 16,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / (in_channels * 9))
        scale2 = np.sqrt(2.0 / (base_channels * 9))

        self.w1 = rng.normal(0, scale1, size=(base_channels, in_channels, 3, 3)).astype(
            np.float32
        )
        self.w2 = rng.normal(0, scale2, size=(1, base_channels, 3, 3)).astype(
            np.float32
        )

    def _conv(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        B, _, H, W = x.shape
        C_out = weight.shape[0]
        x_pad = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
        out = np.zeros((B, C_out, H, W), dtype=np.float32)

        for i in range(H):
            for j in range(W):
                patch = x_pad[:, :, i : i + 3, j : j + 3]
                out[:, :, i, j] = np.tensordot(
                    patch, weight, axes=((1, 2, 3), (1, 2, 3))
                )
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Computes patch validity logits of shape (B, 1, H, W)."""
        h = np.maximum(
            0.2 * self._conv(x, self.w1), self._conv(x, self.w1)
        )  # LeakyReLU 0.2
        logits = self._conv(h, self.w2)
        return logits


class CycleGAN:
    """Cycle-Consistent Generative Adversarial Network (CycleGAN) in pure NumPy.

    Learns unpaired bidirectional domain mapping G_AB: A -> B and G_BA: B -> A
    enforcing cycle consistency: G_BA(G_AB(a)) ~ a and G_AB(G_BA(b)) ~ b.
    """

    def __init__(
        self,
        channels_a: int = 3,
        channels_b: int = 3,
        base_channels: int = 16,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
        seed: int = 42,
    ) -> None:
        self.lambda_cycle = float(lambda_cycle)
        self.lambda_identity = float(lambda_identity)

        # Dual Generators
        self.gen_ab = CycleGANGenerator(
            in_channels=channels_a,
            out_channels=channels_b,
            base_channels=base_channels,
            seed=seed,
        )
        self.gen_ba = CycleGANGenerator(
            in_channels=channels_b,
            out_channels=channels_a,
            base_channels=base_channels,
            seed=seed + 1,
        )

        # Dual Discriminators
        self.disc_a = PatchGANDiscriminator(
            in_channels=channels_a, base_channels=base_channels, seed=seed + 2
        )
        self.disc_b = PatchGANDiscriminator(
            in_channels=channels_b, base_channels=base_channels, seed=seed + 3
        )

    def translate_a2b(self, real_a: np.ndarray) -> np.ndarray:
        """Translates domain A images to domain B."""
        return self.gen_ab.forward(real_a)

    def translate_b2a(self, real_b: np.ndarray) -> np.ndarray:
        """Translates domain B images to domain A."""
        return self.gen_ba.forward(real_b)

    def forward(
        self,
        real_a: np.ndarray,
        real_b: np.ndarray,
    ) -> Dict[str, np.ndarray | float]:
        """Full forward pass generating translations and computing cycle consistency losses."""
        # 1. Forward domain mappings
        fake_b = self.gen_ab.forward(real_a)
        recovered_a = self.gen_ba.forward(fake_b)

        fake_a = self.gen_ba.forward(real_b)
        recovered_b = self.gen_ab.forward(fake_a)

        # 2. Identity mapping
        identity_b = self.gen_ab.forward(real_b)
        identity_a = self.gen_ba.forward(real_a)

        # 3. Cycle Consistency Loss (L1 norm)
        loss_cycle_a = float(np.mean(np.abs(recovered_a - real_a)))
        loss_cycle_b = float(np.mean(np.abs(recovered_b - real_b)))
        total_cycle_loss = self.lambda_cycle * (loss_cycle_a + loss_cycle_b)

        # 4. Identity Loss
        loss_idt_a = float(np.mean(np.abs(identity_a - real_a)))
        loss_idt_b = float(np.mean(np.abs(identity_b - real_b)))
        total_identity_loss = self.lambda_identity * (loss_idt_a + loss_idt_b)

        # 5. Discriminator Realism Evaluations
        pred_fake_b = self.disc_b.forward(fake_b)
        pred_fake_a = self.disc_a.forward(fake_a)
        loss_gan = float(
            np.mean((pred_fake_b - 1.0) ** 2) + np.mean((pred_fake_a - 1.0) ** 2)
        )

        total_gen_loss = loss_gan + total_cycle_loss + total_identity_loss

        return {
            "fake_b": fake_b,
            "fake_a": fake_a,
            "recovered_a": recovered_a,
            "recovered_b": recovered_b,
            "loss_cycle": total_cycle_loss,
            "loss_identity": total_identity_loss,
            "loss_gan": loss_gan,
            "total_loss": total_gen_loss,
        }
