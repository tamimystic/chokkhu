"""DeiT (Data-efficient Image Transformers) with Distillation Token from Scratch (Touvron et al., 2021)."""

from __future__ import annotations

from typing import Any, Tuple, Union
import numpy as np

from chokkhu.core.tensor import Tensor
from ...base import ChokkhuModel
from ...dl.layers import LayerNorm, Linear, Module, Parameter
from .vit import PatchEmbedding, ViTBlock


class DeiT(Module, ChokkhuModel):
    """Data-efficient Image Transformer (DeiT) with Class & Distillation Tokens."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 64,
        depth: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = Parameter(np.zeros((1, 1, embed_dim)))
        self.dist_token = Parameter(np.zeros((1, 1, embed_dim)))
        self.pos_embed = Parameter(
            np.random.randn(1, num_patches + 2, embed_dim) * 0.02
        )

        self.blocks = [
            ViTBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(depth)
        ]
        for i, b in enumerate(self.blocks):
            setattr(self, f"block_{i}", b)

        self.norm = LayerNorm(embed_dim)
        self.head_cls = Linear(embed_dim, num_classes)
        self.head_dist = Linear(embed_dim, num_classes)

    def forward(
        self, x: Tensor, return_dist: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        N = x.shape[0]
        patches = self.patch_embed(x)  # (N, num_patches, D)
        cls_tokens = np.repeat(self.cls_token.data, N, axis=0)
        dist_tokens = np.repeat(self.dist_token.data, N, axis=0)

        # Concatenate [CLS, DIST, PATCHES]
        x_tok = np.concatenate([cls_tokens, dist_tokens, patches.data], axis=1)
        x_emb = Tensor(x_tok, requires_grad=x.requires_grad) + self.pos_embed

        for b in self.blocks:
            x_emb = b(x_emb)

        cls_out = self.head_cls(
            self.norm(Tensor(x_emb.data[:, 0, :], requires_grad=x.requires_grad))
        )
        dist_out = self.head_dist(
            self.norm(Tensor(x_emb.data[:, 1, :], requires_grad=x.requires_grad))
        )
        if return_dist:
            return cls_out, dist_out
        return (cls_out + dist_out) / 2.0

    def fit(self, X: Any, y: Any = None, **kwargs: Any) -> DeiT:
        from ...dl.sequential import Sequential

        seq = Sequential([self])
        seq.fit(X, y, **kwargs)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not isinstance(X, Tensor):
            X = Tensor(np.asarray(X, dtype=np.float64), requires_grad=False)
        out = self.forward(X)
        out_data = (
            (out[0].data + out[1].data) / 2.0 if isinstance(out, tuple) else out.data
        )
        if out_data.ndim == 2 and out_data.shape[1] > 1:
            return np.argmax(out_data, axis=1)
        return out_data.flatten()


DeiTTiny = DeiT
DeiTSmall = DeiT
DeiTBase = DeiT
