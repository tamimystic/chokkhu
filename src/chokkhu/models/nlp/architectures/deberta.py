"""DeBERTa (Decoding-enhanced BERT with Disentangled Attention) Architecture (He et al., 2021)."""

from __future__ import annotations

from typing import Any, Optional, Union
import numpy as np

from chokkhu.core.tensor import Tensor
from ...base import ChokkhuModel
from ...dl.layers import Dropout, LayerNorm, Linear, Module, Parameter
from ...dl.activations import GELU
from ..embeddings import TokenEmbedding


class DisentangledSelfAttention(Module):
    """Disentangled Self-Attention decomposing content and relative position vectors."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        max_relative_positions: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_rel = max_relative_positions

        # Content projections
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)

        # Relative position projection & embedding table (2 * max_rel)
        self.rel_embed = Parameter(
            np.random.randn(2 * max_relative_positions, self.head_dim) * 0.02
        )
        self.q_rel = Linear(self.head_dim, self.head_dim)
        self.k_rel = Linear(self.head_dim, self.head_dim)
        self.dropout = Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        N, S, D = x.shape
        x_flat = x.reshape(-1, D)

        q_c = (
            self.q_proj(x_flat)
            .data.reshape(N, S, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        k_c = (
            self.k_proj(x_flat)
            .data.reshape(N, S, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )
        v_c = (
            self.v_proj(x_flat)
            .data.reshape(N, S, self.num_heads, self.head_dim)
            .swapaxes(1, 2)
        )

        # 1. Content-to-Content: Q_c * K_c^T
        scores_cc = np.matmul(q_c, k_c.swapaxes(-2, -1))

        # Relative position index matrix delta(i, j) = clip(i - j + max_rel, 0, 2*max_rel - 1)
        i_idx = np.arange(S)[:, None]
        j_idx = np.arange(S)[None, :]
        rel_pos_indices = np.clip(i_idx - j_idx + self.max_rel, 0, 2 * self.max_rel - 1)
        rel_embs = self.rel_embed.data[rel_pos_indices]  # (S, S, head_dim)

        # 2. Content-to-Position: Q_c * K_r^T
        # k_r = (S, S, head_dim)
        k_r = self.k_rel(Tensor(rel_embs.reshape(-1, self.head_dim))).data.reshape(
            S, S, self.head_dim
        )
        # q_c: (N, H, S, d), k_r: (S, S, d) -> scores_cp: (N, H, S, S)
        scores_cp = np.einsum("nhsd,srd->nhsr", q_c, k_r)

        # 3. Position-to-Content: K_c * Q_r^T
        q_r = self.q_rel(Tensor(rel_embs.reshape(-1, self.head_dim))).data.reshape(
            S, S, self.head_dim
        )
        scores_pc = np.einsum("nhrd,srd->nhsr", k_c, q_r)

        # Combine disentangled scores scaled by 1 / sqrt(3 * head_dim)
        scale = 1.0 / np.sqrt(3.0 * self.head_dim)
        scores = (scores_cc + scores_cp + scores_pc) * scale

        if mask is not None:
            if mask.dtype == bool:
                scores = np.where(mask, scores, -1e9)
            else:
                scores = scores + mask

        max_s = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - max_s)
        attn_w = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-12)

        context = np.matmul(attn_w, v_c).swapaxes(1, 2).reshape(N * S, self.embed_dim)
        out = self.out_proj(Tensor(context, requires_grad=x.requires_grad))
        return out.reshape(N, S, self.embed_dim)


class DeBERTaLayer(Module):
    """DeBERTa Transformer Layer with Disentangled Self-Attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        max_relative_positions: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = DisentangledSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_relative_positions=max_relative_positions,
            dropout=dropout,
        )
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.linear1 = Linear(embed_dim, dim_feedforward)
        self.act = GELU()
        self.linear2 = Linear(dim_feedforward, embed_dim)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        N, S, D = x.shape
        # Pre-LN
        x_norm1 = self.norm1(x.reshape(-1, D)).reshape(N, S, D)
        attn_out = self.attn(x_norm1, mask=mask)
        x = x + self.dropout(attn_out)

        x_norm2 = self.norm2(x.reshape(-1, D))
        ffn_out = self.linear2(self.act(self.linear1(x_norm2))).reshape(N, S, D)
        x = x + self.dropout(ffn_out)
        return x


class DeBERTa(Module, ChokkhuModel):
    """DeBERTa Architecture from Scratch."""

    def __init__(
        self,
        vocab_size: int = 50265,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        max_relative_positions: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_embed = TokenEmbedding(vocab_size, embed_dim)
        self.norm = LayerNorm(embed_dim)
        self.dropout = Dropout(dropout)

        self.layers = [
            DeBERTaLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                max_relative_positions=max_relative_positions,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]
        for i, layer in enumerate(self.layers):
            setattr(self, f"layer_{i}", layer)

    def forward(
        self,
        input_ids: Union[np.ndarray, Tensor],
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        tok_emb = self.token_embed(input_ids)
        N, S, D = tok_emb.shape
        x = self.norm(tok_emb.reshape(-1, D)).reshape(N, S, D)
        x = self.dropout(x)

        mask = None
        if attention_mask is not None:
            mask = attention_mask[:, np.newaxis, np.newaxis, :].astype(bool)

        for layer in self.layers:
            x = layer(x, mask=mask)

        return x


class DebertaForSequenceClassification(Module, ChokkhuModel):
    """DeBERTa with Sequence Classification Head."""

    def __init__(
        self,
        vocab_size: int = 50265,
        num_classes: int = 2,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        max_relative_positions: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.deberta = DeBERTa(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            max_relative_positions=max_relative_positions,
            dropout=dropout,
        )
        self.classifier = Linear(embed_dim, num_classes)
        self.dropout = Dropout(dropout)

    def forward(
        self,
        input_ids: Union[np.ndarray, Tensor],
        attention_mask: Optional[np.ndarray] = None,
    ) -> Tensor:
        encoder_out = self.deberta(input_ids, attention_mask=attention_mask)
        cls_vec = Tensor(
            encoder_out.data[:, 0, :], requires_grad=encoder_out.requires_grad
        )
        return self.classifier(self.dropout(cls_vec))

    def fit(
        self, X: Any, y: Any = None, **kwargs: Any
    ) -> DebertaForSequenceClassification:
        from ...dl.sequential import Sequential

        seq = Sequential([self])
        seq.fit(X, y, **kwargs)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not isinstance(X, Tensor):
            X = Tensor(np.asarray(X, dtype=np.int64), requires_grad=False)
        out = self.forward(X).data
        if out.ndim == 2 and out.shape[1] > 1:
            return np.argmax(out, axis=1)
        return out.flatten()
