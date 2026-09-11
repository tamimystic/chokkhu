from __future__ import annotations

from typing import Optional, Tuple
import numpy as np


class SAMTwoWayAttentionBlock:
    """Two-way Transformer cross-attention block for SAM Mask Decoder."""

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        mlp_dim: int = 128,
        seed: int = 42,
    ) -> None:
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / embed_dim)

        # 1. Prompt Self-Attention
        self.w_q_self = rng.normal(0, scale, size=(embed_dim, embed_dim)).astype(
            np.float32
        )
        self.w_k_self = rng.normal(0, scale, size=(embed_dim, embed_dim)).astype(
            np.float32
        )
        self.w_v_self = rng.normal(0, scale, size=(embed_dim, embed_dim)).astype(
            np.float32
        )
        self.w_o_self = rng.normal(0, scale, size=(embed_dim, embed_dim)).astype(
            np.float32
        )

        # 2. Cross-Attention: Prompt -> Image
        self.w_q_p2i = rng.normal(0, scale, size=(embed_dim, embed_dim)).astype(
            np.float32
        )
        self.w_k_p2i = rng.normal(0, scale, size=(embed_dim, embed_dim)).astype(
            np.float32
        )
        self.w_v_p2i = rng.normal(0, scale, size=(embed_dim, embed_dim)).astype(
            np.float32
        )
        self.w_o_p2i = rng.normal(0, scale, size=(embed_dim, embed_dim)).astype(
            np.float32
        )

        # 3. Prompt MLP
        scale_mlp = np.sqrt(2.0 / mlp_dim)
        self.w_mlp1 = rng.normal(0, scale, size=(embed_dim, mlp_dim)).astype(np.float32)
        self.w_mlp2 = rng.normal(0, scale_mlp, size=(mlp_dim, embed_dim)).astype(
            np.float32
        )

        # 4. Cross-Attention: Image -> Prompt
        self.w_q_i2p = rng.normal(0, scale, size=(embed_dim, embed_dim)).astype(
            np.float32
        )
        self.w_k_i2p = rng.normal(0, scale, size=(embed_dim, embed_dim)).astype(
            np.float32
        )
        self.w_v_i2p = rng.normal(0, scale, size=(embed_dim, embed_dim)).astype(
            np.float32
        )
        self.w_o_i2p = rng.normal(0, scale, size=(embed_dim, embed_dim)).astype(
            np.float32
        )

    def _attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        B, N_q, D = q.shape
        _, N_k, _ = k.shape

        # Reshape to multi-head: (B, num_heads, N, head_dim)
        Q = q.reshape(B, N_q, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = k.reshape(B, N_k, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = v.reshape(B, N_k, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn = exp_s / (np.sum(exp_s, axis=-1, keepdims=True) + 1e-12)

        out = np.matmul(attn, V)  # (B, num_heads, N_q, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(B, N_q, D)
        return out

    def forward(
        self, queries: np.ndarray, keys: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Queries: Prompt tokens (B, N_p, D), Keys: Image embeddings (B, N_img, D)."""
        # 1. Prompt Self-Attention
        q_s = queries @ self.w_q_self
        k_s = queries @ self.w_k_self
        v_s = queries @ self.w_v_self
        queries = queries + (self._attention(q_s, k_s, v_s) @ self.w_o_self)

        # 2. Cross-Attention: Prompt -> Image
        q_p = queries @ self.w_q_p2i
        k_i = keys @ self.w_k_p2i
        v_i = keys @ self.w_v_p2i
        queries = queries + (self._attention(q_p, k_i, v_i) @ self.w_o_p2i)

        # 3. Prompt MLP
        queries = queries + (np.maximum(0.0, queries @ self.w_mlp1) @ self.w_mlp2)

        # 4. Cross-Attention: Image -> Prompt
        q_i = keys @ self.w_q_i2p
        k_p = queries @ self.w_k_i2p
        v_p = queries @ self.w_v_i2p
        keys = keys + (self._attention(q_i, k_p, v_p) @ self.w_o_i2p)

        return queries, keys


class SegmentAnythingModel:
    """Segment Anything Model (SAM) Promptable Mask Decoder in pure NumPy.

    Decodes promptable point/box inputs into multi-class segmentation masks
    and predicted Intersection-over-Union (IoU) scores.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_mask_tokens: int = 3,
        seed: int = 42,
    ) -> None:
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_mask_tokens = num_mask_tokens

        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / embed_dim)

        # Learned prompt tokens: IoU token + mask tokens
        self.iou_token = rng.normal(0, scale, size=(1, 1, embed_dim)).astype(np.float32)
        self.mask_tokens = rng.normal(
            0, scale, size=(1, num_mask_tokens, embed_dim)
        ).astype(np.float32)

        # Prompt coordinate projector (x, y) -> embed_dim
        self.w_coord_proj = rng.normal(0, scale, size=(2, embed_dim)).astype(np.float32)
        # Foreground / background label embedding
        self.label_embeddings = rng.normal(0, scale, size=(2, embed_dim)).astype(
            np.float32
        )

        # Two-way attention transformer layers
        self.block1 = SAMTwoWayAttentionBlock(
            embed_dim=embed_dim, num_heads=num_heads, seed=seed
        )
        self.block2 = SAMTwoWayAttentionBlock(
            embed_dim=embed_dim, num_heads=num_heads, seed=seed + 1
        )

        # Mask upscaler convs
        self.w_mask_mlp = rng.normal(0, scale, size=(embed_dim, embed_dim)).astype(
            np.float32
        )
        self.w_iou_head = rng.normal(
            0, scale, size=(embed_dim, num_mask_tokens)
        ).astype(np.float32)

    def encode_prompts(
        self,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Encodes point and box prompts into positional token embeddings (B, N_prompts, D)."""
        prompt_tokens_list = []

        if points is not None:
            # points: (B, N_pts, 2)
            pts = np.asarray(points, dtype=np.float32)
            if pts.ndim == 2:
                pts = pts[np.newaxis, ...]
            B, N_pts, _ = pts.shape

            # Coordinate projection: (B, N_pts, D)
            pts_emb = np.matmul(pts, self.w_coord_proj)

            if labels is not None:
                lbls = np.asarray(labels, dtype=int).reshape(B, N_pts)
                lbl_emb = self.label_embeddings[lbls]
                pts_emb = pts_emb + lbl_emb

            prompt_tokens_list.append(pts_emb)

        if boxes is not None:
            # boxes: (B, N_boxes, 4) -> top-left + bottom-right
            bxs = np.asarray(boxes, dtype=np.float32)
            if bxs.ndim == 2:
                bxs = bxs[np.newaxis, ...]
            B = bxs.shape[0]

            tl = bxs[:, :, :2] @ self.w_coord_proj + self.label_embeddings[1]
            br = bxs[:, :, 2:] @ self.w_coord_proj + self.label_embeddings[0]
            prompt_tokens_list.extend([tl, br])

        if prompt_tokens_list:
            return np.concatenate(prompt_tokens_list, axis=1)
        else:
            return np.zeros((1, 0, self.embed_dim), dtype=np.float32)

    def forward(
        self,
        image_embeddings: np.ndarray,
        points: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        boxes: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass predicting multi-mask segmentation and IoU confidence scores.

        Args:
            image_embeddings: (B, embed_dim, H_feat, W_feat)
            points: Optional (B, N_pts, 2) coordinates
            labels: Optional (B, N_pts) 1 for positive, 0 for negative
            boxes: Optional (B, N_boxes, 4) coordinates

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Predicted masks: (B, num_mask_tokens, H_feat, W_feat)
                - Predicted IoU scores: (B, num_mask_tokens)
        """
        B, C, H_f, W_f = image_embeddings.shape
        img_flat = image_embeddings.reshape(B, C, H_f * W_f).transpose(
            0, 2, 1
        )  # (B, N_img, C)

        # 1. Assemble prompt tokens
        special_tokens = np.concatenate(
            [
                np.repeat(self.iou_token, B, axis=0),
                np.repeat(self.mask_tokens, B, axis=0),
            ],
            axis=1,
        )  # (B, 1 + num_masks, D)

        sparse_prompts = self.encode_prompts(points=points, labels=labels, boxes=boxes)
        if sparse_prompts.shape[1] > 0:
            tokens = np.concatenate([special_tokens, sparse_prompts], axis=1)
        else:
            tokens = special_tokens

        # 2. Two-way cross-attention
        tokens, img_flat = self.block1.forward(tokens, img_flat)
        tokens, img_flat = self.block2.forward(tokens, img_flat)

        # 3. Extract output tokens
        iou_token_out = tokens[:, 0, :]  # (B, D)
        mask_tokens_out = tokens[
            :, 1 : 1 + self.num_mask_tokens, :
        ]  # (B, num_masks, D)

        # 4. Predict IoU scores
        iou_scores = iou_token_out @ self.w_iou_head  # (B, num_masks)

        # 5. Predict Mask Logits via dot-product
        mask_weights = mask_tokens_out @ self.w_mask_mlp  # (B, num_masks, D)
        # (B, num_masks, D) @ (B, D, N_img) -> (B, num_masks, N_img)
        masks = np.matmul(mask_weights, img_flat.transpose(0, 2, 1))
        masks = masks.reshape(B, self.num_mask_tokens, H_f, W_f)

        return masks, iou_scores
