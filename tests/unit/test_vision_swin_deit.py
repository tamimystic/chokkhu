"""Unit tests for Swin Transformer and DeiT architectures."""

from __future__ import annotations
import numpy as np

import chokkhu as ck
from chokkhu.core.tensor import Tensor
from chokkhu.models.vision import (
    DeiT,
    DeiTTiny,
    SwinT,
    SwinTransformer,
)
from chokkhu.models.vision.architectures.swin import PatchMerging, WindowAttention


def test_window_attention():
    B_w = 4
    N = 16  # 4x4 window
    dim = 32
    attn = WindowAttention(dim=dim, window_size=4, num_heads=2)
    x = Tensor(np.random.randn(B_w, N, dim))
    out = attn(x)
    assert out.shape == (B_w, N, dim)


def test_patch_merging():
    N = 2
    H, W = 8, 8
    dim = 16
    merging = PatchMerging(dim=dim)
    x = Tensor(np.random.randn(N, H * W, dim))
    out, H_out, W_out = merging(x, H, W)

    assert H_out == 4
    assert W_out == 4
    assert out.shape == (N, 16, 32)


def test_swin_transformer():
    N = 2
    x = Tensor(np.random.randn(N, 3, 32, 32))
    model = SwinTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=5,
        embed_dim=32,
        depths=(1, 1),
        num_heads=(2, 4),
        window_size=4,
    )
    logits = model(x)
    assert logits.shape == (N, 5)

    preds = model.predict(x)
    assert len(preds) == N
    assert SwinT is SwinTransformer


def test_deit_distillation():
    N = 2
    x = Tensor(np.random.randn(N, 3, 32, 32))
    model = DeiT(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=4,
        embed_dim=32,
        depth=2,
        num_heads=2,
    )
    cls_logits, dist_logits = model(x, return_dist=True)
    assert cls_logits.shape == (N, 4)
    assert dist_logits.shape == (N, 4)

    preds = model.predict(x)
    assert len(preds) == N
    assert DeiTTiny is DeiT


def test_top_level_train_swin_and_deit():
    N = 4
    X = np.random.randn(N, 3, 32, 32)
    y = np.array([0, 1, 0, 1])

    m1 = ck.train(
        model="swin",
        X_train=X,
        y_train=y,
        num_classes=2,
        epochs=1,
        verbose=False,
    )
    preds1 = m1.predict(X)
    assert len(preds1) == N

    m2 = ck.train(
        model="deit",
        X_train=X,
        y_train=y,
        num_classes=2,
        epochs=1,
        verbose=False,
    )
    preds2 = m2.predict(X)
    assert len(preds2) == N
