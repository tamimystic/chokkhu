"""Unit tests for Audio Processing Universe (STFT, ISTFT, MelSpectrogram, MFCC, Conformer, AST, Wav2Vec2)."""

from __future__ import annotations
import numpy as np

from chokkhu.core.tensor import Tensor
from chokkhu.models.audio import (
    AST,
    Conformer,
    Conv1D,
    SpecAugment,
    Wav2Vec2,
    compute_deltas,
    istft,
    mel_filterbank,
    melspectrogram,
    mfcc,
    stft,
)
from chokkhu.models import train


def test_stft_and_istft_reconstruction() -> None:
    sr = 16000
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
    sig = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)

    n_fft = 512
    hop_length = 128
    S = stft(sig, n_fft=n_fft, hop_length=hop_length)

    assert S.shape[0] == n_fft // 2 + 1
    assert S.shape[1] > 10
    assert np.iscomplexobj(S)

    recon = istft(S, hop_length=hop_length, length=len(sig))
    assert recon.shape == sig.shape

    interior_slice = slice(n_fft, len(sig) - n_fft)
    np.testing.assert_allclose(sig[interior_slice], recon[interior_slice], atol=1e-3)


def test_mel_filterbank_and_spectrogram() -> None:
    sr = 16000
    n_fft = 512
    n_mels = 64
    fb = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels)
    assert fb.shape == (n_mels, n_fft // 2 + 1)
    assert np.all(fb >= 0.0)

    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    sig = np.sin(2 * np.pi * 500 * t)
    mel_spec = melspectrogram(sig, sr=sr, n_fft=n_fft, n_mels=n_mels, to_db=True)

    assert mel_spec.shape[0] == n_mels
    assert mel_spec.shape[1] > 5


def test_mfcc_and_deltas() -> None:
    sr = 16000
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    sig = np.sin(2 * np.pi * 600 * t)

    mfccs = mfcc(sig, sr=sr, n_mfcc=13, n_mels=40)
    assert mfccs.shape[0] == 13
    assert mfccs.shape[1] > 5

    deltas = compute_deltas(mfccs, width=5)
    assert deltas.shape == mfccs.shape


def test_spec_augment() -> None:
    spec = np.ones((64, 40), dtype=np.float64)
    augmenter = SpecAugment(freq_mask_param=8, time_mask_param=8)
    aug = augmenter(spec)
    assert aug.shape == spec.shape
    assert np.any(aug == 0.0)


def test_conv1d_forward_and_backward() -> None:
    x = Tensor(np.random.randn(2, 4, 32), requires_grad=True)
    conv = Conv1D(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1)
    out = conv(x)
    assert out.shape == (2, 8, 16)

    out.backward(np.ones_like(out.data))
    assert x.grad is not None
    assert x.grad.shape == (2, 4, 32)
    assert conv.weight.grad is not None


def test_conformer_forward() -> None:
    x = Tensor(np.random.randn(2, 16, 80), requires_grad=True)
    model = Conformer(
        in_features=80,
        num_classes=26,
        embed_dim=64,
        num_blocks=2,
        num_heads=2,
        ffn_dim=128,
        conv_kernel_size=7,
    )
    logits = model(x)
    assert logits.shape == (2, 16, 26)


def test_ast_forward() -> None:
    x = Tensor(np.random.randn(2, 1, 32, 32), requires_grad=True)
    model = AST(
        num_classes=10,
        embed_dim=64,
        num_layers=2,
        num_heads=2,
        patch_size=8,
        stride=8,
    )
    logits = model(x)
    assert logits.shape == (2, 10)


def test_wav2vec2_forward() -> None:
    x = Tensor(np.random.randn(2, 1600), requires_grad=True)
    model = Wav2Vec2(
        num_classes=20,
        embed_dim=64,
        num_layers=2,
        num_heads=2,
        dim_feedforward=128,
    )
    logits = model(x)
    assert logits.ndim == 3
    assert logits.shape[0] == 2
    assert logits.shape[2] == 20


def test_train_audio_via_engine() -> None:
    X_train = np.random.randn(4, 10, 40)
    y_train = np.array([0, 1, 0, 1])

    clf = train(
        model="conformer",
        X_train=X_train,
        y_train=y_train,
        in_features=40,
        num_classes=2,
        embed_dim=32,
        num_blocks=1,
        num_heads=2,
        ffn_dim=64,
        conv_kernel_size=5,
        epochs=1,
        batch_size=2,
    )
    preds = clf.predict(X_train)
    assert preds.shape[0] == 4
