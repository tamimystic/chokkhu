"""Audio Signal Processing, Speech Transformers, and Spectrogram Models for Chokkhu."""

from __future__ import annotations

from .architectures import (
    AST,
    AudioSpectrogramTransformer,
    Conformer,
    ConformerBlock,
    ConformerConvModule,
    Wav2Vec2,
    Whisper,
    WhisperDecoder,
    WhisperEncoder,
)
from .conv1d import (
    BatchNorm1D,
    Conv1D,
    DepthwiseConv1D,
    im2col1d,
)
from .rvq import (
    ResidualVectorQuantizer,
    VectorQuantizerStage,
)
from .transforms import (
    SpecAugment,
    compute_deltas,
    dct_type2,
    get_window,
    istft,
    mel_filterbank,
    melspectrogram,
    mfcc,
    power_to_db,
    spectrogram,
    stft,
)
from .vad import VoiceActivityDetector

__all__ = [
    # Transforms
    "get_window",
    "stft",
    "istft",
    "spectrogram",
    "mel_filterbank",
    "power_to_db",
    "melspectrogram",
    "dct_type2",
    "mfcc",
    "compute_deltas",
    "SpecAugment",
    # 1D Convolutions & Layers
    "Conv1D",
    "DepthwiseConv1D",
    "BatchNorm1D",
    "im2col1d",
    # Architectures & Speech Models
    "Conformer",
    "ConformerBlock",
    "ConformerConvModule",
    "AST",
    "AudioSpectrogramTransformer",
    "Wav2Vec2",
    "Whisper",
    "WhisperEncoder",
    "WhisperDecoder",
    # Audio Codecs & Detection
    "ResidualVectorQuantizer",
    "VectorQuantizerStage",
    "VoiceActivityDetector",
]
