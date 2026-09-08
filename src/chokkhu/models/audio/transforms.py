"""Audio Signal Processing Transforms (STFT, ISTFT, MelSpectrogram, MFCC, SpecAugment) from First Principles."""

from __future__ import annotations

from typing import Optional
import numpy as np


def get_window(window: str, win_length: int) -> np.ndarray:
    """Generate analysis window."""
    w = window.lower()
    if w in ("hann", "hanning"):
        return np.hanning(win_length)
    elif w == "hamming":
        return np.hamming(win_length)
    elif w == "blackman":
        return np.blackman(win_length)
    elif w in ("rect", "rectangular"):
        return np.ones(win_length, dtype=np.float64)
    else:
        raise ValueError(f"Unsupported window type: {window}")


def stft(
    x: np.ndarray,
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
) -> np.ndarray:
    """Short-Time Fourier Transform (STFT) from First Principles."""
    is_1d = x.ndim == 1
    if is_1d:
        x = x[np.newaxis, :]

    N, T = x.shape
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 4

    win = get_window(window, win_length)
    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        win = np.pad(win, (pad_left, pad_right), mode="constant")

    if center:
        pad_amount = n_fft // 2
        x = np.pad(x, ((0, 0), (pad_amount, pad_amount)), mode="reflect")
        T = x.shape[1]

    num_frames = 1 + (T - n_fft) // hop_length
    if num_frames <= 0:
        raise ValueError(f"Signal length ({T}) is too short for n_fft={n_fft}")

    frame_indices = (
        np.arange(num_frames)[:, None] * hop_length + np.arange(n_fft)[None, :]
    )
    frames = x[:, frame_indices]

    windowed_frames = frames * win[np.newaxis, np.newaxis, :]
    stft_matrix = np.fft.rfft(windowed_frames, n=n_fft, axis=-1)

    stft_matrix = np.swapaxes(stft_matrix, 1, 2)
    return stft_matrix[0] if is_1d else stft_matrix


def istft(
    stft_matrix: np.ndarray,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    length: Optional[int] = None,
) -> np.ndarray:
    """Inverse Short-Time Fourier Transform (ISTFT) via Overlap-Add."""
    is_1d = stft_matrix.ndim == 2
    if is_1d:
        stft_matrix = stft_matrix[np.newaxis, ...]

    N, freq_bins, num_frames = stft_matrix.shape
    n_fft = (freq_bins - 1) * 2

    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 4

    win = get_window(window, win_length)
    if win_length < n_fft:
        pad_left = (n_fft - win_length) // 2
        pad_right = n_fft - win_length - pad_left
        win = np.pad(win, (pad_left, pad_right), mode="constant")

    stft_t = np.swapaxes(stft_matrix, 1, 2)
    frames = np.fft.irfft(stft_t, n=n_fft, axis=-1)
    windowed_frames = frames * win[np.newaxis, np.newaxis, :]

    expected_len = (num_frames - 1) * hop_length + n_fft
    y: np.ndarray = np.zeros((N, expected_len), dtype=np.float64)
    window_sum: np.ndarray = np.zeros(expected_len, dtype=np.float64)
    win_sq: np.ndarray = win**2

    for frame_idx in range(num_frames):
        sample_start = frame_idx * hop_length
        sample_end = sample_start + n_fft
        y[:, sample_start:sample_end] += windowed_frames[:, frame_idx, :]
        window_sum[sample_start:sample_end] += win_sq

    window_mask = window_sum > 1e-8
    y[:, window_mask] /= window_sum[window_mask]

    if center:
        pad_amount = n_fft // 2
        y = y[:, pad_amount : expected_len - pad_amount]

    if length is not None:
        if y.shape[1] < length:
            y = np.pad(y, ((0, 0), (0, length - y.shape[1])), mode="constant")
        else:
            y = y[:, :length]

    return y[0] if is_1d else y


def spectrogram(
    x: np.ndarray,
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: str = "hann",
    center: bool = True,
    power: float = 2.0,
) -> np.ndarray:
    """Compute Magnitude or Power Spectrogram."""
    stft_out = stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
    )
    mag = np.abs(stft_out)
    return mag**power if power != 1.0 else mag


def mel_filterbank(
    sr: int = 16000,
    n_fft: int = 512,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
) -> np.ndarray:
    """Generate Triangular Mel-scale Filterbank Matrix."""
    if f_max is None:
        f_max = float(sr // 2)

    def hz_to_mel(f: float | np.ndarray) -> np.ndarray:
        return np.asarray(2595.0 * np.log10(1.0 + f / 700.0), dtype=np.float64)

    def mel_to_hz(m: float | np.ndarray) -> np.ndarray:
        return np.asarray(700.0 * (10.0 ** (m / 2595.0) - 1.0), dtype=np.float64)

    mel_min = float(hz_to_mel(f_min))
    mel_max = float(hz_to_mel(f_max))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points: np.ndarray = mel_to_hz(mel_points)

    fft_freqs = np.linspace(0.0, float(sr // 2), n_fft // 2 + 1)
    weights: np.ndarray = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)

    for i in range(n_mels):
        left: float = float(hz_points[i])
        center: float = float(hz_points[i + 1])
        right: float = float(hz_points[i + 2])

        up_mask = (fft_freqs >= left) & (fft_freqs <= center)
        if np.any(up_mask) and center > left:
            weights[i, up_mask] = (fft_freqs[up_mask] - left) / (center - left)

        down_mask = (fft_freqs >= center) & (fft_freqs <= right)
        if np.any(down_mask) and right > center:
            weights[i, down_mask] = (right - fft_freqs[down_mask]) / (right - center)

    return weights


def power_to_db(
    s: np.ndarray, ref: float = 1.0, amin: float = 1e-10, top_db: float = 80.0
) -> np.ndarray:
    """Convert Power Spectrogram to Decibel (dB) scale."""
    s_clean = np.maximum(amin, s)
    log_spec = 10.0 * np.log10(s_clean / ref)
    if top_db is not None:
        max_val: float = float(np.max(log_spec))
        log_spec = np.maximum(log_spec, max_val - top_db)
    return log_spec


def melspectrogram(
    x: np.ndarray,
    sr: int = 16000,
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    window: str = "hann",
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: Optional[float] = None,
    power: float = 2.0,
    to_db: bool = True,
) -> np.ndarray:
    """Compute Mel-Scale Spectrogram from raw audio signal."""
    p_spec = spectrogram(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        power=power,
    )
    mel_fb = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, f_min=f_min, f_max=f_max)

    if p_spec.ndim == 2:
        mel_spec = np.matmul(mel_fb, p_spec)
    else:
        mel_spec = np.matmul(mel_fb[np.newaxis, :, :], p_spec)

    if to_db:
        return power_to_db(mel_spec)
    return mel_spec


def dct_type2(x: np.ndarray, norm: str = "ortho") -> np.ndarray:
    """Discrete Cosine Transform (DCT Type-II) with Orthonormal Normalization."""
    N = x.shape[-2]
    n = np.arange(N)[:, None]
    k = np.arange(N)[None, :]
    dct_matrix = 2.0 * np.cos((np.pi * (2.0 * n + 1.0) * k) / (2.0 * N))

    if norm == "ortho":
        dct_matrix[:, 0] *= 1.0 / np.sqrt(4.0 * N)
        dct_matrix[:, 1:] *= 1.0 / np.sqrt(2.0 * N)

    if x.ndim == 2:
        return np.matmul(dct_matrix.T, x)
    else:
        return np.matmul(dct_matrix.T[np.newaxis, :, :], x)


def mfcc(
    x: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13,
    n_mels: int = 40,
    n_fft: int = 512,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    to_db: bool = True,
) -> np.ndarray:
    """Mel-Frequency Cepstral Coefficients (MFCC) Extraction."""
    mel_spec = melspectrogram(
        x,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        to_db=to_db,
    )
    mfccs = dct_type2(mel_spec, norm="ortho")
    if mfccs.ndim == 2:
        return mfccs[:n_mfcc, :]
    else:
        return mfccs[:, :n_mfcc, :]


def compute_deltas(features: np.ndarray, width: int = 9) -> np.ndarray:
    """Compute Delta (velocity) coefficients of audio features."""
    if width < 3 or width % 2 == 0:
        raise ValueError("width must be an odd integer >= 3")

    half_w = width // 2
    denom = 2.0 * sum(i**2 for i in range(1, half_w + 1))

    is_2d = features.ndim == 2
    if is_2d:
        features = features[np.newaxis, ...]

    N, F, T = features.shape
    padded = np.pad(features, ((0, 0), (0, 0), (half_w, half_w)), mode="edge")
    delta = np.zeros_like(features)

    for i in range(1, half_w + 1):
        delta += i * (
            padded[:, :, half_w + i : half_w + i + T]
            - padded[:, :, half_w - i : half_w - i + T]
        )

    delta /= denom
    return delta[0] if is_2d else delta


class SpecAugment:
    """SpecAugment: Frequency and Time Masking for Spectrograms (Park et al., 2019)."""

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 25,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ) -> None:
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """Apply frequency and time masking to spectrogram."""
        aug_spec = spec.copy()
        is_2d = aug_spec.ndim == 2
        if is_2d:
            aug_spec = aug_spec[np.newaxis, ...]

        N, num_freq, num_time = aug_spec.shape

        for b in range(N):
            for _ in range(self.num_freq_masks):
                f = np.random.randint(0, min(self.freq_mask_param, num_freq))
                f0 = np.random.randint(0, max(1, num_freq - f))
                aug_spec[b, f0 : f0 + f, :] = 0.0

            for _ in range(self.num_time_masks):
                t = np.random.randint(0, min(self.time_mask_param, num_time))
                t0 = np.random.randint(0, max(1, num_time - t))
                aug_spec[b, :, t0 : t0 + t] = 0.0

        return aug_spec[0] if is_2d else aug_spec
