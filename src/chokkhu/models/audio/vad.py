"""Voice Activity Detection (VAD) for Audio Signal Segmentation in pure NumPy."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np


class VoiceActivityDetector:
    """Multi-cue Voice Activity Detector (VAD) in pure NumPy.

    Combines Short-Time Energy (STE), Zero-Crossing Rate (ZCR), and
    Spectral Flux with adaptive thresholding and state-machine smoothing
    to segment speech from ambient noise.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length_ms: float = 25.0,
        hop_length_ms: float = 10.0,
        energy_threshold: float = 0.02,
        zcr_threshold: float = 0.15,
        min_speech_duration_ms: float = 100.0,
        min_silence_duration_ms: float = 200.0,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.frame_len = int(sample_rate * frame_length_ms / 1000.0)
        self.hop_len = int(sample_rate * hop_length_ms / 1000.0)
        self.energy_threshold = float(energy_threshold)
        self.zcr_threshold = float(zcr_threshold)
        self.min_speech_frames = int(min_speech_duration_ms / hop_length_ms)
        self.min_silence_frames = int(min_silence_duration_ms / hop_length_ms)

    def _frame_signal(self, audio: np.ndarray) -> np.ndarray:
        """Splits 1D raw waveform into overlapping frames of shape (num_frames, frame_len)."""
        sig = np.asarray(audio, dtype=np.float32).flatten()
        if len(sig) < self.frame_len:
            pad: np.ndarray = np.zeros(self.frame_len - len(sig), dtype=np.float32)
            sig = np.concatenate([sig, pad])

        num_frames = 1 + (len(sig) - self.frame_len) // self.hop_len
        frames: np.ndarray = np.zeros((num_frames, self.frame_len), dtype=np.float32)

        for i in range(num_frames):
            start = i * self.hop_len
            frames[i] = sig[start : start + self.frame_len]

        return frames

    def compute_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """Computes Short-Time Energy (STE), Zero-Crossing Rate (ZCR), and Spectral Flux."""
        frames = self._frame_signal(audio)
        num_frames = len(frames)

        # 1. Short-Time Energy (RMS)
        ste = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        # Normalize STE to [0, 1]
        ste_norm = (ste - np.min(ste)) / (np.ptp(ste) + 1e-12)

        # 2. Zero-Crossing Rate
        signs = np.sign(frames)
        signs[signs == 0] = 1
        zcr = np.mean(np.abs(np.diff(signs, axis=1)) > 0, axis=1)

        # 3. Spectral Flux via FFT
        window: np.ndarray = np.hanning(self.frame_len).astype(np.float32)
        windowed = frames * window
        magnitudes = np.abs(np.fft.rfft(windowed, axis=1))

        # Flux is positive difference between adjacent frame spectra
        flux: np.ndarray = np.zeros(num_frames, dtype=np.float32)
        if num_frames > 1:
            diff = magnitudes[1:] - magnitudes[:-1]
            flux[1:] = np.mean(np.maximum(0.0, diff), axis=1)
            flux[0] = flux[1]

        flux_norm = (flux - np.min(flux)) / (np.ptp(flux) + 1e-12)

        return {
            "ste": ste_norm,
            "zcr": zcr,
            "spectral_flux": flux_norm,
        }

    def detect(self, audio: np.ndarray) -> Dict[str, Any]:
        """Detects voice activity segments and returns frame mask and timestamp intervals.

        Args:
            audio: 1D waveform array.

        Returns:
            Dict containing:
                - "mask": Boolean array of shape (num_frames,) indicating speech activity.
                - "segments": List of (start_sec, end_sec) speech timestamp intervals.
                - "speech_ratio": Ratio of active speech frames to total audio duration.
        """
        feats = self.compute_features(audio)
        ste = feats["ste"]
        zcr = feats["zcr"]
        flux = feats["spectral_flux"]

        # Combined composite speech score
        speech_score = 0.5 * ste + 0.3 * flux + 0.2 * (1.0 - zcr)
        raw_mask = speech_score > self.energy_threshold

        # State-machine smoothing (hangover / debounce filter)
        smoothed_mask = raw_mask.copy()
        in_speech = False
        speech_len = 0
        silence_len = 0

        for i, val in enumerate(raw_mask):
            if val:
                speech_len += 1
                silence_len = 0
                if speech_len >= self.min_speech_frames:
                    in_speech = True
            else:
                silence_len += 1
                speech_len = 0
                if silence_len >= self.min_silence_frames:
                    in_speech = False

            smoothed_mask[i] = in_speech

        # Extract time intervals (start_sec, end_sec)
        segments: List[Tuple[float, float]] = []
        is_active = False
        seg_start = 0

        for i, active in enumerate(smoothed_mask):
            if active and not is_active:
                is_active = True
                seg_start = i
            elif not active and is_active:
                is_active = False
                t_start = (seg_start * self.hop_len) / self.sample_rate
                t_end = (i * self.hop_len + self.frame_len) / self.sample_rate
                segments.append((round(t_start, 3), round(t_end, 3)))

        if is_active:
            t_start = (seg_start * self.hop_len) / self.sample_rate
            t_end = (
                len(smoothed_mask) * self.hop_len + self.frame_len
            ) / self.sample_rate
            segments.append((round(t_start, 3), round(t_end, 3)))

        ratio = float(np.mean(smoothed_mask)) if len(smoothed_mask) > 0 else 0.0

        return {
            "mask": smoothed_mask,
            "segments": segments,
            "speech_ratio": ratio,
        }
