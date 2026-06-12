"""Audio enhancement, noise reduction, and anomaly detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from utils.media_utils import to_supported_audio
from utils.types import AnomalyEvent


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spectral subtraction parameters
# ---------------------------------------------------------------------------
_STFT_WINDOW = 512        # FFT window size
_STFT_HOP = 128          # Hop length
_SPECTRAL_FLOOR = 0.002  # Spectral floor to avoid negative values
_NOISE_EST_FRAMES = 30    # First N frames used for noise estimation


@dataclass
class AudioEnhancer:
    """Lightweight audio preprocessing and anomaly detection.

    Responsibilities:
    - Load audio/video and extract mono 16kHz waveform
    - Basic denoising via spectral subtraction
    - Extract features for anomaly detection:
      * zero-crossing rate (ZCR)
      * spectral centroid
      * spectral entropy
      * RMS energy
    - Threshold-based anomaly labeling
    """

    config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        enh_cfg = self.config.get("enhancer", {})
        self._target_sr = int(enh_cfg.get("target_sr", 16000))
        self._denoise = bool(enh_cfg.get("denoise", False))
        # Anomaly thresholds
        self._zcr_threshold = float(enh_cfg.get("zcr_threshold", 0.3))
        self._spectral_entropy_threshold = float(
            enh_cfg.get("spectral_entropy_threshold", 8.5)
        )
        self._spectral_floor = float(enh_cfg.get("spectral_floor", _SPECTRAL_FLOOR))

    def load_waveform(self, media_path: str) -> Tuple[np.ndarray, int]:
        """Load audio/video and return (waveform, sample_rate).

        Args:
            media_path: Path to media file.

        Returns:
            waveform: 1D float32 numpy array (mono).
            sample_rate: Integer sample rate.
        """
        _, waveform, sr = to_supported_audio(
            media_path, target_sr=self._target_sr, mono=True
        )
        return waveform, sr

    # ------------------------------------------------------------------
    # Spectral subtraction denoising
    # ------------------------------------------------------------------
    @staticmethod
    def _stft(waveform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Short-time Fourier transform.

        Returns:
            (magnitude, phase) 2D arrays [freq_bins, time_frames].
        """
        from scipy.signal import stft

        _, _, Z = stft(
            waveform,
            fs=16000,
            nperseg=_STFT_WINDOW,
            noverlap=_STFT_WINDOW - _STFT_HOP,
            boundary=None,
        )
        mag = np.abs(Z)
        phase = np.angle(Z)
        return mag, phase

    @staticmethod
    def _istft(mag: np.ndarray, phase: np.ndarray, nperseg: int = _STFT_WINDOW) -> np.ndarray:
        """Inverse STFT from magnitude and phase."""
        from scipy.signal import istft

        Z = mag * np.exp(1j * phase)
        _, waveform = istft(
            Z,
            fs=16000,
            nperseg=nperseg,
            noverlap=nperseg - _STFT_HOP,
            boundary=False,
        )
        return waveform.astype(np.float32)

    def spectral_subtraction(
        self, waveform: np.ndarray, sr: int = 16000
    ) -> np.ndarray:
        """Apply magnitude spectral subtraction for basic noise reduction.

        Args:
            waveform: 1D float32 audio signal.
            sr: Sample rate (used only for STFT config).

        Returns:
            Denoised waveform (same shape as input).
        """
        from scipy.signal import stft, istft

        nperseg = _STFT_WINDOW
        noverlap = nperseg - _STFT_HOP

        _, _, Z = stft(
            waveform,
            fs=sr,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=None,
        )
        mag = np.abs(Z)
        phase = np.angle(Z)

        # Estimate noise from first N frames (assumed silence / noise-only)
        noise_profile = np.median(mag[:, :_NOISE_EST_FRAMES], axis=1, keepdims=True)
        noise_profile = np.maximum(noise_profile, self._spectral_floor)

        # Subtract noise; enforce spectral floor to avoid negative values
        denoised_mag = np.maximum(mag - noise_profile, self._spectral_floor)

        _, denoised_wav = istft(
            denoised_mag * np.exp(1j * phase),
            fs=sr,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=False,
        )
        # Align length to original
        denoised_wav = denoised_wav[: len(waveform)].astype(np.float32)
        return denoised_wav

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    @staticmethod
    def zero_crossing_rate(
        waveform: np.ndarray, frame_length: int = 256, hop_length: int = 128
    ) -> np.ndarray:
        """Compute zero-crossing rate per frame.

        Args:
            waveform: 1D audio signal.
            frame_length: Samples per frame.
            hop_length: Hop size between frames.

        Returns:
            ZCR array of shape [n_frames].
        """
        if len(waveform) < frame_length:
            return np.array([])
        steps = range(0, len(waveform) - frame_length, hop_length)
        zcr = np.zeros(len(steps))
        for i, start in enumerate(steps):
            frame = waveform[start : start + frame_length]
            zcr[i] = np.mean(np.abs(np.diff(np.sign(frame)))) / 2.0
        return zcr

    @staticmethod
    def spectral_centroid(
        waveform: np.ndarray, sr: int = 16000, nperseg: int = 1024, noverlap: int = 512
    ) -> np.ndarray:
        """Compute spectral centroid per STFT frame.

        Args:
            waveform: 1D audio signal.
            sr: Sample rate.
            nperseg: FFT window size.
            noverlap: Overlap in samples.

        Returns:
            Spectral centroid array in Hz, shape [n_frames].
        """
        from scipy.signal import spectrogram

        freqs, _, Sxx = spectrogram(
            waveform,
            fs=sr,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
        )
        # Weight frequency bins by magnitude and normalise
        spectral_sum = np.sum(Sxx, axis=0, keepdims=True)
        spectral_sum = np.maximum(spectral_sum, 1e-10)
        return np.sum(freqs[:, None] * Sxx, axis=0) / spectral_sum[0]

    @staticmethod
    def spectral_entropy(
        waveform: np.ndarray, sr: int = 16000, nperseg: int = 1024, noverlap: int = 512
    ) -> np.ndarray:
        """Compute spectral entropy per STFT frame.

        Args:
            waveform: 1D audio signal.
            sr: Sample rate.
            nperseg: FFT window size.
            noverlap: Overlap in samples.

        Returns:
            Spectral entropy array, shape [n_frames].
        """
        from scipy.signal import spectrogram

        _, _, Sxx = spectrogram(
            waveform,
            fs=sr,
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
        )
        # Normalise to probability distribution per frame
        mag = np.maximum(Sxx, 1e-10)
        p = mag / np.sum(mag, axis=0, keepdims=True)
        # Shannon entropy in bits
        entropy = -np.sum(p * np.log2(p), axis=0)
        return entropy

    @staticmethod
    def rms_energy(
        waveform: np.ndarray, frame_length: int = 256, hop_length: int = 128
    ) -> np.ndarray:
        """Compute RMS energy per frame.

        Returns:
            RMS array of shape [n_frames].
        """
        if len(waveform) < frame_length:
            return np.array([])
        steps = range(0, len(waveform) - frame_length, hop_length)
        rms = np.zeros(len(steps))
        for i, start in enumerate(steps):
            frame = waveform[start : start + frame_length]
            rms[i] = np.sqrt(np.mean(frame**2))
        return rms

    def extract_features(
        self, waveform: np.ndarray, sr: int = 16000
    ) -> Dict[str, np.ndarray]:
        """Extract all hand-crafted features.

        Returns:
            Dict with keys: zcr, spectral_centroid, spectral_entropy, rms_energy,
            frame_times (seconds for each frame).
        """
        hop = _STFT_HOP
        frame_times = (
            np.arange(0, waveform.shape[0] - _STFT_WINDOW, hop)
            + _STFT_WINDOW // 2
        ) / sr

        return {
            "zcr": self.zero_crossing_rate(waveform),
            "spectral_centroid": self.spectral_centroid(waveform, sr=sr),
            "spectral_entropy": self.spectral_entropy(waveform, sr=sr),
            "rms_energy": self.rms_energy(waveform),
            "frame_times": frame_times,
        }

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------
    def detect_anomalies(
        self, waveform: np.ndarray, sr: int = 16000
    ) -> List[AnomalyEvent]:
        """Detect anomalies from audio features using threshold rules.

        Generates AnomalyEvent spans for:
        - High ZCR segments  -> "[强噪声]"
        - High spectral entropy (synthesized / unnatural) -> "[疑似合成]"

        Args:
            waveform: 1D float32 audio signal.
            sr: Sample rate.

        Returns:
            List of AnomalyEvent sorted by start time.
        """
        features = self.extract_features(waveform, sr=sr)
        zcr = features["zcr"]
        spectral_ent = features["spectral_entropy"]
        frame_times = features["frame_times"]

        if len(frame_times) == 0:
            return []

        # Frame duration in seconds
        hop = _STFT_HOP
        frame_duration = hop / sr
        window_frames = max(1, int(0.5 / frame_duration))  # ~500ms window

        def rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
            if len(arr) < w:
                return np.full_like(arr, np.nan)
            return np.convolve(arr, np.ones(w) / w, mode="valid")

        # Smooth over ~500ms
        zcr_smooth = rolling_mean(zcr, window_frames)
        ent_smooth = rolling_mean(spectral_ent, window_frames)
        n_smooth = len(zcr_smooth)
        smooth_times = frame_times[:n_smooth] + (window_frames - 1) * frame_duration / 2

        events: List[AnomalyEvent] = []

        # High ZCR -> "[强噪声]"
        if len(zcr_smooth) > 0:
            zcr_mask = zcr_smooth > self._zcr_threshold
            for start_idx, end_idx in self._contiguous_regions(zcr_mask):
                start_t = float(smooth_times[start_idx])
                end_t = float(smooth_times[min(end_idx, len(smooth_times) - 1)])
                if end_t - start_t >= 0.1:
                    avg_conf = float(np.nanmean(zcr_smooth[start_idx:end_idx]))
                    events.append(
                        AnomalyEvent(
                            start=start_t,
                            end=end_t,
                            label="[强噪声]",
                            confidence=min(avg_conf / (self._zcr_threshold * 2), 1.0),
                        )
                    )

        # High spectral entropy -> "[疑似合成]"
        if len(ent_smooth) > 0:
            ent_mask = ent_smooth > self._spectral_entropy_threshold
            for start_idx, end_idx in self._contiguous_regions(ent_mask):
                start_t = float(smooth_times[start_idx])
                end_t = float(smooth_times[min(end_idx, len(smooth_times) - 1)])
                if end_t - start_t >= 0.1:
                    avg_conf = float(np.nanmean(ent_smooth[start_idx:end_idx]))
                    events.append(
                        AnomalyEvent(
                            start=start_t,
                            end=end_t,
                            label="[疑似合成]",
                            confidence=min(
                                (avg_conf - self._spectral_entropy_threshold)
                                / (self._spectral_entropy_threshold * 0.5),
                                1.0,
                            ),
                        )
                    )

        events.sort(key=lambda e: e.start)
        return events

    @staticmethod
    def _contiguous_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
        """Return start/end indices of contiguous True regions in a boolean array."""
        padded = np.concatenate(([False], mask, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        return list(zip(starts, ends))
