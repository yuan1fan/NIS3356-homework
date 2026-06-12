"""Audio/video validation, resampling, and duration utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

from .media_utils import VIDEO_EXTENSIONS, is_video

SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | {
    ".wav",
    ".mp3",
    ".ogg",
    ".flac",
    ".m4a",
    ".aac",
    ".wma",
}


def validate_media(media_path: str | Path) -> None:
    """Validate that the media file exists, has a supported extension, and is non-empty.

    Args:
        media_path: Path to input media file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not supported or file is empty.
    """
    path = Path(media_path)
    if not path.exists():
        raise FileNotFoundError(f"Media file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported media format: {path.suffix}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    if path.stat().st_size == 0:
        raise ValueError(f"Media file is empty (0 bytes): {path}")
    if path.stat().st_size < 64:
        raise ValueError(
            f"Media file too small ({path.stat().st_size} bytes) — "
            "may be corrupt or not a valid audio file"
        )


def load_and_resample(
    media_path: str | Path,
    target_sr: int = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load audio/video, resample to target sample rate, and convert to mono.

    Args:
        media_path: Path to input media file.
        target_sr: Target sample rate in Hz.
        mono: Whether to convert to mono channel.

    Returns:
        waveform: 1D float32 numpy array if mono, else 2D.
        sample_rate: Actual sample rate after loading.
    """
    from .media_utils import to_supported_audio

    _, waveform, sr = to_supported_audio(
        media_path, target_sr=target_sr, mono=mono
    )
    return waveform, sr


def check_duration(
    media_path: str | Path,
    max_seconds: float = 300.0,
) -> float:
    """Return duration in seconds and enforce an upper bound.

    Args:
        media_path: Path to input media file.
        max_seconds: Maximum allowed duration.

    Returns:
        Duration in seconds.

    Raises:
        ValueError: If duration exceeds max_seconds.
    """
    from .media_utils import to_supported_audio

    audio_path, _, sr = to_supported_audio(media_path, target_sr=16000, mono=True)
    info = sf.info(str(audio_path))
    duration = float(info.frames) / float(info.samplerate)
    if duration > max_seconds:
        raise ValueError(
            f"Media duration {duration:.1f}s exceeds limit of {max_seconds:.1f}s"
        )
    return duration
