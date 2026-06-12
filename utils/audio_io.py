"""Audio/video I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def extract_audio(media_path: str) -> Tuple[np.ndarray, int]:
    """Extract audio from audio or video file.

    Args:
        media_path: Path to input media file.

    Returns:
        waveform: 1D float32 numpy array.
        sample_rate: Integer sample rate.
    """
    # TODO: Use ffmpeg / soundfile to extract and normalize audio
    raise NotImplementedError
