"""Media input preprocessing: video/audio detection and audio extraction."""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".flv",
    ".wmv",
    ".m4v",
}

# Formats soundfile cannot read natively on most systems — must go through ffmpeg.
_SOUNDFILE_UNSUPPORTED = {
    ".m4a",
    ".aac",
    ".wma",
    ".ogg",   # may need avbin / ffmpeg backend on some setups
}


def is_video(media_path: str | Path) -> bool:
    """Return True if the file looks like a video by extension."""
    return Path(media_path).suffix.lower() in VIDEO_EXTENSIONS


def has_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


def extract_audio_from_video(
    video_path: str | Path,
    target_sr: int = 16000,
    mono: bool = True,
) -> Tuple[str, np.ndarray, int]:
    """Extract audio from a video file into a temporary wav.

    Args:
        video_path: Path to video file.
        target_sr: Target sample rate.
        mono: Convert to mono channel.

    Returns:
        (wav_path, waveform, sample_rate)
    """
    if not has_ffmpeg():
        raise RuntimeError(
            "ffmpeg is required for this file format. "
            "Install with: brew install ffmpeg"
        )

    import tempfile

    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    wav_path = tmp_wav.name

    ac = "1" if mono else "2"
    cmd = (
        f"ffmpeg -y -i {shlex.quote(str(video_path))} "
        f"-vn -acodec pcm_s16le -ac {ac} -ar {target_sr} {shlex.quote(wav_path)}"
    )
    import subprocess

    subprocess.run(
        cmd,
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if mono and data.ndim > 1:
        data = np.mean(data, axis=1)
    return wav_path, data, sr


def to_supported_audio(
    media_path: str | Path,
    target_sr: int = 16000,
    mono: bool = True,
) -> Tuple[str, np.ndarray, int]:
    """Return a normalized audio file path suitable for ASR.

    For audio files, returns the original path.
    For video files, extracts audio to a temp wav.

    Args:
        media_path: Path to input media.
        target_sr: Target sample rate.
        mono: Whether to convert to mono.

    Returns:
        (audio_path, waveform, sample_rate)
    """
    media_path = Path(media_path)
    suffix = media_path.suffix.lower()

    if is_video(media_path) or suffix in _SOUNDFILE_UNSUPPORTED:
        # Video or audio format not natively supported by soundfile → convert via ffmpeg.
        if not has_ffmpeg():
            raise RuntimeError(
                f"ffmpeg is required for .{suffix[1:]} files. "
                "Install with: brew install ffmpeg"
            )
        return extract_audio_from_video(media_path, target_sr=target_sr, mono=mono)

    data, sr = sf.read(str(media_path), dtype="float32", always_2d=False)
    if mono and data.ndim > 1:
        data = np.mean(data, axis=1)
    return str(media_path), data, sr
