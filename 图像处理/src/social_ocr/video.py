from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .preprocess import write_image


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
VIDEO_LIKE_SUFFIXES = VIDEO_SUFFIXES | {".bin"}


@dataclass(frozen=True)
class VideoFrame:
    index: int
    timestamp_seconds: float
    image: np.ndarray


@dataclass(frozen=True)
class FrameRegion:
    name: str
    image: np.ndarray
    crop_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class SamplingPlan:
    duration_seconds: float
    interval_seconds: float
    max_frames: int
    timestamps: list[float]


def is_probable_video_file(path: str | Path) -> bool:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return True
    if suffix != ".bin" or not path.exists():
        return False
    header = path.read_bytes()[:256]
    lowered = header.lower()
    if lowered.startswith(b"<!doctype") or lowered.startswith(b"<html"):
        return False
    return b"ftyp" in header[:64] or header.startswith(b"\x1aE\xdf\xa3") or header.startswith(b"RIFF")


def get_video_metadata(path: str | Path) -> dict[str, float | int]:
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        return {
            "fps": round(fps, 4),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": round(duration, 3),
        }
    finally:
        capture.release()


def extract_video_frames(
    path: str | Path,
    interval_seconds: float | None = None,
    max_frames: int = 32,
) -> list[VideoFrame]:
    path = Path(path)
    if not is_probable_video_file(path):
        raise ValueError(f"Not a supported video file: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Cannot open video: {path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        timestamps = build_sampling_plan(duration, interval_seconds, max_frames).timestamps
        frames: list[VideoFrame] = []
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frame_index = int(round(timestamp * fps)) if fps > 0 else len(frames)
            frames.append(VideoFrame(frame_index, round(timestamp, 3), frame))

        if not frames:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(VideoFrame(0, 0.0, frame))
        return frames
    finally:
        capture.release()


def build_sampling_plan(
    duration_seconds: float,
    interval_seconds: float | None = None,
    max_frames: int = 32,
) -> SamplingPlan:
    max_frames = max(1, int(max_frames))
    duration_seconds = max(0.0, float(duration_seconds or 0.0))
    if interval_seconds is None:
        interval_seconds = _auto_interval(duration_seconds, max_frames)
    interval_seconds = max(2.0, float(interval_seconds))
    timestamps = _sample_timestamps(duration_seconds, interval_seconds, max_frames)
    return SamplingPlan(
        duration_seconds=round(duration_seconds, 3),
        interval_seconds=round(interval_seconds, 3),
        max_frames=max_frames,
        timestamps=timestamps,
    )


def crop_frame_regions(image: np.ndarray, region_names: Iterable[str]) -> list[FrameRegion]:
    h, w = image.shape[:2]
    regions: list[FrameRegion] = []
    for name in region_names:
        name = name.strip().lower()
        if name == "full":
            box = (0, 0, w, h)
        elif name == "top":
            box = (0, 0, w, max(1, int(h * 0.36)))
        elif name == "center":
            y1 = int(h * 0.22)
            y2 = max(y1 + 1, int(h * 0.78))
            box = (0, y1, w, y2)
        elif name == "bottom":
            box = (0, int(h * 0.58), w, h)
        else:
            continue
        x1, y1, x2, y2 = box
        regions.append(FrameRegion(name=name, image=image[y1:y2, x1:x2].copy(), crop_box=box))
    return regions


def save_video_frames(frames: list[VideoFrame], output_dir: str | Path, video_id: str) -> list[str]:
    output_dir = Path(output_dir)
    paths: list[str] = []
    for frame in frames:
        path = output_dir / f"{video_id}_t{frame.timestamp_seconds:.1f}.jpg"
        write_image(path, frame.image)
        paths.append(str(path))
    return paths


def _sample_timestamps(duration_seconds: float, interval_seconds: float, max_frames: int) -> list[float]:
    max_frames = max(1, int(max_frames))
    if duration_seconds <= 0:
        return [0.0]

    interval_seconds = max(2.0, float(interval_seconds))
    timestamps = [0.0]
    current = interval_seconds
    while current < duration_seconds and len(timestamps) < max_frames:
        timestamps.append(round(current, 3))
        current += interval_seconds

    if len(timestamps) < max_frames and duration_seconds > 1:
        last = max(0.0, duration_seconds - 0.25)
        if all(abs(last - item) > 0.5 for item in timestamps):
            timestamps.append(round(last, 3))
    return timestamps[:max_frames]


def _auto_interval(duration_seconds: float, max_frames: int) -> float:
    if duration_seconds <= 0:
        return 2.0
    if duration_seconds < 64:
        return 2.0
    return max(2.0, duration_seconds / max(1, max_frames))
