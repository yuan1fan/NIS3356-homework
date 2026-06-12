from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class ImageVariant:
    name: str
    image: np.ndarray
    description: str


def read_image(path: str | Path) -> np.ndarray:
    """Read an image from paths that may contain non-ASCII characters."""
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write an image to paths that may contain non-ASCII characters."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, buffer = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image as {suffix}: {path}")
    buffer.tofile(str(path))


def limit_long_side(image: np.ndarray, max_side: int = 1600) -> np.ndarray:
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return image.copy()
    scale = max_side / long_side
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def to_gray_bgr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def clahe_enhance(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    merged = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def sharpen(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def denoise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)


def adaptive_binary(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def upscale(image: np.ndarray, factor: float = 1.5) -> np.ndarray:
    h, w = image.shape[:2]
    return cv2.resize(
        image,
        (max(1, int(w * factor)), max(1, int(h * factor))),
        interpolation=cv2.INTER_CUBIC,
    )


def generate_variants(image: np.ndarray, max_side: int = 1600) -> list[ImageVariant]:
    base = limit_long_side(image, max_side=max_side)

    transforms: list[tuple[str, str, Callable[[np.ndarray], np.ndarray]]] = [
        ("original", "Original image with only max-side normalization.", lambda x: x.copy()),
        ("gray", "Gray-scale image converted back to BGR.", to_gray_bgr),
        ("clahe", "CLAHE contrast enhancement on luminance channel.", clahe_enhance),
        ("sharpen", "Unsharp masking for small or compressed text.", sharpen),
        ("denoise", "Color non-local means denoising.", denoise),
        ("binary", "Adaptive binary thresholding.", adaptive_binary),
        ("upscale", "1.5x bicubic upscaling for small text.", upscale),
        (
            "clahe_sharpen",
            "CLAHE enhancement followed by sharpening.",
            lambda x: sharpen(clahe_enhance(x)),
        ),
    ]

    return [
        ImageVariant(name=name, image=transform(base), description=description)
        for name, description, transform in transforms
    ]


def blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def text_area_ratio(boxes: list[list[list[float]]], image_shape: tuple[int, ...]) -> float:
    h, w = image_shape[:2]
    area = max(1.0, float(h * w))
    total = 0.0
    for box in boxes:
        points = np.array(box, dtype=np.float32)
        total += abs(float(cv2.contourArea(points)))
    return min(total / area, 1.0)

