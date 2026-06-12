from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextBlock:
    text: str
    confidence: float
    box: list[list[float]]
    region_type: str = "unknown"


@dataclass
class OcrResult:
    variant_name: str
    text_blocks: list[TextBlock] = field(default_factory=list)
    raw_result: Any = None

    @property
    def full_text(self) -> str:
        return "\n".join(block.text for block in self.text_blocks if block.text).strip()

    @property
    def avg_confidence(self) -> float:
        if not self.text_blocks:
            return 0.0
        return sum(block.confidence for block in self.text_blocks) / len(self.text_blocks)

    @property
    def min_confidence(self) -> float:
        if not self.text_blocks:
            return 0.0
        return min(block.confidence for block in self.text_blocks)

    @property
    def low_confidence_ratio(self) -> float:
        if not self.text_blocks:
            return 1.0
        low_count = sum(1 for block in self.text_blocks if block.confidence < 0.65)
        return low_count / len(self.text_blocks)

    @property
    def char_count(self) -> int:
        return len(re.sub(r"\s+", "", self.full_text))


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_region_type(box: list[list[float]], image_width: int, image_height: int) -> str:
    if not box:
        return "unknown"
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    x_center = (min(xs) + max(xs)) / 2
    y_center = (min(ys) + max(ys)) / 2
    height = max(ys) - min(ys)

    if y_center < image_height * 0.18:
        return "header"
    if y_center > image_height * 0.82:
        return "footer"
    if height > image_height * 0.055 and x_center < image_width * 0.75:
        return "title"
    if x_center > image_width * 0.75:
        return "side_or_metric"
    return "body"


def choose_best_result(results: list[dict]) -> dict:
    if not results:
        raise ValueError("No OCR results to choose from.")
    return max(results, key=lambda item: item["score"])


def score_result(
    avg_confidence: float,
    char_count: int,
    block_count: int,
    low_confidence_ratio: float,
    text_area_ratio: float,
) -> float:
    # The score rewards confident, non-empty OCR and mildly rewards text coverage.
    char_score = min(char_count / 120.0, 1.0)
    block_score = min(block_count / 12.0, 1.0)
    area_score = min(text_area_ratio / 0.20, 1.0)
    score = (
        avg_confidence * 0.55
        + char_score * 0.20
        + block_score * 0.10
        + area_score * 0.10
        + (1.0 - low_confidence_ratio) * 0.05
    )
    return round(float(score), 4)

