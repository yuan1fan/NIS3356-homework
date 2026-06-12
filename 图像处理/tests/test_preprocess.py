from __future__ import annotations

import numpy as np

from src.social_ocr.postprocess import score_result
from src.social_ocr.ocr_engine import PaddleOcrEngine
from src.social_ocr.preprocess import generate_variants


def test_generate_variants_non_empty() -> None:
    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    variants = generate_variants(image)
    assert len(variants) >= 6
    assert all(variant.image.size > 0 for variant in variants)


def test_score_rewards_confidence_and_text() -> None:
    weak = score_result(0.40, 5, 1, 0.9, 0.01)
    strong = score_result(0.95, 80, 8, 0.0, 0.08)
    assert strong > weak


def test_ocr_engine_methods_exist() -> None:
    assert callable(PaddleOcrEngine.recognize)
