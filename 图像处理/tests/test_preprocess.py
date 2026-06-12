from __future__ import annotations

import numpy as np

from src.social_ocr.postprocess import score_result
from src.social_ocr.ocr_engine import PaddleOcrEngine
from src.social_ocr.preprocess import generate_variants
from src.social_ocr.pipeline import _dedupe_text_lines
from src.social_ocr.video import build_sampling_plan, extract_video_frames, is_probable_video_file


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


def test_video_frame_extraction(tmp_path) -> None:
    import cv2

    video_path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (120, 80),
    )
    for index in range(12):
        frame = np.full((80, 120, 3), 255, dtype=np.uint8)
        cv2.putText(frame, f"T{index}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        writer.write(frame)
    writer.release()

    frames = extract_video_frames(video_path, interval_seconds=2, max_frames=3)

    assert is_probable_video_file(video_path)
    assert 1 <= len(frames) <= 3
    assert frames[0].image.shape[:2] == (80, 120)


def test_html_bin_is_not_video(tmp_path) -> None:
    html_path = tmp_path / "video_page.bin"
    html_path.write_bytes(b"<!doctype html><html></html>")

    assert not is_probable_video_file(html_path)


def test_default_sampling_plan_uses_two_seconds_under_64s() -> None:
    plan = build_sampling_plan(50.0, interval_seconds=None, max_frames=32)

    assert plan.interval_seconds == 2.0
    assert len(plan.timestamps) == 26
    assert plan.timestamps[:4] == [0.0, 2.0, 4.0, 6.0]


def test_default_sampling_plan_spreads_long_video_over_32_frames() -> None:
    plan = build_sampling_plan(128.0, interval_seconds=None, max_frames=32)

    assert plan.interval_seconds == 4.0
    assert len(plan.timestamps) == 32
    assert plan.timestamps[0] == 0.0
    assert plan.timestamps[-1] == 124.0


def test_video_llm_text_dedupes_neighbor_repeats() -> None:
    lines = _dedupe_text_lines(
        [
            "中国队为什么没进世界杯？\n足球",
            "中国队为什么没进世界杯？\n足球",
            "中国队为什么没进世界杯？",
            "佛得角\n库拉索",
        ]
    )

    assert lines == ["中国队为什么没进世界杯？", "足球", "佛得角", "库拉索"]
