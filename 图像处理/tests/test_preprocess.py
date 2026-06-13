from __future__ import annotations

import numpy as np
from PIL import Image

from src.social_ocr.postprocess import score_result
from src.social_ocr.ocr_engine import PaddleOcrEngine, parse_paddle_result
from src.social_ocr.preprocess import generate_variants, read_image
from src.social_ocr import pipeline
from src.social_ocr.pipeline import _compact_text, _dedupe_text_lines, _summarize_frame_preprocess
from src.social_ocr.linebreak_repair import should_join_ocr_lines
from src.social_ocr.video import build_sampling_plan, extract_video_frames, is_probable_video_file


def test_generate_variants_non_empty() -> None:
    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    variants = generate_variants(image)
    assert len(variants) >= 6
    assert all(variant.image.size > 0 for variant in variants)


def test_gif_content_with_jpg_suffix_reads_first_frame(tmp_path) -> None:
    path = tmp_path / "animated.jpg"
    frame1 = Image.new("RGB", (32, 24), (255, 255, 255))
    frame2 = Image.new("RGB", (32, 24), (0, 0, 0))
    frame1.save(path, format="GIF", save_all=True, append_images=[frame2], duration=100, loop=0)

    image = read_image(path)

    assert image.shape == (24, 32, 3)
    assert image.dtype == np.uint8


def test_media_batch_skips_unreadable_images(tmp_path, monkeypatch) -> None:
    bad_image = tmp_path / "bad.jpg"
    bad_image.write_bytes(b"not an image")
    monkeypatch.setattr(pipeline, "PaddleOcrEngine", lambda device="cpu": object())

    report = pipeline.process_media_batch(tmp_path, output_dir=tmp_path / "out")

    assert report["image_count"] == 0
    assert report["skipped_image_count"] == 1
    assert report["skipped_images"][0]["path"].endswith("bad.jpg")


def test_score_rewards_confidence_and_text() -> None:
    weak = score_result(0.40, 5, 1, 0.9, 0.01)
    strong = score_result(0.95, 80, 8, 0.0, 0.08)
    assert strong > weak


def test_ocr_engine_methods_exist() -> None:
    assert callable(PaddleOcrEngine.recognize)


def test_empty_numpy_ocr_fields_parse_as_no_blocks() -> None:
    raw = [{"rec_texts": [], "rec_scores": np.array([]), "rec_polys": np.array([])}]

    blocks = parse_paddle_result(raw, image_width=320, image_height=240)

    assert blocks == []


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


def test_compact_text_removes_repeated_source_watermarks() -> None:
    compact = _compact_text(
        "\n".join(
            [
                "400多名老人遭养生诈骗",
                "@北京日报",
                "店员以提供免费按摩、低价足疗券等",
                "@北京日板",
                "以其身体状况不好为由，引荐所谓的专家",
                "@北京日報",
                "涉案超3000万",
            ]
        )
    )

    assert "400多名老人遭养生诈骗" in compact
    assert "店员以提供免费按摩" in compact
    assert "涉案超3000万" in compact
    assert "北京日" not in compact


def test_compact_text_repairs_ocr_hard_linebreaks() -> None:
    compact = _compact_text(
        "\n".join(
            [
                "400多名老人遭养生诈骗",
                "自报",
                "涉案超3000万",
                "北京警方近期打掉一个专门针对老",
                "年人的诈骗团伙，抓获31名犯罪嫌",
                "疑人，涉及朝阳、顺义、平谷、密",
                "云4个区20家门店",
                "店员以提供免费按摩、低价足疗券等",
                "方式将老年人吸引至店内，按摩过程",
                "中通过聊天锁定一些子女不在身边、",
                "经济条件较好的老年人",
                "以其身体状况不好为由，引荐所谓",
                "的“专家”做免费体检。并虚构各",
                "种病症，称如不及时治疗将危及生",
                "命，诱骗充值高额治疗费用，涉及",
                "肠道清洗、祛湿排毒等多个项目",
                "每个项目单次收费1万至2万元不等",
                "共计400余老年人被骗，涉案金额",
                "3000万余元",
            ]
        )
    )

    assert "老；年人" not in compact
    assert "犯罪嫌；疑人" not in compact
    assert "平谷、密；云" not in compact
    assert "涉案金额3000万余元" in compact
    assert "400多名老人遭养生诈骗；自报；涉案超3000万；北京警方" in compact
    assert "20家门店；店员以提供免费按摩" in compact


def test_linebreak_repair_uses_boundary_classifier_for_ambiguous_breaks() -> None:
    assert should_join_ocr_lines("网络平台需要对热点话题中的文本", "图片和视频进行采集")
    assert not should_join_ocr_lines("涉案超3000万", "北京警方近期打掉一个专门针对老年人的诈骗团伙")


def test_video_preprocess_summary_uses_common_variant_and_average_score() -> None:
    summary = _summarize_frame_preprocess(
        [
            {"best_variant": "original", "best_score": 0.8},
            {"best_variant": "clahe", "best_score": 0.6},
            {"best_variant": "original", "best_score": 1.0},
        ]
    )

    assert summary == {"best_variant": "original", "best_score": 0.8}
