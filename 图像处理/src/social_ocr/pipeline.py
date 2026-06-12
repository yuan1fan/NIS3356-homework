from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .ocr_engine import PaddleOcrEngine
from .postprocess import choose_best_result, score_result
from .preprocess import blur_score, generate_variants, read_image, text_area_ratio, write_image
from .video import (
    VIDEO_LIKE_SUFFIXES,
    build_sampling_plan,
    crop_frame_regions,
    extract_video_frames,
    get_video_metadata,
    is_probable_video_file,
    save_video_frames,
)
from .visualize import draw_text_blocks


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ProgressCallback = Callable[[dict[str, Any]], None]


def process_image(
    image_path: str | Path,
    output_dir: str | Path = "outputs",
    platform: str = "unknown",
    engine: PaddleOcrEngine | None = None,
    save_variants: bool = True,
    variant_names: set[str] | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    image_path = Path(image_path)
    engine = engine or PaddleOcrEngine(device=device)
    return process_image_array(
        read_image(image_path),
        image_id=image_path.stem,
        source_path=image_path,
        output_dir=output_dir,
        platform=platform,
        engine=engine,
        save_variants=save_variants,
        variant_names=variant_names,
        device=device,
        media_type="image",
    )


def process_image_array(
    image: np.ndarray,
    image_id: str,
    source_path: str | Path,
    output_dir: str | Path = "outputs",
    platform: str = "unknown",
    engine: PaddleOcrEngine | None = None,
    save_variants: bool = True,
    variant_names: set[str] | None = None,
    device: str = "cpu",
    media_type: str = "image",
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    source_path = Path(source_path)
    engine = engine or PaddleOcrEngine(device=device)

    variants = generate_variants(image)
    if variant_names is not None:
        variants = [variant for variant in variants if variant.name in variant_names]
    if not variants:
        raise ValueError("No preprocessing variants selected.")

    variant_records: list[dict[str, Any]] = []
    for variant in variants:
        ocr_result = engine.recognize(variant.image, variant.name)
        boxes = [block.box for block in ocr_result.text_blocks]
        area_ratio = text_area_ratio(boxes, variant.image.shape)
        score = score_result(
            avg_confidence=ocr_result.avg_confidence,
            char_count=ocr_result.char_count,
            block_count=len(ocr_result.text_blocks),
            low_confidence_ratio=ocr_result.low_confidence_ratio,
            text_area_ratio=area_ratio,
        )

        variant_path = None
        if save_variants:
            variant_path = output_dir / "variants" / image_id / f"{variant.name}.png"
            write_image(variant_path, variant.image)

        variant_records.append(
            {
                "variant_name": variant.name,
                "description": variant.description,
                "score": score,
                "avg_confidence": round(ocr_result.avg_confidence, 4),
                "min_confidence": round(ocr_result.min_confidence, 4),
                "low_confidence_ratio": round(ocr_result.low_confidence_ratio, 4),
                "char_count": ocr_result.char_count,
                "text_block_count": len(ocr_result.text_blocks),
                "text_area_ratio": round(area_ratio, 4),
                "blur_score": round(blur_score(variant.image), 4),
                "full_text": ocr_result.full_text,
                "text_blocks": [block.__dict__ for block in ocr_result.text_blocks],
                "variant_image_path": str(variant_path) if variant_path else None,
                "_image_for_visualization": variant.image,
            }
        )

    best = choose_best_result(variant_records)
    visualization_path = output_dir / "visualizations" / f"{image_id}_{best['variant_name']}.png"
    draw_text_blocks(best["_image_for_visualization"], best["text_blocks"], visualization_path)

    clean_variant_records = []
    for record in variant_records:
        cleaned = dict(record)
        cleaned.pop("_image_for_visualization", None)
        clean_variant_records.append(cleaned)

    result = {
        "image_id": image_id,
        "media_id": image_id,
        "media_type": media_type,
        "image_path": str(source_path),
        "source_media": str(source_path),
        "platform": platform,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "best_variant": best["variant_name"],
        "best_score": best["score"],
        "ocr_text": best["full_text"],
        "text_blocks": best["text_blocks"],
        "quality_assessment": {
            "avg_confidence": best["avg_confidence"],
            "min_confidence": best["min_confidence"],
            "low_confidence_ratio": best["low_confidence_ratio"],
            "text_block_count": best["text_block_count"],
            "char_count": best["char_count"],
            "text_area_ratio": best["text_area_ratio"],
            "blur_score": best["blur_score"],
            "needs_review": bool(
                best["avg_confidence"] < 0.75
                or best["char_count"] < 8
                or best["low_confidence_ratio"] > 0.35
            ),
        },
        "variant_results": clean_variant_records,
        "visualization_path": str(visualization_path),
        "for_llm_summary": _make_llm_summary(platform, best),
    }
    if extra_metadata:
        result["metadata"] = extra_metadata

    json_path = output_dir / "json" / f"{image_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    llm_result = make_llm_result(result)
    llm_json_path = output_dir / "llm_json" / f"{image_id}.json"
    llm_json_path.parent.mkdir(parents=True, exist_ok=True)
    llm_json_path.write_text(json.dumps(llm_result, ensure_ascii=False, indent=2), encoding="utf-8")

    result["json_path"] = str(json_path)
    result["llm_json_path"] = str(llm_json_path)
    return result


def process_video(
    video_path: str | Path,
    output_dir: str | Path = "outputs",
    platform: str = "unknown",
    engine: PaddleOcrEngine | None = None,
    frame_interval_seconds: float | None = None,
    max_frames: int = 32,
    frame_regions: tuple[str, ...] = ("full", "bottom"),
    save_frames: bool = True,
    variant_names: set[str] | None = None,
    device: str = "cpu",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    engine = engine or PaddleOcrEngine(device=device)
    video_id = video_path.stem

    metadata = get_video_metadata(video_path)
    sampling_plan = build_sampling_plan(
        float(metadata.get("duration_seconds", 0.0) or 0.0),
        frame_interval_seconds,
        max_frames,
    )
    frames = extract_video_frames(
        video_path,
        interval_seconds=sampling_plan.interval_seconds,
        max_frames=max_frames,
    )
    saved_frame_paths = (
        save_video_frames(frames, output_dir / "video_frames" / video_id, video_id)
        if save_frames
        else []
    )

    frame_results: list[dict[str, Any]] = []
    if progress_callback:
        progress_callback(
            {
                "event": "video_start",
                "path": str(video_path),
                "media_id": video_id,
                "frame_count": len(frames),
                "region_count": len(frame_regions),
            }
        )
    for frame in frames:
        for region in crop_frame_regions(frame.image, frame_regions):
            region_id = f"{video_id}_t{frame.timestamp_seconds:.1f}_{region.name}"
            result = process_image_array(
                region.image,
                image_id=region_id,
                source_path=video_path,
                output_dir=output_dir / "frame_ocr",
                platform=platform,
                engine=engine,
                save_variants=False,
                variant_names=variant_names,
                device=device,
                media_type="video_frame",
                extra_metadata={
                    "video_path": str(video_path),
                    "frame_index": frame.index,
                    "timestamp_seconds": frame.timestamp_seconds,
                    "region": region.name,
                    "crop_box": region.crop_box,
                },
            )
            frame_results.append(
                {
                    "frame_index": frame.index,
                    "timestamp_seconds": frame.timestamp_seconds,
                    "region": region.name,
                    "crop_box": region.crop_box,
                    "ocr_text": result["ocr_text"],
                    "quality_assessment": result["quality_assessment"],
                    "best_variant": result["best_variant"],
                    "json_path": result["json_path"],
                    "llm_json_path": result["llm_json_path"],
                    "visualization_path": result["visualization_path"],
                }
            )
            if progress_callback:
                progress_callback(
                    {
                        "event": "advance",
                        "kind": "video_frame",
                        "path": str(video_path),
                        "media_id": video_id,
                        "timestamp_seconds": frame.timestamp_seconds,
                        "region": region.name,
                        "text_chars": result["quality_assessment"]["char_count"],
                    }
                )

    text_lines = _dedupe_text_lines(item["ocr_text"] for item in frame_results)
    quality = _summarize_frame_quality(frame_results)
    result = {
        "image_id": video_id,
        "media_id": video_id,
        "media_type": "video",
        "video_path": str(video_path),
        "source_media": str(video_path),
        "platform": platform,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "video_metadata": metadata,
        "frame_sampling": {
            "frame_interval_seconds": sampling_plan.interval_seconds,
            "max_frames": max_frames,
            "sampled_frame_count": len(frames),
            "sampled_timestamps": [frame.timestamp_seconds for frame in frames],
            "planned_timestamps": sampling_plan.timestamps,
            "frame_regions": list(frame_regions),
            "saved_frame_paths": saved_frame_paths,
        },
        "ocr_text": "\n".join(text_lines),
        "frame_results": frame_results,
        "quality_assessment": quality,
        "for_llm_summary": _make_video_llm_summary(platform, video_id, text_lines, quality),
    }

    json_path = output_dir / "json" / f"{video_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    llm_result = make_llm_result(result)
    llm_json_path = output_dir / "llm_json" / f"{video_id}.json"
    llm_json_path.parent.mkdir(parents=True, exist_ok=True)
    llm_json_path.write_text(json.dumps(llm_result, ensure_ascii=False, indent=2), encoding="utf-8")

    result["json_path"] = str(json_path)
    result["llm_json_path"] = str(llm_json_path)
    return result


def process_batch(
    input_dir: str | Path = "data/raw",
    output_dir: str | Path = "outputs",
    platform: str = "unknown",
    patterns: tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"),
    limit: int | None = None,
    variant_names: set[str] | None = None,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    input_dir = Path(input_dir)
    image_paths: list[Path] = []
    for pattern in patterns:
        image_paths.extend(sorted(input_dir.glob(pattern)))
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    engine = PaddleOcrEngine(device=device)
    results = [
        process_image(
            path,
            output_dir=output_dir,
            platform=platform,
            engine=engine,
            variant_names=variant_names,
            device=device,
        )
        for path in image_paths
    ]

    summary_path = Path(output_dir) / "reports" / "batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    llm_summary_path = Path(output_dir) / "reports" / "llm_batch_summary.json"
    llm_summary = [make_llm_result(result) for result in results]
    llm_summary_path.write_text(json.dumps(llm_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def process_media_batch(
    input_dir: str | Path,
    output_dir: str | Path = "outputs",
    platform: str = "unknown",
    image_limit: int | None = None,
    video_limit: int | None = None,
    recursive: bool = True,
    variant_names: set[str] | None = None,
    device: str = "cpu",
    frame_interval_seconds: float | None = None,
    max_video_frames: int = 32,
    frame_regions: tuple[str, ...] = ("full", "bottom"),
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    globber = input_dir.rglob if recursive else input_dir.glob
    files = [path for path in globber("*") if path.is_file()]
    image_paths = sorted(path for path in files if path.suffix.lower() in IMAGE_SUFFIXES)
    video_paths = sorted(path for path in files if path.suffix.lower() in VIDEO_LIKE_SUFFIXES)

    if image_limit is not None:
        image_paths = image_paths[:image_limit]
    if video_limit is not None:
        video_paths = video_paths[:video_limit]
    if not image_paths and not video_paths:
        raise FileNotFoundError(f"No supported media found in {input_dir}")

    engine = PaddleOcrEngine(device=device)
    image_results = []
    for index, path in enumerate(image_paths, start=1):
        result = process_image(
            path,
            output_dir=output_dir,
            platform=platform,
            engine=engine,
            variant_names=variant_names,
            device=device,
        )
        image_results.append(result)
        if progress_callback:
            progress_callback(
                {
                    "event": "advance",
                    "kind": "image",
                    "path": str(path),
                    "media_id": result["media_id"],
                    "index": index,
                    "total": len(image_paths),
                    "text_chars": result["quality_assessment"]["char_count"],
                }
            )

    video_results = []
    skipped_videos = []
    for path in video_paths:
        if not is_probable_video_file(path):
            skipped_videos.append({"path": str(path), "reason": "not_probable_video"})
            continue
        try:
            video_results.append(
                process_video(
                    path,
                    output_dir=output_dir,
                    platform=platform,
                    engine=engine,
                    frame_interval_seconds=frame_interval_seconds,
                    max_frames=max_video_frames,
                    frame_regions=frame_regions,
                    variant_names=variant_names,
                    device=device,
                    progress_callback=progress_callback,
                )
            )
        except Exception as exc:  # noqa: BLE001
            skipped_videos.append({"path": str(path), "reason": str(exc)})

    report = {
        "input_dir": str(input_dir),
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "image_count": len(image_results),
        "video_count": len(video_results),
        "skipped_video_count": len(skipped_videos),
        "image_results": image_results,
        "video_results": video_results,
        "skipped_videos": skipped_videos,
    }
    summary_path = Path(output_dir) / "reports" / "media_batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    llm_summary = [make_llm_result(item) for item in image_results + video_results]
    llm_summary_path = Path(output_dir) / "reports" / "llm_media_batch_summary.json"
    llm_summary_path.write_text(json.dumps(llm_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def make_llm_result(result: dict[str, Any]) -> dict[str, Any]:
    quality = result.get("quality_assessment", {})
    data = {
        "media_id": result.get("media_id") or result.get("image_id"),
        "media_type": result.get("media_type", "image"),
        "platform": result.get("platform"),
        "source_media": result.get("source_media") or result.get("image_path") or result.get("video_path"),
        "processed_at": result.get("processed_at"),
        "ocr_text": result.get("ocr_text", ""),
        "ocr_text_compact": _compact_text(result.get("ocr_text", "")),
        "for_llm_summary": result.get("for_llm_summary", ""),
        "ocr_quality": {
            "avg_confidence": quality.get("avg_confidence"),
            "needs_review": quality.get("needs_review"),
            "char_count": quality.get("char_count"),
            "text_block_count": quality.get("text_block_count"),
        },
        "preprocess": {
            "best_variant": result.get("best_variant"),
            "score": result.get("best_score"),
        },
    }
    if result.get("media_type") == "video":
        data["video"] = {
            "duration_seconds": result.get("video_metadata", {}).get("duration_seconds"),
            "sampled_frame_count": result.get("frame_sampling", {}).get("sampled_frame_count"),
        }
    return data


def _make_llm_summary(platform: str, best: dict[str, Any]) -> str:
    text = best.get("full_text", "").replace("\n", "；")
    if len(text) > 180:
        text = text[:177] + "..."
    return (
        f"平台：{platform}。最佳OCR预处理版本：{best['variant_name']}。"
        f"平均置信度：{best['avg_confidence']:.2f}。"
        f"识别到的主要文字：{text}"
    )


def _compact_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "；".join(lines)


def _dedupe_text_lines(texts: Iterable[str]) -> list[str]:
    lines: list[str] = []
    seen_keys: set[str] = set()
    for text in texts:
        for line in str(text or "").splitlines():
            line = line.strip()
            if len(line) < 2:
                continue
            key = _text_dedupe_key(line)
            if not key or key in seen_keys or _is_near_duplicate_key(key, seen_keys):
                continue
            seen_keys.add(key)
            lines.append(line)
    return lines


def _text_dedupe_key(text: str) -> str:
    return re.sub(r"[\W_]+", "", text).lower()


def _is_near_duplicate_key(key: str, seen_keys: set[str]) -> bool:
    if len(key) < 6:
        return False
    for old_key in seen_keys:
        if len(old_key) < 6:
            continue
        shorter, longer = sorted((key, old_key), key=len)
        if shorter in longer and len(shorter) / len(longer) >= 0.72:
            return True
    return False


def _summarize_frame_quality(frame_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not frame_results:
        return {
            "avg_confidence": 0.0,
            "needs_review": True,
            "char_count": 0,
            "text_block_count": 0,
        }

    confidences = [
        item.get("quality_assessment", {}).get("avg_confidence", 0.0)
        for item in frame_results
    ]
    char_count = sum(
        item.get("quality_assessment", {}).get("char_count", 0)
        for item in frame_results
    )
    block_count = sum(
        item.get("quality_assessment", {}).get("text_block_count", 0)
        for item in frame_results
    )
    avg_conf = sum(confidences) / len(confidences)
    return {
        "avg_confidence": round(avg_conf, 4),
        "needs_review": avg_conf < 0.70 or char_count < 8,
        "char_count": char_count,
        "text_block_count": block_count,
    }


def _make_video_llm_summary(
    platform: str,
    video_id: str,
    text_lines: list[str],
    quality: dict[str, Any],
) -> str:
    text = "；".join(text_lines)
    if len(text) > 220:
        text = text[:217] + "..."
    return (
        f"平台：{platform}。视频ID：{video_id}。"
        f"抽帧OCR平均置信度：{quality.get('avg_confidence', 0):.2f}。"
        f"识别到的视频画面文字：{text}"
    )
