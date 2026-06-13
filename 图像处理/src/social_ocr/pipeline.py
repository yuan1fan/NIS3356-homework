from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .captioning import (
    DEFAULT_CAPTION_MODEL,
    ImageCaptioner,
    attach_caption_to_visual_semantics,
    should_generate_caption,
)
from .clip_semantics import ChineseClipAnalyzer, summarize_video_visual_semantics
from .ocr_engine import PaddleOcrEngine
from .paddle_ocr_worker import PaddleOcrWorker
from .linebreak_repair import repair_ocr_linebreaks
from .postprocess import choose_best_result, score_result
from .preprocess import blur_score, generate_variants, read_image, text_area_ratio, write_image
from .torch_visual_worker import TorchVisualWorker
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


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
ProgressCallback = Callable[[dict[str, Any]], None]
SOURCE_LINE_KEYWORDS = (
    "日报",
    "日報",
    "日板",
    "晚报",
    "时报",
    "新闻",
    "电视台",
    "融媒",
    "发布",
    "客户端",
    "观察",
    "网",
)


class _LazyCaptioner:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self._captioner: ImageCaptioner | None = None

    def caption_image(self, image: np.ndarray) -> dict[str, Any]:
        if self._captioner is None:
            self._captioner = ImageCaptioner(model_name=self.model_name, device=self.device)
        return self._captioner.caption_image(image)


def _create_ocr_engine(device: str) -> Any:
    if device.startswith("gpu"):
        return PaddleOcrWorker(device=device)
    return PaddleOcrEngine(device=device)


def _create_visual_backends(
    enable_clip: bool,
    enable_caption: bool,
    clip_model: str,
    caption_model: str,
    device: str,
) -> tuple[Any | None, Any | None, TorchVisualWorker | None]:
    if not enable_clip and not enable_caption:
        return None, None, None

    worker = TorchVisualWorker(
        clip_model=clip_model,
        caption_model=caption_model,
        device=device,
    )
    clip_analyzer = worker if enable_clip else None
    captioner = worker if enable_caption else None
    return clip_analyzer, captioner, worker


def process_image(
    image_path: str | Path,
    output_dir: str | Path = "outputs",
    platform: str = "unknown",
    engine: PaddleOcrEngine | None = None,
    save_variants: bool = True,
    variant_names: set[str] | None = None,
    device: str = "cpu",
    clip_analyzer: ChineseClipAnalyzer | None = None,
    enable_clip: bool = False,
    clip_model: str = "OFA-Sys/chinese-clip-vit-base-patch16",
    captioner: ImageCaptioner | None = None,
    enable_caption: bool = False,
    caption_model: str = DEFAULT_CAPTION_MODEL,
) -> dict[str, Any]:
    image_path = Path(image_path)
    worker: TorchVisualWorker | None = None
    own_engine = engine is None
    if clip_analyzer is None and captioner is None:
        clip_analyzer, captioner, worker = _create_visual_backends(
            enable_clip,
            enable_caption,
            clip_model,
            caption_model,
            device,
        )
    engine = engine or _create_ocr_engine(device)
    try:
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
            clip_analyzer=clip_analyzer,
            captioner=captioner,
        )
    finally:
        if worker:
            worker.close()
        if own_engine and isinstance(engine, PaddleOcrWorker):
            engine.close()


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
    clip_analyzer: ChineseClipAnalyzer | None = None,
    captioner: ImageCaptioner | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    source_path = Path(source_path)
    engine = engine or _create_ocr_engine(device)

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
    }
    if clip_analyzer:
        result["visual_semantics"] = clip_analyzer.analyze_image(image, best["full_text"])
        if captioner and should_generate_caption(result["visual_semantics"]):
            result["visual_semantics"] = attach_caption_to_visual_semantics(
                result["visual_semantics"],
                captioner.caption_image(image),
            )
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
    clip_analyzer: ChineseClipAnalyzer | None = None,
    enable_clip: bool = False,
    clip_model: str = "OFA-Sys/chinese-clip-vit-base-patch16",
    captioner: ImageCaptioner | None = None,
    enable_caption: bool = False,
    caption_model: str = DEFAULT_CAPTION_MODEL,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    worker: TorchVisualWorker | None = None
    own_engine = engine is None
    if clip_analyzer is None and captioner is None:
        clip_analyzer, captioner, worker = _create_visual_backends(
            enable_clip,
            enable_caption,
            clip_model,
            caption_model,
            device,
        )
    engine = engine or _create_ocr_engine(device)
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
                clip_analyzer=clip_analyzer,
                captioner=None,
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
                    "best_score": result["best_score"],
                    "json_path": result["json_path"],
                    "llm_json_path": result["llm_json_path"],
                    "visualization_path": result["visualization_path"],
                    "visual_semantics": result.get("visual_semantics"),
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
    preprocess = _summarize_frame_preprocess(frame_results)
    visual_semantics = summarize_video_visual_semantics(frame_results)
    if visual_semantics and captioner and should_generate_caption(visual_semantics):
        caption_source = _select_caption_frame(frames, frame_results)
        if caption_source:
            frame, frame_result = caption_source
            visual_semantics = attach_caption_to_visual_semantics(
                visual_semantics,
                captioner.caption_image(frame.image),
                source={
                    "frame_index": frame.index,
                    "timestamp_seconds": frame.timestamp_seconds,
                    "region": frame_result.get("region"),
                },
            )
    if worker:
        worker.close()
    if own_engine and isinstance(engine, PaddleOcrWorker):
        engine.close()
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
        "best_variant": preprocess["best_variant"],
        "best_score": preprocess["best_score"],
        "ocr_text": "\n".join(text_lines),
        "frame_results": frame_results,
        "quality_assessment": quality,
    }
    if visual_semantics:
        result["visual_semantics"] = visual_semantics

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
    enable_clip: bool = False,
    clip_model: str = "OFA-Sys/chinese-clip-vit-base-patch16",
    enable_caption: bool = False,
    caption_model: str = DEFAULT_CAPTION_MODEL,
) -> list[dict[str, Any]]:
    input_dir = Path(input_dir)
    image_paths: list[Path] = []
    for pattern in patterns:
        image_paths.extend(sorted(input_dir.glob(pattern)))
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    clip_analyzer, captioner, worker = _create_visual_backends(
        enable_clip,
        enable_caption,
        clip_model,
        caption_model,
        device,
    )
    engine = _create_ocr_engine(device)
    try:
        results = [
            process_image(
                path,
                output_dir=output_dir,
                platform=platform,
                engine=engine,
                variant_names=variant_names,
                device=device,
                clip_analyzer=clip_analyzer,
                captioner=captioner,
            )
            for path in image_paths
        ]
    finally:
        if worker:
            worker.close()
        if isinstance(engine, PaddleOcrWorker):
            engine.close()

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
    enable_clip: bool = False,
    clip_model: str = "OFA-Sys/chinese-clip-vit-base-patch16",
    enable_caption: bool = False,
    caption_model: str = DEFAULT_CAPTION_MODEL,
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

    clip_analyzer, captioner, worker = _create_visual_backends(
        enable_clip,
        enable_caption,
        clip_model,
        caption_model,
        device,
    )
    engine = _create_ocr_engine(device)
    image_results = []
    skipped_images = []
    try:
        for index, path in enumerate(image_paths, start=1):
            try:
                result = process_image(
                    path,
                    output_dir=output_dir,
                    platform=platform,
                    engine=engine,
                    variant_names=variant_names,
                    device=device,
                    clip_analyzer=clip_analyzer,
                    captioner=captioner,
                )
            except Exception as exc:  # noqa: BLE001
                skipped_images.append({"path": str(path), "reason": str(exc)})
                if progress_callback:
                    progress_callback(
                        {
                            "event": "advance",
                            "kind": "image_skipped",
                            "path": str(path),
                            "media_id": path.stem,
                            "index": index,
                            "total": len(image_paths),
                            "text_chars": 0,
                        }
                    )
                continue
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
                        clip_analyzer=clip_analyzer,
                        captioner=captioner,
                        progress_callback=progress_callback,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                skipped_videos.append({"path": str(path), "reason": str(exc)})
    finally:
        if worker:
            worker.close()
        if isinstance(engine, PaddleOcrWorker):
            engine.close()

    report = {
        "input_dir": str(input_dir),
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "image_count": len(image_results),
        "skipped_image_count": len(skipped_images),
        "video_count": len(video_results),
        "skipped_video_count": len(skipped_videos),
        "image_results": image_results,
        "skipped_images": skipped_images,
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
    if result.get("visual_semantics"):
        data["visual_semantics"] = result["visual_semantics"]
    if result.get("media_type") == "video":
        data["video"] = {
            "duration_seconds": result.get("video_metadata", {}).get("duration_seconds"),
            "sampled_frame_count": result.get("frame_sampling", {}).get("sampled_frame_count"),
        }
    return data


def _compact_text(text: str) -> str:
    lines = _clean_llm_text_lines(text.splitlines())
    return repair_ocr_linebreaks(lines)


def _dedupe_text_lines(texts: Iterable[str]) -> list[str]:
    lines: list[str] = []
    seen_keys: set[str] = set()
    for text in texts:
        for line in str(text or "").splitlines():
            cleaned_lines = _clean_llm_text_lines([line])
            if not cleaned_lines:
                continue
            line = cleaned_lines[0]
            key = _text_dedupe_key(line)
            if not key or key in seen_keys or _is_near_duplicate_key(key, seen_keys):
                continue
            seen_keys.add(key)
            lines.append(line)
    return lines


def _clean_llm_text_lines(lines: Iterable[str]) -> list[str]:
    cleaned: list[str] = []
    seen_keys: set[str] = set()
    source_counts: Counter[str] = Counter()
    for raw_line in lines:
        line = _normalize_compact_line(str(raw_line or ""))
        if len(line) < 2 or _is_low_value_source_line(line):
            continue
        key = _text_dedupe_key(line)
        if not key or key in seen_keys or _is_near_duplicate_key(key, seen_keys):
            continue
        seen_keys.add(key)
        if _looks_like_source_name(line):
            source_counts[key] += 1
            if source_counts[key] > 1:
                continue
        cleaned.append(line)
    return cleaned


def _normalize_compact_line(line: str) -> str:
    line = re.sub(r"\s+", "", line.strip())
    line = re.sub(
        r"@[\u4e00-\u9fffA-Za-z0-9]{0,8}"
        r"(?:日报|日報|日板|晚报|时报|新闻|电视台|融媒|发布|客户端|观察|网)",
        "",
        line,
    )
    line = re.sub(r"^[•·。:：,，;；|丨/\\]+", "", line)
    line = re.sub(r"[•·。:：,，;；|丨/\\]+$", "", line)
    return line


def _is_low_value_source_line(line: str) -> bool:
    key = _text_dedupe_key(line)
    if not key:
        return True
    if line.startswith("@") and len(key) <= 12:
        return True
    if _looks_like_source_name(line) and len(key) <= 8:
        return True
    return False


def _looks_like_source_name(line: str) -> bool:
    text = line.lstrip("@")
    if len(_text_dedupe_key(text)) > 12:
        return False
    return any(keyword in text for keyword in SOURCE_LINE_KEYWORDS)


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


def _select_caption_frame(frames: list[Any], frame_results: list[dict[str, Any]]) -> tuple[Any, dict[str, Any]] | None:
    if not frames or not frame_results:
        return None
    frame_by_index = {frame.index: frame for frame in frames}
    candidates = [
        item
        for item in frame_results
        if should_generate_caption(item.get("visual_semantics"))
        and item.get("frame_index") in frame_by_index
    ]
    if not candidates:
        candidates = [
            item
            for item in frame_results
            if item.get("frame_index") in frame_by_index
        ]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda item: (
            float(item.get("quality_assessment", {}).get("avg_confidence") or 0.0),
            float(item.get("best_score") or 0.0),
        ),
    )
    return frame_by_index[best["frame_index"]], best


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


def _summarize_frame_preprocess(frame_results: list[dict[str, Any]]) -> dict[str, Any]:
    variants = [
        str(item.get("best_variant"))
        for item in frame_results
        if item.get("best_variant")
    ]
    scores = [
        float(item.get("best_score") or 0.0)
        for item in frame_results
        if item.get("best_score") is not None
    ]
    best_variant = Counter(variants).most_common(1)[0][0] if variants else None
    best_score = round(sum(scores) / len(scores), 4) if scores else None
    return {"best_variant": best_variant, "best_score": best_score}
