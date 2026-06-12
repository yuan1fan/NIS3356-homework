from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .ocr_engine import PaddleOcrEngine
from .postprocess import choose_best_result, score_result
from .preprocess import blur_score, generate_variants, read_image, text_area_ratio, write_image
from .visualize import draw_text_blocks


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
    output_dir = Path(output_dir)
    engine = engine or PaddleOcrEngine(device=device)

    original = read_image(image_path)
    variants = generate_variants(original)
    if variant_names is not None:
        variants = [variant for variant in variants if variant.name in variant_names]
    if not variants:
        raise ValueError("No preprocessing variants selected.")
    image_id = image_path.stem

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
        "image_path": str(image_path),
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
    llm_result = make_llm_result(result)

    json_path = output_dir / "json" / f"{image_id}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    llm_json_path = output_dir / "llm_json" / f"{image_id}.json"
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


def make_llm_result(result: dict[str, Any]) -> dict[str, Any]:
    quality = result.get("quality_assessment", {})
    return {
        "image_id": result.get("image_id"),
        "platform": result.get("platform"),
        "source_image": result.get("image_path"),
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
