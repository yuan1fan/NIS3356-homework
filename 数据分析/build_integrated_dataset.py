"""Build an integrated dataset from crawler, OCR/visual, NLP, and optional ASR data."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = MODULE_DIR / "integrated_outputs"
EXPECTED_MODALITIES = ("text", "nlp", "ocr", "vision", "asr")
MODALITY_WEIGHTS = {
    "text": 0.25,
    "nlp": 0.25,
    "ocr": 0.20,
    "vision": 0.20,
    "asr": 0.10,
}
CROSS_MODAL_MIN_TEXT_LENGTH = 10
CROSS_MODAL_CONSISTENT_THRESHOLD = 0.35
CROSS_MODAL_PARTIAL_THRESHOLD = 0.08
CROSS_MODAL_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+")
VISION_DESCRIPTION_MARKERS = {
    "图片",
    "照片",
    "画面",
    "场景",
    "人物",
    "标签",
    "视觉",
    "媒体",
    "文字",
    "新闻",
    "娱乐",
    "明星",
    "事故",
    "现场",
    "描述",
    "image",
    "photo",
    "person",
    "scene",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a UTF-8 JSONL file containing objects."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected an object in {path} at line {line_number}"
                )
            rows.append(row)
    return rows


def load_json_array(path: Path) -> list[dict[str, Any]]:
    """Load a UTF-8 JSON array containing objects."""
    with path.open("r", encoding="utf-8-sig") as input_file:
        data = json.load(input_file)
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError(f"Expected an array of objects in {path}")
    return data


def normalize_media_path(path_value: Any) -> str | None:
    """Return the normalized path portion beginning with media/."""
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    normalized = path_value.replace("\\", "/").strip()
    marker_index = normalized.lower().rfind("/media/")
    if marker_index >= 0:
        normalized = normalized[marker_index + 1 :]
    elif not normalized.lower().startswith("media/"):
        return None
    return PurePosixPath(normalized).as_posix().lower()


def selected_post(crawler_record: dict[str, Any]) -> dict[str, Any]:
    """Return the selected representative post."""
    return (crawler_record.get("selection") or {}).get("selected_post") or {}


def local_media_paths(post: dict[str, Any], media_type: str) -> list[str]:
    """Collect non-empty local paths for a crawler media type."""
    paths: list[str] = []
    for item in ((post.get("media") or {}).get(media_type) or []):
        if isinstance(item, dict):
            path_value = item.get("local_path")
            if isinstance(path_value, str) and path_value.strip():
                paths.append(path_value.strip())
    return paths


def build_parent_records(
    crawler_rows: list[dict[str, Any]],
    crawl_run_id: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Create one parent record per representative post."""
    parents: list[dict[str, Any]] = []
    media_to_parent: dict[str, int] = {}

    for record_index, crawler_row in enumerate(crawler_rows, start=1):
        topic_data = crawler_row.get("topic") or {}
        post = selected_post(crawler_row)
        raw_text = post.get("text")
        raw_post_text = raw_text if isinstance(raw_text, str) else ""
        image_paths = local_media_paths(post, "images")
        video_paths = local_media_paths(post, "videos")
        audio_paths = local_media_paths(post, "audios")

        parent = {
            "parent_document_id": (
                f"{crawl_run_id}:record:{record_index:04d}"
            ),
            "crawl_run_id": crawl_run_id,
            "record_index": record_index,
            "topic": topic_data.get("word", ""),
            "topic_rank": topic_data.get("rank"),
            "post_id": post.get("post_id", ""),
            "raw_post_text": raw_post_text,
            "ocr_texts": [],
            "visual_summaries": [],
            "asr_texts": [],
            "asr_count": 0,
            "nlp_result": None,
            "merged_text": "",
            "media_info": {
                "image_paths": image_paths,
                "video_paths": video_paths,
                "audio_paths": audio_paths,
                "image_path_count": len(image_paths),
                "video_path_count": len(video_paths),
                "audio_path_count": len(audio_paths),
                "ocr_count": 0,
                "empty_ocr_count": 0,
                "image_ocr_count": 0,
                "video_ocr_count": 0,
                "visual_summary_count": 0,
                "empty_visual_summary_count": 0,
            },
            "metrics": post.get("metrics") or {},
            "quality": {
                "ocr_average_confidence": None,
                "ocr_needs_review_count": 0,
                "ocr_alignment_complete": False,
                "visual_semantics_available": False,
                "nlp_aligned": False,
                "asr_available": False,
            },
            "_ocr_confidences": [],
        }
        parents.append(parent)

        for local_path in image_paths + video_paths + audio_paths:
            normalized = normalize_media_path(local_path)
            if normalized is None:
                continue
            if normalized in media_to_parent:
                raise ValueError(f"Duplicate crawler media path: {local_path}")
            media_to_parent[normalized] = record_index - 1

    return parents, media_to_parent


def attach_image_results(
    parents: list[dict[str, Any]],
    media_to_parent: dict[str, int],
    image_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attach OCR text and visual summaries using source_media."""
    unmatched: list[dict[str, Any]] = []
    aligned_count = 0
    ocr_nonempty_count = 0
    visual_summary_nonempty_count = 0

    for image_row in image_rows:
        relative_path = normalize_media_path(image_row.get("source_media"))
        parent_index = media_to_parent.get(relative_path or "")
        if parent_index is None:
            unmatched.append(
                {
                    "media_id": image_row.get("media_id"),
                    "source_media": image_row.get("source_media"),
                }
            )
            continue

        aligned_count += 1
        parent = parents[parent_index]
        media_info = parent["media_info"]
        media_info["ocr_count"] += 1

        media_type = image_row.get("media_type")
        if media_type == "image":
            media_info["image_ocr_count"] += 1
        elif media_type == "video":
            media_info["video_ocr_count"] += 1

        compact_text = image_row.get("ocr_text_compact")
        if isinstance(compact_text, str) and compact_text.strip():
            parent["ocr_texts"].append(compact_text.strip())
            ocr_nonempty_count += 1
        else:
            media_info["empty_ocr_count"] += 1

        visual_semantics = image_row.get("visual_semantics")
        visual_summary = (
            visual_semantics.get("visual_summary")
            if isinstance(visual_semantics, dict)
            else None
        )
        if isinstance(visual_semantics, dict):
            parent["quality"]["visual_semantics_available"] = True
        if isinstance(visual_summary, str) and visual_summary.strip():
            parent["visual_summaries"].append(visual_summary.strip())
            media_info["visual_summary_count"] += 1
            visual_summary_nonempty_count += 1
        else:
            media_info["empty_visual_summary_count"] += 1

        ocr_quality = image_row.get("ocr_quality") or {}
        confidence = ocr_quality.get("avg_confidence")
        if isinstance(confidence, (int, float)):
            parent["_ocr_confidences"].append(float(confidence))
        if ocr_quality.get("needs_review") is True:
            parent["quality"]["ocr_needs_review_count"] += 1

    for parent in parents:
        confidences = parent.pop("_ocr_confidences")
        if confidences:
            parent["quality"]["ocr_average_confidence"] = round(
                mean(confidences), 4
            )
        expected_count = (
            parent["media_info"]["image_path_count"]
            + parent["media_info"]["video_path_count"]
        )
        parent["quality"]["ocr_alignment_complete"] = (
            parent["media_info"]["ocr_count"] == expected_count
        )

    return {
        "input_count": len(image_rows),
        "aligned_count": aligned_count,
        "failed_count": len(unmatched),
        "ocr_nonempty_count": ocr_nonempty_count,
        "visual_summary_nonempty_count": visual_summary_nonempty_count,
        "unmatched": unmatched,
    }


def attach_nlp_results(
    parents: list[dict[str, Any]],
    nlp_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Align NLP records by order, topic, and raw text preview."""
    unmatched: list[dict[str, Any]] = []
    aligned_count = 0

    for row_index, nlp_row in enumerate(nlp_rows):
        if row_index >= len(parents):
            unmatched.append(
                {"nlp_index": row_index + 1, "reason": "no_parent_record"}
            )
            continue

        parent = parents[row_index]
        topic_matches = nlp_row.get("topic") == parent["topic"]
        preview_matches = (
            nlp_row.get("raw_text_preview") == parent["raw_post_text"][:100]
        )
        if topic_matches and preview_matches:
            parent["nlp_result"] = nlp_row
            parent["quality"]["nlp_aligned"] = True
            aligned_count += 1
        else:
            unmatched.append(
                {
                    "nlp_index": row_index + 1,
                    "expected_topic": parent["topic"],
                    "actual_topic": nlp_row.get("topic"),
                    "topic_matches": topic_matches,
                    "preview_matches": preview_matches,
                }
            )

    for parent in parents[len(nlp_rows) :]:
        unmatched.append(
            {
                "parent_document_id": parent["parent_document_id"],
                "reason": "missing_nlp_result",
            }
        )

    return {
        "input_count": len(nlp_rows),
        "aligned_count": aligned_count,
        "failed_count": len(unmatched),
        "unmatched": unmatched,
    }


def finalize_records(parents: list[dict[str, Any]]) -> None:
    """Merge text sources in order and remove exact duplicates."""
    for parent in parents:
        merged_parts: list[str] = []
        seen: set[str] = set()
        candidates = [
            parent["raw_post_text"],
            *parent["ocr_texts"],
            *parent["visual_summaries"],
            *parent["asr_texts"],
        ]
        for candidate in candidates:
            if not isinstance(candidate, str):
                continue
            text = candidate.strip()
            if text and text not in seen:
                merged_parts.append(text)
                seen.add(text)
        parent["merged_text"] = "\n\n".join(merged_parts)


def nonempty_texts(values: Any) -> list[str]:
    """Return stripped, non-empty strings from a list-like value."""
    if not isinstance(values, list):
        return []
    return [
        value.strip()
        for value in values
        if isinstance(value, str) and value.strip()
    ]


def combine_text_sources(sources: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """Combine named text sources in order while removing exact duplicates."""
    combined_parts: list[str] = []
    included_sources: list[str] = []
    seen: set[str] = set()
    for source_name, source_text in sources:
        source_parts = []
        for part in source_text.split("\n\n"):
            text = part.strip()
            if text and text not in seen:
                source_parts.append(text)
                seen.add(text)
        if source_parts:
            combined_parts.extend(source_parts)
            included_sources.append(source_name)
    return "\n\n".join(combined_parts), included_sources


def text_feature_set(text: str) -> set[str]:
    """Extract simple Chinese characters and lowercase ASCII tokens."""
    return {
        token.lower()
        for token in CROSS_MODAL_TOKEN_RE.findall(text)
        if token.strip()
    }


def jaccard_overlap(first_text: str, second_text: str) -> float:
    """Calculate an explainable set-based text overlap score."""
    first = text_feature_set(first_text)
    second = text_feature_set(second_text)
    union = first | second
    if not union:
        return 0.0
    return round(len(first & second) / len(union), 2)


def build_cross_modal_analysis(
    post_text: str,
    ocr_text: str,
    vision_summary: str,
) -> dict[str, Any]:
    """Build lightweight cross-modal complement and consistency indicators."""
    text_ocr_overlap = jaccard_overlap(post_text, ocr_text)
    text_vision_overlap = jaccard_overlap(post_text, vision_summary)

    post_features = text_feature_set(post_text)
    ocr_features = text_feature_set(ocr_text)
    vision_features = text_feature_set(vision_summary)
    ocr_new_features = ocr_features - post_features
    vision_new_features = vision_features - post_features

    ocr_adds_information = (
        len(ocr_text.strip()) >= CROSS_MODAL_MIN_TEXT_LENGTH
        and text_ocr_overlap < 0.85
        and len(ocr_new_features) >= 3
    )
    vision_lower = vision_summary.lower()
    has_visual_description = any(
        marker in vision_lower for marker in VISION_DESCRIPTION_MARKERS
    )
    vision_adds_information = (
        len(vision_summary.strip()) >= CROSS_MODAL_MIN_TEXT_LENGTH
        and has_visual_description
        and len(vision_new_features) >= 3
    )

    side_scores = []
    if ocr_text.strip():
        side_scores.append(text_ocr_overlap)
    if vision_summary.strip():
        side_scores.append(text_vision_overlap)
    if not side_scores:
        modal_consistency = "unknown"
    elif max(side_scores) >= CROSS_MODAL_CONSISTENT_THRESHOLD:
        modal_consistency = "consistent"
    elif max(side_scores) >= CROSS_MODAL_PARTIAL_THRESHOLD:
        modal_consistency = "partial"
    else:
        modal_consistency = "weak"

    extra_sources = []
    notes = []
    if ocr_adds_information:
        extra_sources.append("ocr")
        notes.append("OCR提供了正文之外的图片文字信息")
    if vision_adds_information:
        extra_sources.append("vision")
        notes.append("视觉语义提供了场景、人物或类别信息")
    if modal_consistency == "consistent":
        notes.append("图像侧信息与微博正文具有较高的粗略重合度")
    elif modal_consistency == "partial":
        notes.append("图像侧信息与正文部分相关，同时包含补充信息")
    elif modal_consistency == "weak":
        notes.append("图像侧信息与正文粗略重合度较低，建议结合原媒体查看")
    else:
        notes.append("缺少图像侧文本信息，暂无法判断一致性")

    return {
        "ocr_adds_information": ocr_adds_information,
        "vision_adds_information": vision_adds_information,
        "text_ocr_overlap_score": text_ocr_overlap,
        "text_vision_overlap_score": text_vision_overlap,
        "modal_consistency": modal_consistency,
        "extra_information_sources": extra_sources,
        "analysis_notes": notes,
    }


def percentile(values: list[float], quantile: float) -> float | None:
    """Return a linearly interpolated percentile for finite numeric values."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return (
        ordered[lower_index]
        + (ordered[upper_index] - ordered[lower_index]) * fraction
    )


def build_batch_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate batch-level values needed by record enrichment."""
    engagement_values = []
    for record in records:
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            continue
        engagement = metrics.get("engagement_score")
        if isinstance(engagement, (int, float)) and not isinstance(
            engagement, bool
        ):
            engagement_values.append(float(engagement))
    return {
        "engagement_p75": percentile(engagement_values, 0.75),
        "engagement_value_count": len(engagement_values),
    }


def enrich_multimodal_fields(
    record: dict[str, Any],
    batch_stats: dict[str, Any] | None = None,
) -> None:
    """Add multimodal fusion metadata without changing existing fields."""
    raw_post_text = (
        record.get("raw_post_text").strip()
        if isinstance(record.get("raw_post_text"), str)
        else ""
    )
    ocr_texts = nonempty_texts(record.get("ocr_texts"))
    vision_texts = nonempty_texts(record.get("visual_summaries"))
    asr_texts = nonempty_texts(record.get("asr_texts"))
    nlp_result = record.get("nlp_result")

    modality_presence = {
        "text": bool(raw_post_text),
        "nlp": isinstance(nlp_result, dict) and bool(nlp_result),
        "ocr": bool(ocr_texts),
        "vision": bool(vision_texts),
        "asr": bool(asr_texts),
    }
    available_modalities = [
        modality
        for modality in EXPECTED_MODALITIES
        if modality_presence[modality]
    ]
    record["available_modalities"] = available_modalities
    record["missing_modalities"] = [
        modality
        for modality in EXPECTED_MODALITIES
        if not modality_presence[modality]
    ]
    record["multimodal_score"] = round(
        sum(MODALITY_WEIGHTS[item] for item in available_modalities),
        2,
    )

    ocr_text = "\n\n".join(ocr_texts)
    vision_summary = "\n\n".join(vision_texts)
    asr_text = "\n\n".join(asr_texts)
    combined_text, included_sources = combine_text_sources(
        [
            ("post_text", raw_post_text),
            ("ocr", ocr_text),
            ("vision", vision_summary),
            ("asr", asr_text),
        ]
    )
    record["fused_text"] = {
        "post_text": raw_post_text,
        "ocr_text": ocr_text,
        "vision_summary": vision_summary,
        "asr_text": asr_text,
        "combined_text": combined_text,
        "sources": included_sources,
    }
    cross_modal_analysis = build_cross_modal_analysis(
        raw_post_text,
        ocr_text,
        vision_summary,
    )
    record["cross_modal_analysis"] = cross_modal_analysis

    metrics = record.get("metrics")
    engagement = (
        metrics.get("engagement_score")
        if isinstance(metrics, dict)
        else None
    )
    engagement_threshold = (
        batch_stats.get("engagement_p75")
        if isinstance(batch_stats, dict)
        else None
    )
    high_interaction = (
        isinstance(engagement, (int, float))
        and not isinstance(engagement, bool)
        and isinstance(engagement_threshold, (int, float))
        and float(engagement) > float(engagement_threshold)
    )

    sentiment_label = ""
    if isinstance(nlp_result, dict):
        sentiment = nlp_result.get("sentiment")
        if isinstance(sentiment, dict):
            label = sentiment.get("label")
            if isinstance(label, str):
                sentiment_label = label.strip().lower()
        elif isinstance(sentiment, str):
            sentiment_label = sentiment.strip().lower()
    has_negative_sentiment = sentiment_label in {
        "negative",
        "负面",
        "消极",
    }
    has_multimodal_evidence = bool(ocr_texts or vision_texts or asr_texts)
    short_text_with_extra_modalities = (
        len(raw_post_text) <= 30 and bool(ocr_texts or vision_texts)
    )

    review_reasons = []
    if high_interaction and has_negative_sentiment:
        review_reasons.append("high_interaction_with_negative_sentiment")
    if short_text_with_extra_modalities:
        review_reasons.append("short_text_with_ocr_or_vision")
    weak_with_attention_signal = (
        cross_modal_analysis["modal_consistency"] == "weak"
        and (high_interaction or has_negative_sentiment)
    )
    if weak_with_attention_signal:
        review_reasons.append("weak_cross_modal_consistency")
    record["safety_indicators"] = {
        "high_interaction": high_interaction,
        "has_negative_sentiment": has_negative_sentiment,
        "has_multimodal_evidence": has_multimodal_evidence,
        "has_cross_modal_extra_info": bool(
            cross_modal_analysis["extra_information_sources"]
        ),
        "short_text_with_extra_modalities": short_text_with_extra_modalities,
        "needs_review": bool(review_reasons),
        "review_reasons": review_reasons,
    }


def enrich_records(parents: list[dict[str, Any]]) -> dict[str, Any]:
    """Enrich all records using shared batch-level statistics."""
    batch_stats = build_batch_stats(parents)
    for parent in parents:
        enrich_multimodal_fields(parent, batch_stats)
    return batch_stats


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write UTF-8 JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(
    path: Path,
    parents: list[dict[str, Any]],
    image_stats: dict[str, Any],
    nlp_stats: dict[str, Any],
    batch_stats: dict[str, Any],
) -> None:
    """Write a concise integration validation report."""
    parent_ids = [parent["parent_document_id"] for parent in parents]
    merged_nonempty = sum(bool(parent["merged_text"]) for parent in parents)
    records_with_visual = sum(
        bool(parent["visual_summaries"]) for parent in parents
    )
    image_count = sum(
        parent["media_info"]["image_ocr_count"] for parent in parents
    )
    video_count = sum(
        parent["media_info"]["video_ocr_count"] for parent in parents
    )
    total_ocr_texts = sum(len(parent["ocr_texts"]) for parent in parents)
    total_visual_summaries = sum(
        len(parent["visual_summaries"]) for parent in parents
    )
    review_count = sum(
        (parent.get("safety_indicators") or {}).get("needs_review") is True
        for parent in parents
    )
    consistency_counts = {
        name: sum(
            (parent.get("cross_modal_analysis") or {}).get(
                "modal_consistency"
            )
            == name
            for parent in parents
        )
        for name in ("consistent", "partial", "weak", "unknown")
    }

    report = f"""# 多模块整合数据构建报告

## 构建结果

- 爬虫主表记录数：{len(parents)}
- 生成父记录数：{len(parent_ids)}
- 重复 `parent_document_id`：{len(parent_ids) - len(set(parent_ids))}
- `merged_text` 非空记录数：{merged_nonempty}
- 含视觉摘要的父记录数：{records_with_visual}

## OCR 与视觉语义

- 媒体结果总数：{image_stats['input_count']}
- 图片记录数：{image_count}
- 视频记录数：{video_count}
- 对齐成功：{image_stats['aligned_count']}
- 对齐失败：{image_stats['failed_count']}
- 非空 OCR 文本：{image_stats['ocr_nonempty_count']}
- 非空视觉摘要：{image_stats['visual_summary_nonempty_count']}
- 整合后的 OCR 文本数：{total_ocr_texts}
- 整合后的视觉摘要数：{total_visual_summaries}

## NLP 与 ASR

- NLP 输入记录数：{nlp_stats['input_count']}
- NLP 对齐成功：{nlp_stats['aligned_count']}
- NLP 对齐失败：{nlp_stats['failed_count']}
- ASR 当前缺失，但所有记录均预留 `asr_texts: []` 和 `asr_count: 0`。
- ASR 缺失不会中断整合流程。

## 多模态融合字段

- 每条记录均新增 `available_modalities`、`missing_modalities`、
  `multimodal_score`、`fused_text`、`safety_indicators` 和
  `cross_modal_analysis`。
- 互动量 75 分位阈值：{batch_stats['engagement_p75']}
- PRE 阶段建议关注记录数：{review_count}
- `needs_review` 仅表示候选关注，不代表最终内容安全判定。
- 跨模态一致性分布：{json.dumps(consistency_counts, ensure_ascii=False)}

## 校验结论

- 父键是否唯一：{'是' if len(parent_ids) == len(set(parent_ids)) else '否'}
- `merged_text` 是否全部非空：{'是' if merged_nonempty == len(parents) else '否'}
- 图像处理结果是否全部对齐：{'是' if image_stats['failed_count'] == 0 else '否'}
- NLP 结果是否全部对齐：{'是' if nlp_stats['failed_count'] == 0 else '否'}
- 视觉摘要是否进入统一数据集：{'是' if total_visual_summaries else '否'}

## 无法对齐的结果

### 图像处理

```json
{json.dumps(image_stats['unmatched'], ensure_ascii=False, indent=2)}
```

### NLP

```json
{json.dumps(nlp_stats['unmatched'], ensure_ascii=False, indent=2)}
```
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse required input paths and output settings."""
    parser = argparse.ArgumentParser(
        description="Build the integrated multimodal dataset"
    )
    parser.add_argument("--crawler-jsonl", type=Path, required=True)
    parser.add_argument("--ocr-json", type=Path, required=True)
    parser.add_argument("--nlp-jsonl", type=Path, required=True)
    parser.add_argument("--crawl-run-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def main() -> None:
    """Build the integrated dataset and validation report."""
    args = parse_args()
    crawler_rows = load_jsonl(args.crawler_jsonl)
    image_rows = load_json_array(args.ocr_json)
    nlp_rows = load_jsonl(args.nlp_jsonl)

    parents, media_to_parent = build_parent_records(
        crawler_rows,
        args.crawl_run_id,
    )
    image_stats = attach_image_results(parents, media_to_parent, image_rows)
    nlp_stats = attach_nlp_results(parents, nlp_rows)
    finalize_records(parents)
    batch_stats = enrich_records(parents)

    output_path = args.output_dir / "integrated_records.jsonl"
    report_path = args.output_dir / "integration_build_report.md"
    write_jsonl(output_path, parents)
    write_report(report_path, parents, image_stats, nlp_stats, batch_stats)

    print(f"Parent records: {len(parents)}")
    print(
        "Image results aligned: "
        f"{image_stats['aligned_count']}/{image_stats['input_count']}"
    )
    print(
        f"NLP aligned: {nlp_stats['aligned_count']}/{nlp_stats['input_count']}"
    )
    print(
        "Visual summaries: "
        f"{image_stats['visual_summary_nonempty_count']}"
    )
    print("ASR available: False")
    print(f"Wrote: {output_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
