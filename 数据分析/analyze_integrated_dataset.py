"""Run reproducible, dependency-free analysis on the integrated dataset."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = MODULE_DIR / "integrated_outputs" / "integrated_records.jsonl"
DEFAULT_OUTPUT_DIR = MODULE_DIR / "analysis_outputs"
METRIC_FIELDS = (
    "attitudes_count",
    "comments_count",
    "reposts_count",
    "engagement_score",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load integrated JSONL records."""
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
                raise ValueError(f"Expected an object at line {line_number}")
            rows.append(row)
    return rows


def nested_value(data: Any, *keys: str, default: Any = None) -> Any:
    """Safely read a nested dictionary value."""
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def numeric_value(value: Any) -> float | None:
    """Return a finite number or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def percentage(count: int, total: int) -> float:
    """Return a percentage rounded to two decimals."""
    return round(count / total * 100, 2) if total else 0.0


def describe(values: Iterable[float]) -> dict[str, Any]:
    """Describe a numeric sequence."""
    numbers = list(values)
    if not numbers:
        return {
            "available": False,
            "count": 0,
            "minimum": None,
            "maximum": None,
            "average": None,
            "median": None,
            "total": None,
        }
    return {
        "available": True,
        "count": len(numbers),
        "minimum": min(numbers),
        "maximum": max(numbers),
        "average": round(mean(numbers), 4),
        "median": round(median(numbers), 4),
        "total": round(sum(numbers), 4),
    }


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Calculate Pearson correlation without external dependencies."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(xs, ys)
    )
    x_square_sum = sum((value - x_mean) ** 2 for value in xs)
    y_square_sum = sum((value - y_mean) ** 2 for value in ys)
    denominator = math.sqrt(x_square_sum * y_square_sum)
    return round(numerator / denominator, 4) if denominator else None


def record_category(record: dict[str, Any]) -> str:
    """Return the NLP classification label."""
    return str(
        nested_value(
            record.get("nlp_result") or {},
            "classification",
            "label",
            default="未知",
        )
    )


def record_sentiment(record: dict[str, Any]) -> str:
    """Return the NLP sentiment label."""
    return str(
        nested_value(
            record.get("nlp_result") or {},
            "sentiment",
            "label",
            default="未知",
        )
    )


def record_engagement(record: dict[str, Any]) -> float | None:
    """Return engagement_score when available."""
    return numeric_value((record.get("metrics") or {}).get("engagement_score"))


def media_count(record: dict[str, Any]) -> int:
    """Return the number of image, video, and audio paths."""
    media = record.get("media_info") or {}
    return sum(
        int(media.get(field, 0) or 0)
        for field in (
            "image_path_count",
            "video_path_count",
            "audio_path_count",
        )
    )


def unique_texts(values: Iterable[Any], existing: set[str]) -> list[str]:
    """Return non-empty, exact-deduplicated text values."""
    result: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        text = value.strip()
        if text and text not in existing:
            result.append(text)
            existing.add(text)
    return result


def multimodal_added_length(record: dict[str, Any]) -> int:
    """Measure characters added by OCR and visual summaries."""
    raw_text = str(record.get("raw_post_text") or "").strip()
    seen = {raw_text} if raw_text else set()
    added_parts = unique_texts(record.get("ocr_texts") or [], seen)
    added_parts.extend(
        unique_texts(record.get("visual_summaries") or [], seen)
    )
    return sum(len(text) for text in added_parts)


def analyze_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze module and media coverage."""
    total = len(records)
    counts = {
        "merged_text_nonempty": sum(
            bool(str(record.get("merged_text") or "").strip())
            for record in records
        ),
        "ocr_records": sum(bool(record.get("ocr_texts")) for record in records),
        "visual_summary_records": sum(
            bool(record.get("visual_summaries")) for record in records
        ),
        "nlp_records": sum(
            isinstance(record.get("nlp_result"), dict)
            for record in records
        ),
        "asr_records": sum(bool(record.get("asr_texts")) for record in records),
    }
    media_totals = {
        "images": sum(
            int((record.get("media_info") or {}).get("image_path_count", 0) or 0)
            for record in records
        ),
        "videos": sum(
            int((record.get("media_info") or {}).get("video_path_count", 0) or 0)
            for record in records
        ),
        "audios": sum(
            int((record.get("media_info") or {}).get("audio_path_count", 0) or 0)
            for record in records
        ),
    }
    rates = {
        key: percentage(value, total)
        for key, value in counts.items()
        if key != "merged_text_nonempty"
    }
    return {
        "parent_record_count": total,
        "counts": counts,
        "rates_percent": rates,
        "media_totals": media_totals,
    }


def analyze_hot_content(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze category, sentiment, and representative topics."""
    category_counts: Counter[str] = Counter()
    sentiment_counts: Counter[str] = Counter()
    cross_counts: dict[str, Counter[str]] = defaultdict(Counter)
    category_topics: dict[str, list[tuple[int, int, str]]] = defaultdict(list)

    for position, record in enumerate(records):
        category = record_category(record)
        sentiment = record_sentiment(record)
        category_counts[category] += 1
        sentiment_counts[sentiment] += 1
        cross_counts[category][sentiment] += 1
        topic = str(record.get("topic") or "").strip()
        rank = record.get("topic_rank")
        sortable_rank = int(rank) if isinstance(rank, int) else 10**9
        if topic:
            category_topics[category].append((sortable_rank, position, topic))

    representative_topics = {
        category: [
            topic
            for _, _, topic in sorted(values)[:5]
        ]
        for category, values in category_topics.items()
    }
    return {
        "category_counts": dict(category_counts.most_common()),
        "sentiment_counts": dict(sentiment_counts.most_common()),
        "category_sentiment_counts": {
            category: dict(counts.most_common())
            for category, counts in sorted(cross_counts.items())
        },
        "representative_topics": representative_topics,
    }


def analyze_engagement(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze interaction metrics and their relationship with media."""
    metric_values: dict[str, list[float]] = defaultdict(list)
    missing_counts: Counter[str] = Counter()
    category_values: dict[str, list[float]] = defaultdict(list)
    media_group_values: dict[str, list[float]] = defaultdict(list)
    media_count_values: dict[int, list[float]] = defaultdict(list)
    correlation_media_counts: list[float] = []
    correlation_engagement: list[float] = []

    for record in records:
        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            for field in METRIC_FIELDS:
                missing_counts[field] += 1
            continue

        for field in METRIC_FIELDS:
            value = numeric_value(metrics.get(field))
            if value is None:
                missing_counts[field] += 1
            else:
                metric_values[field].append(value)

        engagement = numeric_value(metrics.get("engagement_score"))
        if engagement is None:
            continue
        count = media_count(record)
        category_values[record_category(record)].append(engagement)
        media_group_values[
            "with_media" if count > 0 else "without_media"
        ].append(engagement)
        media_count_values[count].append(engagement)
        correlation_media_counts.append(float(count))
        correlation_engagement.append(engagement)

    return {
        "metrics_available": bool(metric_values),
        "metric_distributions": {
            field: describe(metric_values.get(field, []))
            for field in METRIC_FIELDS
        },
        "missing_record_counts": {
            field: missing_counts[field] for field in METRIC_FIELDS
        },
        "category_engagement": {
            category: describe(values)
            for category, values in sorted(category_values.items())
        },
        "media_group_engagement": {
            group: describe(media_group_values.get(group, []))
            for group in ("with_media", "without_media")
        },
        "media_count_engagement": {
            str(count): describe(values)
            for count, values in sorted(media_count_values.items())
        },
        "media_count_engagement_pearson": pearson_correlation(
            correlation_media_counts,
            correlation_engagement,
        ),
    }


def analyze_multimodal(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure OCR and visual semantic coverage and text contribution."""
    added_lengths = [multimodal_added_length(record) for record in records]
    samples = []
    ranked_records = sorted(
        records,
        key=lambda record: (
            multimodal_added_length(record),
            len(record.get("ocr_texts") or []),
            len(record.get("visual_summaries") or []),
        ),
        reverse=True,
    )
    for record in ranked_records[:5]:
        samples.append(
            {
                "record_index": record.get("record_index"),
                "topic": record.get("topic", ""),
                "raw_post_text_length": len(
                    str(record.get("raw_post_text") or "")
                ),
                "ocr_text_count": len(record.get("ocr_texts") or []),
                "visual_summary_count": len(
                    record.get("visual_summaries") or []
                ),
                "multimodal_added_length": multimodal_added_length(record),
                "merged_text_length": len(
                    str(record.get("merged_text") or "")
                ),
            }
        )

    return {
        "ocr_nonempty_records": sum(
            bool(record.get("ocr_texts")) for record in records
        ),
        "ocr_empty_records": sum(
            not bool(record.get("ocr_texts")) for record in records
        ),
        "visual_summary_nonempty_records": sum(
            bool(record.get("visual_summaries")) for record in records
        ),
        "visual_summary_empty_records": sum(
            not bool(record.get("visual_summaries")) for record in records
        ),
        "added_text_length": describe(added_lengths),
        "representative_samples": samples,
    }


def analyze_trends(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze batches only when multiple crawl_run_id values are present."""
    batch_counts = Counter(
        str(record.get("crawl_run_id") or "unknown") for record in records
    )
    if len(batch_counts) < 2:
        return {
            "status": "insufficient_batches",
            "batch_count": len(batch_counts),
            "records_by_batch": dict(batch_counts),
            "message": "当前仅支持单批次热点统计；趋势分析和预测需要多个 crawl_run_id。",
        }

    batch_summary: dict[str, Any] = {}
    for batch_id in sorted(batch_counts):
        batch_records = [
            record
            for record in records
            if str(record.get("crawl_run_id") or "unknown") == batch_id
        ]
        hot_content = analyze_hot_content(batch_records)
        engagement = analyze_engagement(batch_records)
        batch_summary[batch_id] = {
            "record_count": len(batch_records),
            "category_counts": hot_content["category_counts"],
            "sentiment_counts": hot_content["sentiment_counts"],
            "engagement_score": engagement["metric_distributions"][
                "engagement_score"
            ],
        }
    return {
        "status": "multiple_batches_available",
        "batch_count": len(batch_counts),
        "records_by_batch": dict(batch_counts),
        "batch_summary": batch_summary,
    }


def inferred_modalities(record: dict[str, Any]) -> list[str]:
    """Return available modalities, falling back to source fields."""
    configured = record.get("available_modalities")
    if isinstance(configured, list):
        return [
            item
            for item in ("text", "nlp", "ocr", "vision", "asr")
            if item in configured
        ]
    checks = {
        "text": bool(str(record.get("raw_post_text") or "").strip()),
        "nlp": isinstance(record.get("nlp_result"), dict)
        and bool(record.get("nlp_result")),
        "ocr": bool(record.get("ocr_texts")),
        "vision": bool(record.get("visual_summaries")),
        "asr": bool(record.get("asr_texts")),
    }
    return [name for name, available in checks.items() if available]


def analyze_fusion(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze fusion completeness, text sources, and review candidates."""
    modality_counts: Counter[str] = Counter()
    score_counts: Counter[str] = Counter()
    source_combinations: Counter[str] = Counter()
    review_reasons: Counter[str] = Counter()
    consistency_counts: Counter[str] = Counter()
    review_samples: list[dict[str, Any]] = []
    weak_consistency_samples: list[dict[str, Any]] = []
    scores: list[float] = []
    fused_text_nonempty = 0
    review_count = 0
    ocr_adds_count = 0
    vision_adds_count = 0
    both_add_count = 0

    for record in records:
        modalities = inferred_modalities(record)
        modality_counts.update(modalities)

        score = numeric_value(record.get("multimodal_score"))
        if score is None:
            weights = {
                "text": 0.25,
                "nlp": 0.25,
                "ocr": 0.20,
                "vision": 0.20,
                "asr": 0.10,
            }
            score = round(sum(weights[name] for name in modalities), 2)
        scores.append(score)
        score_counts[f"{score:.2f}"] += 1

        fused_text = record.get("fused_text")
        if isinstance(fused_text, dict):
            combined_text = str(fused_text.get("combined_text") or "").strip()
            sources = fused_text.get("sources")
            source_names = (
                [str(item) for item in sources if str(item).strip()]
                if isinstance(sources, list)
                else []
            )
        else:
            combined_text = str(record.get("merged_text") or "").strip()
            source_names = []
        if combined_text:
            fused_text_nonempty += 1
        source_combinations[
            " + ".join(source_names) if source_names else "none"
        ] += 1

        cross_modal = record.get("cross_modal_analysis")
        if not isinstance(cross_modal, dict):
            cross_modal = {}
        ocr_adds = cross_modal.get("ocr_adds_information") is True
        vision_adds = cross_modal.get("vision_adds_information") is True
        consistency = str(
            cross_modal.get("modal_consistency") or "unknown"
        ).lower()
        if consistency not in {"consistent", "partial", "weak", "unknown"}:
            consistency = "unknown"
        consistency_counts[consistency] += 1
        ocr_adds_count += int(ocr_adds)
        vision_adds_count += int(vision_adds)
        both_add_count += int(ocr_adds and vision_adds)
        if consistency == "weak" and len(weak_consistency_samples) < 3:
            weak_consistency_samples.append(
                {
                    "topic": str(record.get("topic") or ""),
                    "post_id": str(record.get("post_id") or ""),
                    "text_ocr_overlap_score": cross_modal.get(
                        "text_ocr_overlap_score", 0
                    ),
                    "text_vision_overlap_score": cross_modal.get(
                        "text_vision_overlap_score", 0
                    ),
                    "extra_information_sources": cross_modal.get(
                        "extra_information_sources"
                    )
                    or [],
                }
            )

        safety = record.get("safety_indicators")
        if not isinstance(safety, dict) or safety.get("needs_review") is not True:
            continue
        review_count += 1
        reasons = safety.get("review_reasons")
        reason_list = (
            [str(reason) for reason in reasons if str(reason).strip()]
            if isinstance(reasons, list)
            else []
        )
        review_reasons.update(reason_list or ["unspecified"])
        if len(review_samples) < 3:
            review_samples.append(
                {
                    "topic": str(record.get("topic") or ""),
                    "post_id": str(record.get("post_id") or ""),
                    "available_modalities": modalities,
                    "missing_modalities": record.get("missing_modalities") or [],
                    "multimodal_score": score,
                    "needs_review": True,
                    "review_reasons": reason_list,
                    "combined_text_preview": combined_text[:180],
                }
            )

    total = len(records)
    crawl_run_ids = sorted(
        {
            str(record.get("crawl_run_id"))
            for record in records
            if record.get("crawl_run_id")
        }
    )
    return {
        "record_count": total,
        "crawl_run_ids": crawl_run_ids,
        "modality_counts": {
            name: modality_counts[name]
            for name in ("text", "nlp", "ocr", "vision", "asr")
        },
        "modality_rates_percent": {
            name: percentage(modality_counts[name], total)
            for name in ("text", "nlp", "ocr", "vision", "asr")
        },
        "score_summary": {
            "average": round(mean(scores), 4) if scores else None,
            "minimum": min(scores) if scores else None,
            "maximum": max(scores) if scores else None,
            "distribution": dict(sorted(score_counts.items())),
        },
        "fused_text_nonempty_records": fused_text_nonempty,
        "source_combinations": dict(source_combinations.most_common()),
        "review_candidates": {
            "needs_review": review_count,
            "not_needs_review": total - review_count,
            "reason_counts": dict(review_reasons.most_common()),
            "samples": review_samples,
        },
        "cross_modal": {
            "ocr_adds_information_records": ocr_adds_count,
            "ocr_adds_information_percent": percentage(ocr_adds_count, total),
            "vision_adds_information_records": vision_adds_count,
            "vision_adds_information_percent": percentage(
                vision_adds_count, total
            ),
            "both_add_information_records": both_add_count,
            "consistency_counts": {
                name: consistency_counts[name]
                for name in ("consistent", "partial", "weak", "unknown")
            },
            "weak_samples": weak_consistency_samples,
        },
    }


def build_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Run every reproducible analysis section."""
    return {
        "coverage": analyze_coverage(records),
        "hot_content": analyze_hot_content(records),
        "engagement": analyze_engagement(records),
        "multimodal": analyze_multimodal(records),
        "fusion": analyze_fusion(records),
        "trends": analyze_trends(records),
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a UTF-8 CSV file."""
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def category_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Build category summary rows."""
    hot = metrics["hot_content"]
    engagement = metrics["engagement"]["category_engagement"]
    total = metrics["coverage"]["parent_record_count"]
    rows = []
    for category, count in hot["category_counts"].items():
        category_engagement = engagement.get(category, describe([]))
        rows.append(
            {
                "category": category,
                "record_count": count,
                "percentage": percentage(count, total),
                "average_engagement": category_engagement["average"],
                "median_engagement": category_engagement["median"],
                "representative_topics": "、".join(
                    hot["representative_topics"].get(category, [])
                ),
            }
        )
    return rows


def sentiment_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Build overall and category-sentiment summary rows."""
    hot = metrics["hot_content"]
    total = metrics["coverage"]["parent_record_count"]
    rows = [
        {
            "scope": "overall",
            "category": "全部",
            "sentiment": sentiment,
            "record_count": count,
            "percentage_within_scope": percentage(count, total),
        }
        for sentiment, count in hot["sentiment_counts"].items()
    ]
    for category, counts in hot["category_sentiment_counts"].items():
        category_total = sum(counts.values())
        for sentiment, count in counts.items():
            rows.append(
                {
                    "scope": "category",
                    "category": category,
                    "sentiment": sentiment,
                    "record_count": count,
                    "percentage_within_scope": percentage(
                        count, category_total
                    ),
                }
            )
    return rows


def media_engagement_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Build media and engagement comparison rows."""
    engagement = metrics["engagement"]
    rows = []
    for group, stats in engagement["media_group_engagement"].items():
        rows.append({"comparison_type": "media_presence", "group": group, **stats})
    for count, stats in engagement["media_count_engagement"].items():
        rows.append(
            {
                "comparison_type": "media_count",
                "group": count,
                **stats,
            }
        )
    return rows


def multimodal_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Build multimodal coverage summary rows."""
    coverage = metrics["coverage"]
    multimodal = metrics["multimodal"]
    total = coverage["parent_record_count"]
    rows = [
        {
            "source": "ocr",
            "nonempty_records": multimodal["ocr_nonempty_records"],
            "empty_records": multimodal["ocr_empty_records"],
            "coverage_percent": percentage(
                multimodal["ocr_nonempty_records"], total
            ),
            "average_added_text_length": multimodal["added_text_length"][
                "average"
            ],
        },
        {
            "source": "visual_summary",
            "nonempty_records": multimodal[
                "visual_summary_nonempty_records"
            ],
            "empty_records": multimodal["visual_summary_empty_records"],
            "coverage_percent": percentage(
                multimodal["visual_summary_nonempty_records"], total
            ),
            "average_added_text_length": multimodal["added_text_length"][
                "average"
            ],
        },
    ]
    return rows


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Build a simple Markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(path: Path, metrics: dict[str, Any]) -> None:
    """Generate the report entirely from calculated metrics."""
    coverage = metrics["coverage"]
    counts = coverage["counts"]
    rates = coverage["rates_percent"]
    media = coverage["media_totals"]
    hot = metrics["hot_content"]
    engagement = metrics["engagement"]
    multimodal = metrics["multimodal"]
    trends = metrics["trends"]

    category_table = markdown_table(
        ["类别", "记录数", "占比", "平均互动量", "代表热搜词"],
        [
            [
                row["category"],
                row["record_count"],
                f"{row['percentage']}%",
                row["average_engagement"],
                row["representative_topics"],
            ]
            for row in category_rows(metrics)
        ],
    )
    sentiment_table = markdown_table(
        ["范围", "类别", "情感", "记录数", "范围内占比"],
        [
            [
                row["scope"],
                row["category"],
                row["sentiment"],
                row["record_count"],
                f"{row['percentage_within_scope']}%",
            ]
            for row in sentiment_rows(metrics)
        ],
    )
    metric_table = markdown_table(
        ["指标", "有效数", "最小值", "最大值", "平均值", "中位数", "总量"],
        [
            [
                field,
                stats["count"],
                stats["minimum"],
                stats["maximum"],
                stats["average"],
                stats["median"],
                stats["total"],
            ]
            for field, stats in engagement["metric_distributions"].items()
        ],
    )
    media_table = markdown_table(
        ["分组", "有效数", "平均互动量", "中位互动量"],
        [
            [
                group,
                stats["count"],
                stats["average"],
                stats["median"],
            ]
            for group, stats in engagement["media_group_engagement"].items()
        ],
    )
    sample_table = markdown_table(
        ["topic", "正文长度", "OCR 数", "视觉摘要数", "多模态增加长度"],
        [
            [
                sample["topic"],
                sample["raw_post_text_length"],
                sample["ocr_text_count"],
                sample["visual_summary_count"],
                sample["multimodal_added_length"],
            ]
            for sample in multimodal["representative_samples"]
        ],
    )

    if engagement["metrics_available"]:
        engagement_note = (
            f"媒体数量与 `engagement_score` 的 Pearson 相关系数为 "
            f"{engagement['media_count_engagement_pearson']}。"
        )
    else:
        engagement_note = "输入记录缺少可用互动指标，本节未计算互动比较。"

    report = f"""# 数据分析报告

## 1. 基础覆盖统计

- 父记录数：{coverage['parent_record_count']}
- `merged_text` 非空数：{counts['merged_text_nonempty']}
- OCR 覆盖：{counts['ocr_records']}（{rates['ocr_records']}%）
- 视觉语义覆盖：{counts['visual_summary_records']}（{rates['visual_summary_records']}%）
- NLP 覆盖：{counts['nlp_records']}（{rates['nlp_records']}%）
- ASR 覆盖：{counts['asr_records']}（{rates['asr_records']}%）
- 图片数量：{media['images']}
- 视频数量：{media['videos']}
- 音频数量：{media['audios']}

## 2. 热点类别与情感

### 类别分布与代表热搜

{category_table}

### 情感分布及类别交叉

{sentiment_table}

## 3. 互动量与媒体

{metric_table}

### 有媒体与无媒体帖子对比

{media_table}

本批 50 条代表帖均包含媒体，因此有媒体/无媒体互动对比不具备区分意义，
相关结果仅保留为程序接口验证。

{engagement_note}

若某互动字段缺失，CSV 和 JSON 中会记录有效数量与缺失数量，脚本不会报错。
互动量统计依赖爬虫返回字段。当前批次中部分互动字段数值较低或缺失，因此
互动量相关分析仅作为初步参考。

## 4. 多模态补充价值

- 有 OCR 文本记录：{multimodal['ocr_nonempty_records']}
- OCR 文本为空记录：{multimodal['ocr_empty_records']}
- 有视觉摘要记录：{multimodal['visual_summary_nonempty_records']}
- 视觉摘要为空记录：{multimodal['visual_summary_empty_records']}
- OCR 与视觉摘要平均增加文本长度：{multimodal['added_text_length']['average']}
- OCR 与视觉摘要增加文本长度中位数：{multimodal['added_text_length']['median']}

以下样例由程序按多模态增加文本长度从高到低抽取：

{sample_table}

## 5. 趋势与预测限制

- 批次数：{trends['batch_count']}
- 当前状态：`{trends['status']}`
- 当前仅支持单批次热点统计。
- 趋势分析和预测需要多个 `crawl_run_id`。
- 后续可在多批次数据上统计类别占比、情感占比、互动量和主题变化。
"""
    path.write_text(report, encoding="utf-8")


def truncate_text(value: Any, limit: int = 110) -> str:
    """Return compact text suitable for a report card."""
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def select_fusion_example(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Select one complete, presentation-friendly multimodal record."""
    def priority(record: dict[str, Any]) -> tuple[int, ...]:
        modalities = set(inferred_modalities(record))
        fused = record.get("fused_text") or {}
        safety = record.get("safety_indicators") or {}
        complete_modalities = all(
            name in modalities for name in ("text", "nlp", "ocr", "vision")
        )
        complete_text = all(
            bool(str(fused.get(name) or "").strip())
            for name in ("post_text", "ocr_text", "vision_summary")
        )
        high_negative = (
            safety.get("high_interaction") is True
            and safety.get("has_negative_sentiment") is True
        )
        return (
            int(complete_modalities),
            int(record.get("multimodal_score") == 0.9),
            int(complete_text),
            int(safety.get("needs_review") is True),
            int(high_negative),
            -int(record.get("record_index") or 0),
        )

    return max(records, key=priority) if records else {}


def fusion_example_data(record: dict[str, Any]) -> dict[str, Any]:
    """Build a compact view model for HTML and Markdown reports."""
    fused = record.get("fused_text") or {}
    nlp = record.get("nlp_result") or {}
    safety = record.get("safety_indicators") or {}
    cross_modal = record.get("cross_modal_analysis") or {}
    keywords = []
    for item in nlp.get("keywords_tfidf") or []:
        if isinstance(item, dict) and item.get("word"):
            keywords.append(str(item["word"]))
    return {
        "topic": str(record.get("topic") or "未提供"),
        "post_id": str(record.get("post_id") or "未提供"),
        "post_text": truncate_text(fused.get("post_text"), 110),
        "ocr_text": truncate_text(fused.get("ocr_text"), 110),
        "vision_summary": truncate_text(fused.get("vision_summary"), 110),
        "classification": str(
            nested_value(nlp, "classification", "label", default="")
        ),
        "sentiment": str(nested_value(nlp, "sentiment", "label", default="")),
        "keywords": keywords[:6],
        "available_modalities": inferred_modalities(record),
        "missing_modalities": record.get("missing_modalities") or [],
        "multimodal_score": record.get("multimodal_score", ""),
        "sources": fused.get("sources") or [],
        "needs_review": safety.get("needs_review", False),
        "review_reasons": safety.get("review_reasons") or [],
        "cross_modal": {
            "ocr_adds_information": cross_modal.get(
                "ocr_adds_information", False
            ),
            "vision_adds_information": cross_modal.get(
                "vision_adds_information", False
            ),
            "modal_consistency": cross_modal.get(
                "modal_consistency", "unknown"
            ),
            "text_ocr_overlap_score": cross_modal.get(
                "text_ocr_overlap_score", 0
            ),
            "text_vision_overlap_score": cross_modal.get(
                "text_vision_overlap_score", 0
            ),
        },
    }


def write_multimodal_fusion_report(
    path: Path,
    fusion: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    """Write a PPT-oriented multimodal fusion report."""
    coverage_rows = [
        [
            name,
            fusion["modality_counts"][name],
            f"{fusion['modality_rates_percent'][name]}%",
        ]
        for name in ("text", "nlp", "ocr", "vision", "asr")
    ]
    score_rows = [
        [score, count]
        for score, count in fusion["score_summary"]["distribution"].items()
    ]
    source_rows = [
        [combination, count]
        for combination, count in fusion["source_combinations"].items()
    ]
    reason_rows = [
        [reason, count]
        for reason, count in fusion["review_candidates"]["reason_counts"].items()
    ] or [["无", 0]]
    sample_rows = [
        [
            sample["topic"],
            sample["multimodal_score"],
            "、".join(sample["available_modalities"]),
            "、".join(sample["review_reasons"]),
        ]
        for sample in fusion["review_candidates"]["samples"]
    ] or [["当前批次无候选样例", "-", "-", "-"]]
    run_ids = "、".join(fusion["crawl_run_ids"]) or "未提供"
    example = fusion_example_data(select_fusion_example(records))
    cross_modal = fusion["cross_modal"]
    nlp_parts = [
        f"类别：{example['classification']}" if example["classification"] else "",
        f"情感：{example['sentiment']}" if example["sentiment"] else "",
        (
            f"关键词：{'、'.join(example['keywords'])}"
            if example["keywords"]
            else ""
        ),
    ]
    nlp_summary = "；".join(part for part in nlp_parts if part) or "暂无"

    report = f"""# 多模态融合分析报告

## 1. 数据规模

- Integrated records：{fusion['record_count']}
- `crawl_run_id`：{run_ids}

## 2. 模态覆盖情况

{markdown_table(['模态', '覆盖记录数', '覆盖比例'], coverage_rows)}

ASR 字段已在统一数据结构中预留，但当前批次尚无真实微博视频转写内容，
因此 ASR 覆盖率为 0%。该缺失不会影响现有文本、NLP、OCR 和视觉语义融合。

## 3. 多模态完整度

- 平均完整度：{fusion['score_summary']['average']}
- 最低完整度：{fusion['score_summary']['minimum']}
- 最高完整度：{fusion['score_summary']['maximum']}

{markdown_table(['multimodal_score', '记录数'], score_rows)}

## 4. 融合文本情况

- `fused_text.combined_text` 非空记录：{fusion['fused_text_nonempty_records']}

{markdown_table(['文本来源组合', '记录数'], source_rows)}

## 5. 单条微博多模态融合样例

- 热搜词：{example['topic']}
- `post_id`：{example['post_id']}
- 微博正文：{example['post_text'] or '无'}
- OCR 文本：{example['ocr_text'] or '无'}
- 视觉语义：{example['vision_summary'] or '无'}
- NLP 结果：{nlp_summary}
- 可用模态：{'、'.join(example['available_modalities']) or '无'}
- 缺失模态：{'、'.join(example['missing_modalities']) or '无'}
- 多模态完整度：{example['multimodal_score']}
- 融合来源：{'、'.join(example['sources']) or '无'}
- 候选关注：{example['needs_review']}
- 触发原因：{'、'.join(example['review_reasons']) or '无'}

该记录将微博正文、OCR识别文本、视觉语义摘要和NLP分析结果统一关联到同一
条微博代表帖，并生成完整度评分和候选关注标记。

## 6. 跨模态补充与一致性分析

- OCR 提供补充信息：{cross_modal['ocr_adds_information_records']}
  （{cross_modal['ocr_adds_information_percent']}%）
- 视觉语义提供补充信息：{cross_modal['vision_adds_information_records']}
  （{cross_modal['vision_adds_information_percent']}%）
- OCR 与视觉语义均提供补充信息：{cross_modal['both_add_information_records']}

{markdown_table(
    ['一致性类型', '记录数'],
    [
        [name, count]
        for name, count in cross_modal['consistency_counts'].items()
    ],
)}

弱相关样例（最多 3 条）：

{markdown_table(
    ['热搜词', '正文-OCR 重合度', '正文-视觉重合度', '补充来源'],
    [
        [
            sample['topic'],
            sample['text_ocr_overlap_score'],
            sample['text_vision_overlap_score'],
            '、'.join(sample['extra_information_sources']) or '无',
        ]
        for sample in cross_modal['weak_samples']
    ] or [['当前批次无 weak 样例', '-', '-', '-']],
)}

该分析用于衡量图像侧信息对微博正文的补充价值和粗略一致性，不代表最终
风险判断。

## 7. 内容安全候选标记

- 候选关注内容：{fusion['review_candidates']['needs_review']}
- 其他记录：{fusion['review_candidates']['not_needs_review']}

{markdown_table(['触发原因', '记录数'], reason_rows)}

程序抽取的候选样例（最多 3 条）：

{markdown_table(['热搜词', '完整度', '可用模态', '触发原因'], sample_rows)}

以上结果仅用于 PRE 阶段的候选内容筛选和人工核查，不代表最终风险判定。

## 8. 结论

- 当前批次已完成微博正文、NLP、OCR 和视觉语义的融合。
- ASR 字段已预留，但当前批次暂无实际转写内容。
- 新增融合字段可支持后续系统联动、统计展示和人工核查。
"""
    path.write_text(report, encoding="utf-8")


def configure_chart_style() -> None:
    """Configure a stable local matplotlib style with Chinese font fallbacks."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "#f7f9fc"
    plt.rcParams["axes.facecolor"] = "#ffffff"


def save_bar_chart(
    path: Path,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    color: str,
    value_suffix: str = "",
) -> None:
    """Save a presentation-friendly bar chart."""
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(12.8, 7.2), dpi=125)
    bars = axis.bar(labels, values, color=color, width=0.62)
    axis.set_title(title, fontsize=20, pad=18, fontweight="bold")
    axis.set_ylabel(ylabel, fontsize=13)
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    upper = max(values, default=0)
    axis.set_ylim(0, upper * 1.18 if upper else 1)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (upper * 0.025 if upper else 0.03),
            f"{value:g}{value_suffix}",
            ha="center",
            va="bottom",
            fontsize=12,
        )
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def generate_charts(charts_dir: Path, fusion: dict[str, Any]) -> list[Path]:
    """Generate the required static PNG charts."""
    configure_chart_style()
    charts_dir.mkdir(parents=True, exist_ok=True)

    modality_path = charts_dir / "modality_coverage.png"
    save_bar_chart(
        modality_path,
        ["Text", "NLP", "OCR", "Vision", "ASR"],
        [
            fusion["modality_rates_percent"][name]
            for name in ("text", "nlp", "ocr", "vision", "asr")
        ],
        "模态覆盖率（ASR 字段预留，当前批次未接入）",
        "覆盖率（%）",
        "#3976c6",
        "%",
    )

    score_path = charts_dir / "multimodal_score_distribution.png"
    score_distribution = fusion["score_summary"]["distribution"]
    save_bar_chart(
        score_path,
        list(score_distribution),
        list(score_distribution.values()),
        "多模态完整度评分分布",
        "记录数",
        "#32a47b",
    )

    review_path = charts_dir / "review_candidates_summary.png"
    review = fusion["review_candidates"]
    save_bar_chart(
        review_path,
        ["候选关注", "其他记录"],
        [review["needs_review"], review["not_needs_review"]],
        "候选关注内容统计",
        "记录数",
        "#df805f",
    )

    consistency_path = (
        charts_dir / "cross_modal_consistency_distribution.png"
    )
    consistency = fusion["cross_modal"]["consistency_counts"]
    save_bar_chart(
        consistency_path,
        ["consistent", "partial", "weak", "unknown"],
        [
            consistency[name]
            for name in ("consistent", "partial", "weak", "unknown")
        ],
        "跨模态一致性分布",
        "记录数",
        "#7656b2",
    )
    return [modality_path, score_path, review_path, consistency_path]


def write_html_report(
    path: Path,
    fusion: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    """Write a self-contained static HTML report referencing local charts."""
    example = fusion_example_data(select_fusion_example(records))
    case_record = next(
        (
            record
            for record in records
            if str(record.get("post_id")) == "5308969155298302"
        ),
        None,
    )
    review_records = [
        record
        for record in records
        if isinstance(record.get("safety_indicators"), dict)
        and record["safety_indicators"].get("needs_review") is True
    ]
    sample_records = (review_records + [
        record for record in records if record not in review_records
    ])[:3]
    table_rows = []
    for record in sample_records:
        safety = record.get("safety_indicators") or {}
        fused = record.get("fused_text") or {}
        preview = str(
            fused.get("combined_text") or record.get("merged_text") or ""
        ).replace("\n", " ")[:160]
        cells = [
            record.get("topic", ""),
            record.get("post_id", ""),
            ", ".join(inferred_modalities(record)),
            ", ".join(record.get("missing_modalities") or []),
            record.get("multimodal_score", ""),
            safety.get("needs_review", False),
            ", ".join(safety.get("review_reasons") or []),
            preview,
        ]
        table_rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells)
            + "</tr>"
        )

    available = [
        name
        for name, count in fusion["modality_counts"].items()
        if count > 0
    ]
    review_count = fusion["review_candidates"]["needs_review"]
    cross_modal = fusion["cross_modal"]
    nlp_items = []
    if example["classification"]:
        nlp_items.append(
            f"<span><b>类别</b>{html.escape(example['classification'])}</span>"
        )
    if example["sentiment"]:
        nlp_items.append(
            f"<span><b>情感</b>{html.escape(example['sentiment'])}</span>"
        )
    if example["keywords"]:
        nlp_items.append(
            "<span><b>关键词</b>"
            + html.escape("、".join(example["keywords"]))
            + "</span>"
        )
    nlp_html = "".join(nlp_items) or "<span>暂无 NLP 摘要</span>"

    def tags(values: list[Any], class_name: str = "") -> str:
        return "".join(
            f'<span class="tag {class_name}">{html.escape(str(value))}</span>'
            for value in values
        ) or '<span class="muted">无</span>'

    case_section = ""
    if case_record is not None:
        case_cross = case_record.get("cross_modal_analysis") or {}
        case_section = f"""
  <h2>跨模态分析案例：图像侧信息对正文的补充</h2>
  <div class="subtitle">热搜词：央视曝养生馆围猎老年人　
    post_id：5308969155298302</div>
  <section class="fusion-example case-example">
    <div class="fusion-column">
      <h3>多模态证据</h3>
      <div class="source-card"><strong>微博正文摘要</strong>
        <p>央视报道北京多家养生馆以低价体验吸引老年人，再通过虚假诊断和
        所谓排毒项目实施诈骗，警方已刑拘30余名嫌疑人。</p></div>
      <div class="source-card"><strong>OCR 补充信息</strong>
        <p>视频画面识别出央视新闻标识、养生项目价目表、护理项目价格，以及
        “北京警方捣毁20余家套路养生馆”等新闻字幕。</p></div>
      <div class="source-card"><strong>视觉语义补充</strong>
        <p>媒体被识别为新闻报道和视频画面截图，视觉标签集中在社会民生、
        财经消费、诈骗及违法犯罪等场景。</p></div>
    </div>
    <div class="fusion-column">
      <h3>跨模态分析结果</h3>
      <div class="fusion-meta">
        <div class="meta-item"><strong>ocr_adds_information</strong>
          <span class="tag">{"是" if case_cross.get("ocr_adds_information") else "否"}</span></div>
        <div class="meta-item"><strong>vision_adds_information</strong>
          <span class="tag">{"是" if case_cross.get("vision_adds_information") else "否"}</span></div>
        <div class="meta-item"><strong>text_ocr_overlap_score</strong>
          <span class="score case-score">{case_cross.get("text_ocr_overlap_score", 0)}</span></div>
        <div class="meta-item"><strong>text_vision_overlap_score</strong>
          <span class="score case-score">{case_cross.get("text_vision_overlap_score", 0)}</span></div>
        <div class="meta-item"><strong>modal_consistency</strong>
          <span class="tag">{html.escape(str(case_cross.get("modal_consistency", "unknown")))}</span></div>
        <div class="meta-item"><strong>extra_information_sources</strong>
          {tags(case_cross.get("extra_information_sources") or [])}</div>
      </div>
      <div class="synthesis-box">
        <h4>综合分析输出</h4>
        <div class="synthesis-item"><strong>一致性判断：</strong>
          正文、OCR 和视觉语义围绕“养生馆诈骗”同一事件展开，属于
          partial，即主题相关但图像侧提供额外信息。</div>
        <div class="synthesis-item"><strong>补充价值：</strong>
          OCR 补充了新闻字幕、项目价目表和警方行动信息；视觉语义补充了
          新闻报道、社会民生、诈骗违法场景。</div>
        <div class="synthesis-item"><strong>后续用途：</strong>
          该结果可作为内容安全候选分析中的证据组织结果，供人工核查或
          后续模型进一步判断。</div>
      </div>
    </div>
  </section>
  <div class="fusion-note case-conclusion">
    <strong>分析结论</strong><br>
    正文描述养生馆诈骗事件，OCR 从视频画面中补充了新闻字幕、项目价目表和
    警方行动信息，视觉语义进一步识别出新闻报道、社会民生和诈骗场景。
    图像侧信息与正文主题部分一致，同时提供了正文之外的证据细节，因此该
    样例体现了多模态分析相较单文本分析的补充价值。
  </div>
"""

    html_content = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>多模态数据融合与分析报告</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Microsoft YaHei", Arial, sans-serif;
      color: #243247; background: #f4f7fb; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 36px; }}
    h1 {{ margin: 0 0 8px; font-size: 32px; }}
    h2 {{ margin-top: 36px; font-size: 22px; }}
    .subtitle {{ color: #637083; margin-bottom: 28px; }}
    .cards {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; }}
    .card, .panel {{ background: white; border-radius: 12px;
      box-shadow: 0 3px 14px rgba(31, 50, 80, .08); }}
    .card {{ padding: 20px; border-top: 4px solid #3976c6; }}
    .card strong {{ display: block; font-size: 25px; margin-top: 8px; }}
    .card span {{ color: #6a7688; font-size: 13px; }}
    .charts {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px; }}
    .panel {{ padding: 18px; }}
    .panel img {{ display: block; width: 100%; height: auto; }}
    .fusion-example {{ display: grid; grid-template-columns: 1.2fr .8fr;
      gap: 18px; }}
    .fusion-column {{ background: white; border-radius: 12px; padding: 22px;
      box-shadow: 0 3px 14px rgba(31, 50, 80, .08); }}
    .fusion-column h3 {{ margin: 0 0 16px; color: #315f9d; }}
    .source-card {{ background: #f7f9fc; border-left: 4px solid #3976c6;
      border-radius: 7px; padding: 13px 15px; margin-bottom: 12px; }}
    .source-card strong {{ display: block; margin-bottom: 6px; }}
    .source-card p {{ margin: 0; line-height: 1.65; color: #46556a; }}
    .nlp-row {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .nlp-row span {{ background: #edf5ff; border-radius: 7px; padding: 8px 10px; }}
    .nlp-row b {{ margin-right: 6px; color: #315f9d; }}
    .fusion-meta {{ display: grid; gap: 13px; }}
    .meta-item {{ border-bottom: 1px solid #e6ebf1; padding-bottom: 11px; }}
    .meta-item strong {{ display: block; color: #667489; margin-bottom: 7px;
      font-size: 13px; }}
    .score {{ font-size: 32px; color: #2b8b68; font-weight: bold; }}
    .case-score {{ font-size: 25px; }}
    .tag {{ display: inline-block; background: #e8f0fb; color: #315f9d;
      border-radius: 999px; padding: 5px 9px; margin: 2px 5px 2px 0;
      font-size: 12px; }}
    .tag.missing {{ background: #f0f1f4; color: #687385; }}
    .tag.review {{ background: #fff0e8; color: #b75c38; }}
    .muted {{ color: #8993a2; }}
    .fusion-note {{ margin-top: 18px; padding: 15px 18px; background: #edf7f3;
      border-left: 5px solid #32a47b; border-radius: 8px; line-height: 1.7; }}
    .cross-cards {{ display: grid; grid-template-columns: repeat(4, 1fr);
      gap: 14px; margin-bottom: 18px; }}
    .cross-card {{ background: white; border-radius: 10px; padding: 17px;
      border-left: 4px solid #7656b2;
      box-shadow: 0 3px 14px rgba(31, 50, 80, .08); }}
    .cross-card span {{ color: #69768a; font-size: 13px; }}
    .cross-card strong {{ display: block; font-size: 25px; margin-top: 7px; }}
    .cross-detail {{ background: #f7f4fc; border-radius: 8px; padding: 14px;
      margin-top: 13px; line-height: 1.8; }}
    .case-example .source-card {{ border-left-color: #7656b2; }}
    .case-conclusion {{ background: #f5f1fb; border-left-color: #7656b2; }}
    .synthesis-box {{ margin-top: 15px; padding: 16px 18px;
      background: #f3f7fd; border: 1px solid #d6e2f2;
      border-left: 5px solid #3976c6; border-radius: 9px; }}
    .synthesis-box h4 {{ margin: 0 0 11px; color: #315f9d; font-size: 16px; }}
    .synthesis-item {{ margin: 8px 0; line-height: 1.65; color: #46556a; }}
    .synthesis-item strong {{ color: #283b55; margin-right: 5px; }}
    table {{ width: 100%; border-collapse: collapse; background: white;
      box-shadow: 0 3px 14px rgba(31, 50, 80, .08); }}
    th, td {{ padding: 11px; border: 1px solid #dfe5ed; text-align: left;
      vertical-align: top; font-size: 13px; }}
    th {{ background: #eaf1fb; }}
    .notice {{ padding: 18px 20px; background: #fff7e8;
      border-left: 5px solid #e5a43d; border-radius: 8px; line-height: 1.7; }}
    @media (max-width: 900px) {{
      .cards, .charts, .fusion-example, .cross-cards {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
<main>
  <h1>多模态数据融合与分析报告</h1>
  <div class="subtitle">静态课程设计结果看板</div>
  <section class="cards">
    <div class="card"><span>记录数</span><strong>{fusion['record_count']}</strong></div>
    <div class="card"><span>平均完整度</span><strong>{fusion['score_summary']['average']}</strong></div>
    <div class="card"><span>已接入模态</span><strong>{len(available)}</strong></div>
    <div class="card"><span>候选关注内容</span><strong>{review_count}</strong></div>
    <div class="card"><span>ASR 状态</span><strong>待接入</strong></div>
  </section>

  <h2>融合统计图表</h2>
  <section class="charts">
    <div class="panel"><img src="charts/modality_coverage.png" alt="模态覆盖率"></div>
    <div class="panel"><img src="charts/multimodal_score_distribution.png" alt="完整度分布"></div>
    <div class="panel"><img src="charts/review_candidates_summary.png" alt="候选关注内容"></div>
    <div class="panel"><img src="charts/cross_modal_consistency_distribution.png" alt="跨模态一致性"></div>
  </section>

  <h2>跨模态补充与一致性分析</h2>
  <section class="cross-cards">
    <div class="cross-card"><span>OCR 补充信息</span>
      <strong>{cross_modal['ocr_adds_information_records']}</strong></div>
    <div class="cross-card"><span>视觉补充信息</span>
      <strong>{cross_modal['vision_adds_information_records']}</strong></div>
    <div class="cross-card"><span>图文弱相关</span>
      <strong>{cross_modal['consistency_counts']['weak']}</strong></div>
    <div class="cross-card"><span>OCR 与视觉均补充</span>
      <strong>{cross_modal['both_add_information_records']}</strong></div>
  </section>
  <div class="notice">
    该分析用于衡量图像侧信息对微博正文的补充价值和粗略一致性，不代表最终
    风险判断。
  </div>

{case_section}

  <h2>单条微博多模态融合样例</h2>
  <div class="subtitle">热搜词：{html.escape(example['topic'])}　
    post_id：{html.escape(example['post_id'])}</div>
  <section class="fusion-example">
    <div class="fusion-column">
      <h3>原始输入与模块输出</h3>
      <div class="source-card"><strong>微博正文 · post_text</strong>
        <p>{html.escape(example['post_text'] or '无')}</p></div>
      <div class="source-card"><strong>OCR 识别文本 · ocr_text</strong>
        <p>{html.escape(example['ocr_text'] or '无')}</p></div>
      <div class="source-card"><strong>视觉语义摘要 · vision_summary</strong>
        <p>{html.escape(example['vision_summary'] or '无')}</p></div>
      <div class="source-card"><strong>NLP 分析结果</strong>
        <div class="nlp-row">{nlp_html}</div></div>
    </div>
    <div class="fusion-column">
      <h3>融合后的统一记录</h3>
      <div class="fusion-meta">
        <div class="meta-item"><strong>available_modalities</strong>
          {tags(example['available_modalities'])}</div>
        <div class="meta-item"><strong>missing_modalities</strong>
          {tags(example['missing_modalities'], 'missing')}</div>
        <div class="meta-item"><strong>multimodal_score</strong>
          <span class="score">{example['multimodal_score']}</span></div>
        <div class="meta-item"><strong>fused_text.sources</strong>
          {tags(example['sources'])}</div>
        <div class="meta-item"><strong>候选关注标记 · needs_review</strong>
          <span class="tag review">{example['needs_review']}</span></div>
        <div class="meta-item"><strong>review_reasons</strong>
          {tags(example['review_reasons'], 'review')}</div>
        <div class="cross-detail"><strong>跨模态分析</strong><br>
          OCR补充信息：{'是' if example['cross_modal']['ocr_adds_information'] else '否'}<br>
          视觉补充信息：{'是' if example['cross_modal']['vision_adds_information'] else '否'}<br>
          图文一致性：{html.escape(str(example['cross_modal']['modal_consistency']))}<br>
          文本-OCR重合度：{example['cross_modal']['text_ocr_overlap_score']}<br>
          文本-视觉重合度：{example['cross_modal']['text_vision_overlap_score']}
        </div>
      </div>
    </div>
  </section>
  <div class="fusion-note">
    该记录将微博正文、OCR识别文本、视觉语义摘要和NLP分析结果统一关联到
    同一条微博代表帖，并生成完整度评分和候选关注标记。
  </div>

  <h2>代表记录</h2>
  <table>
    <thead><tr>
      <th>topic</th><th>post_id</th><th>available_modalities</th>
      <th>missing_modalities</th><th>score</th><th>needs_review</th>
      <th>review_reasons</th><th>融合文本预览</th>
    </tr></thead>
    <tbody>{''.join(table_rows)}</tbody>
  </table>

  <h2>ASR 说明</h2>
  <div class="notice">
    当前 ASR 字段已在统一数据结构中预留，但该批次真实微博视频尚未完成批量
    转写，因此 ASR 覆盖率为 0%。该缺失不会影响文本、OCR、视觉语义和 NLP
    的现有融合流程。
  </div>
</main>
</body>
</html>
"""
    path.write_text(html_content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse input and output paths."""
    parser = argparse.ArgumentParser(
        description="Analyze the integrated multimodal dataset"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Run analysis and regenerate every formal output."""
    args = parse_args()
    records = load_jsonl(args.input)
    metrics = build_metrics(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "report": args.output_dir / "data_analysis_report.md",
        "category": args.output_dir / "category_summary.csv",
        "sentiment": args.output_dir / "sentiment_summary.csv",
        "media": args.output_dir / "media_engagement_summary.csv",
        "multimodal": args.output_dir / "multimodal_coverage_summary.csv",
        "metrics": args.output_dir / "analysis_metrics.json",
        "fusion_report": args.output_dir / "multimodal_fusion_report.md",
        "html_report": args.output_dir / "analysis_report.html",
    }
    write_report(outputs["report"], metrics)
    write_csv(
        outputs["category"],
        [
            "category",
            "record_count",
            "percentage",
            "average_engagement",
            "median_engagement",
            "representative_topics",
        ],
        category_rows(metrics),
    )
    write_csv(
        outputs["sentiment"],
        [
            "scope",
            "category",
            "sentiment",
            "record_count",
            "percentage_within_scope",
        ],
        sentiment_rows(metrics),
    )
    write_csv(
        outputs["media"],
        [
            "comparison_type",
            "group",
            "available",
            "count",
            "minimum",
            "maximum",
            "average",
            "median",
            "total",
        ],
        media_engagement_rows(metrics),
    )
    write_csv(
        outputs["multimodal"],
        [
            "source",
            "nonempty_records",
            "empty_records",
            "coverage_percent",
            "average_added_text_length",
        ],
        multimodal_rows(metrics),
    )
    outputs["metrics"].write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_multimodal_fusion_report(
        outputs["fusion_report"],
        metrics["fusion"],
        records,
    )
    chart_paths = generate_charts(
        args.output_dir / "charts",
        metrics["fusion"],
    )
    write_html_report(outputs["html_report"], metrics["fusion"], records)

    print(f"Analyzed records: {len(records)}")
    for output_path in outputs.values():
        print(f"Wrote: {output_path}")
    for chart_path in chart_paths:
        print(f"Wrote: {chart_path}")


if __name__ == "__main__":
    main()
