"""Run reproducible, dependency-free analysis on the integrated dataset."""

from __future__ import annotations

import argparse
import csv
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


def build_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Run every reproducible analysis section."""
    return {
        "coverage": analyze_coverage(records),
        "hot_content": analyze_hot_content(records),
        "engagement": analyze_engagement(records),
        "multimodal": analyze_multimodal(records),
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

    print(f"Analyzed records: {len(records)}")
    for output_path in outputs.values():
        print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
