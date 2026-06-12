from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from src.social_ocr.pipeline import IMAGE_SUFFIXES, process_media_batch  # noqa: E402
from src.social_ocr.video import VIDEO_LIKE_SUFFIXES, is_probable_video_file  # noqa: E402


VARIANT_SETS = {
    "full": None,
    "fast": {"original", "gray", "clahe", "clahe_sharpen"},
    "minimal": {"original", "clahe"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR over all images and videos from a Weibo crawl output directory."
    )
    parser.add_argument(
        "--crawler-outputs-dir",
        default=str(PROJECT_ROOT / "数据爬取" / "outputs"),
        help="Directory containing timestamped crawler runs.",
    )
    parser.add_argument(
        "--crawl-run-dir",
        default="",
        help="Specific crawler run directory. Defaults to the latest directory containing media/.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="OCR output directory. Defaults to outputs_from_crawl/<crawl-run-name>.",
    )
    parser.add_argument("--platform", default="weibo_hotsearch", help="Platform label in OCR JSON.")
    parser.add_argument("--variant-set", choices=sorted(VARIANT_SETS), default="minimal")
    parser.add_argument("--device", default="cpu", help="Paddle inference device, for example cpu or gpu:0.")
    parser.add_argument("--image-limit", type=int, help="Optional image limit for quick tests.")
    parser.add_argument("--video-limit", type=int, help="Optional video limit for quick tests.")
    parser.add_argument(
        "--frame-regions",
        default="full,bottom",
        help="Comma-separated video frame regions: full,top,center,bottom.",
    )
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=None,
        help="Override video frame interval. Default auto: 2s under 64s, otherwise duration/32.",
    )
    parser.add_argument("--max-video-frames", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true", help="Only print media counts; do not run OCR.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.crawl_run_dir) if args.crawl_run_dir else find_latest_crawl_run(Path(args.crawler_outputs_dir))
    media_dir = run_dir / "media"
    if not media_dir.exists():
        raise FileNotFoundError(f"Cannot find media directory: {media_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else MODULE_ROOT / "outputs_from_crawl" / run_dir.name
    stats = count_media(media_dir)
    print(f"Crawl run: {run_dir}")
    print(f"Media dir: {media_dir}")
    print(
        "Found "
        f"{stats['image_count']} images, "
        f"{stats['video_like_count']} video-like files, "
        f"{stats['probable_video_count']} probable videos."
    )
    if args.dry_run:
        print("Dry run finished; OCR was not executed.")
        return

    frame_regions = tuple(item.strip() for item in args.frame_regions.split(",") if item.strip())
    report = process_media_batch(
        input_dir=media_dir,
        output_dir=output_dir,
        platform=args.platform,
        image_limit=args.image_limit,
        video_limit=args.video_limit,
        variant_names=VARIANT_SETS[args.variant_set],
        device=args.device,
        frame_interval_seconds=args.frame_interval,
        max_video_frames=args.max_video_frames,
        frame_regions=frame_regions,
    )

    write_run_manifest(output_dir, run_dir, media_dir, stats, report)
    print(
        f"Processed {report['image_count']} images and {report['video_count']} videos. "
        f"Skipped videos: {report['skipped_video_count']}."
    )
    print(f"Detailed report: {output_dir / 'reports' / 'media_batch_summary.json'}")
    print(f"LLM report: {output_dir / 'reports' / 'llm_media_batch_summary.json'}")


def find_latest_crawl_run(outputs_dir: Path) -> Path:
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Crawler outputs directory does not exist: {outputs_dir}")
    candidates = [
        path
        for path in outputs_dir.iterdir()
        if path.is_dir() and (path / "media").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No crawler run with media/ found under {outputs_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def count_media(media_dir: Path) -> dict[str, int]:
    files = [path for path in media_dir.rglob("*") if path.is_file()]
    image_count = sum(1 for path in files if path.suffix.lower() in IMAGE_SUFFIXES)
    video_like = [path for path in files if path.suffix.lower() in VIDEO_LIKE_SUFFIXES]
    probable_video_count = sum(1 for path in video_like if is_probable_video_file(path))
    return {
        "image_count": image_count,
        "video_like_count": len(video_like),
        "probable_video_count": probable_video_count,
    }


def write_run_manifest(
    output_dir: Path,
    run_dir: Path,
    media_dir: Path,
    stats: dict[str, int],
    report: dict,
) -> None:
    manifest = {
        "crawl_run_dir": str(run_dir),
        "media_dir": str(media_dir),
        "media_stats": stats,
        "ocr_output_dir": str(output_dir),
        "image_count": report.get("image_count"),
        "video_count": report.get("video_count"),
        "skipped_video_count": report.get("skipped_video_count"),
        "llm_report": str(output_dir / "reports" / "llm_media_batch_summary.json"),
    }
    path = output_dir / "reports" / "crawl_media_ocr_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
