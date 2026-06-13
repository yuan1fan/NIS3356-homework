from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from src.social_ocr.pipeline import IMAGE_SUFFIXES, process_media_batch  # noqa: E402
from src.social_ocr.video import VIDEO_LIKE_SUFFIXES, build_sampling_plan, get_video_metadata, is_probable_video_file  # noqa: E402


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
    parser.add_argument(
        "--mode",
        choices=("cpu", "gpu"),
        default="cpu",
        help="Inference mode shortcut. gpu maps to gpu:0 unless --device is set.",
    )
    parser.add_argument(
        "--device",
        default="",
        help="Advanced Paddle device override, for example cpu, gpu:0, gpu:1.",
    )
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
    parser.add_argument(
        "--enable-clip",
        action="store_true",
        help="Enable Chinese-CLIP zero-shot visual type and semantic tag classification.",
    )
    parser.add_argument(
        "--clip-model",
        default="OFA-Sys/chinese-clip-vit-base-patch16",
        help="Hugging Face Chinese-CLIP model name.",
    )
    parser.add_argument(
        "--enable-caption",
        action="store_true",
        help="Generate one-sentence image captions for non-text-heavy visual types.",
    )
    parser.add_argument(
        "--caption-model",
        default="Salesforce/blip-image-captioning-base",
        help="Hugging Face image captioning model name.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print media counts; do not run OCR.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.mode, args.device)
    run_dir = Path(args.crawl_run_dir) if args.crawl_run_dir else find_latest_crawl_run(Path(args.crawler_outputs_dir))
    media_dir = run_dir / "media"
    if not media_dir.exists():
        raise FileNotFoundError(f"Cannot find media directory: {media_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else MODULE_ROOT / "outputs_from_crawl" / run_dir.name
    frame_regions = tuple(item.strip() for item in args.frame_regions.split(",") if item.strip())
    stats = count_media(
        media_dir,
        args.frame_interval,
        args.max_video_frames,
        frame_regions,
        args.image_limit,
        args.video_limit,
    )
    print(f"Crawl run: {run_dir}")
    print(f"Media dir: {media_dir}")
    print(f"OCR mode: {args.mode}; device: {device}; variant set: {args.variant_set}")
    print(f"Chinese-CLIP visual semantics: {'enabled' if args.enable_clip else 'disabled'}")
    print(f"Conditional image captioning: {'enabled' if args.enable_caption else 'disabled'}")
    print(
        "Found "
        f"{stats['image_count']} images, "
        f"{stats['video_like_count']} video-like files, "
        f"{stats['probable_video_count']} probable videos. "
        f"Selected {stats['selected_image_count']} images and "
        f"{stats['selected_probable_video_count']} probable videos. "
        f"Estimated OCR units for this run: {stats['estimated_ocr_units']}."
    )
    if args.dry_run:
        print("Dry run finished; OCR was not executed.")
        return

    progress = ProgressPrinter(total=stats["estimated_ocr_units"], enabled=not args.quiet)
    report = process_media_batch(
        input_dir=media_dir,
        output_dir=output_dir,
        platform=args.platform,
        image_limit=args.image_limit,
        video_limit=args.video_limit,
        variant_names=VARIANT_SETS[args.variant_set],
        device=device,
        frame_interval_seconds=args.frame_interval,
        max_video_frames=args.max_video_frames,
        frame_regions=frame_regions,
        enable_clip=args.enable_clip,
        clip_model=args.clip_model,
        enable_caption=args.enable_caption,
        caption_model=args.caption_model,
        progress_callback=progress,
    )
    progress.finish()

    write_run_manifest(output_dir, run_dir, media_dir, stats, report)
    print(
        f"Processed {report['image_count']} images and {report['video_count']} videos. "
        f"Skipped images: {report.get('skipped_image_count', 0)}. "
        f"Skipped videos: {report['skipped_video_count']}."
    )
    print(f"Detailed report: {output_dir / 'reports' / 'media_batch_summary.json'}")
    print(f"LLM report: {output_dir / 'reports' / 'llm_media_batch_summary.json'}")


def resolve_device(mode: str, device_override: str) -> str:
    if device_override:
        return device_override
    return "gpu:0" if mode == "gpu" else "cpu"


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


def count_media(
    media_dir: Path,
    frame_interval_seconds: float | None,
    max_video_frames: int,
    frame_regions: tuple[str, ...],
    image_limit: int | None = None,
    video_limit: int | None = None,
) -> dict[str, int]:
    files = [path for path in media_dir.rglob("*") if path.is_file()]
    image_paths = sorted(path for path in files if path.suffix.lower() in IMAGE_SUFFIXES)
    video_like = sorted(path for path in files if path.suffix.lower() in VIDEO_LIKE_SUFFIXES)
    probable_videos = [path for path in video_like if is_probable_video_file(path)]
    selected_images = image_paths[:image_limit] if image_limit is not None else image_paths
    selected_video_like = video_like[:video_limit] if video_limit is not None else video_like
    selected_probable_videos = [
        path for path in selected_video_like if is_probable_video_file(path)
    ]
    estimated_video_units = sum(
        estimate_video_ocr_units(path, frame_interval_seconds, max_video_frames, frame_regions)
        for path in selected_probable_videos
    )
    return {
        "image_count": len(image_paths),
        "video_like_count": len(video_like),
        "probable_video_count": len(probable_videos),
        "selected_image_count": len(selected_images),
        "selected_video_like_count": len(selected_video_like),
        "selected_probable_video_count": len(selected_probable_videos),
        "estimated_ocr_units": len(selected_images) + estimated_video_units,
    }


def estimate_video_ocr_units(
    path: Path,
    frame_interval_seconds: float | None,
    max_video_frames: int,
    frame_regions: tuple[str, ...],
) -> int:
    try:
        metadata = get_video_metadata(path)
        plan = build_sampling_plan(
            float(metadata.get("duration_seconds", 0.0) or 0.0),
            frame_interval_seconds,
            max_video_frames,
        )
        return max(1, len(plan.timestamps)) * max(1, len(frame_regions))
    except Exception:
        return max(1, max_video_frames) * max(1, len(frame_regions))


class ProgressPrinter:
    def __init__(self, total: int, enabled: bool = True) -> None:
        self.total = max(1, int(total))
        self.enabled = enabled
        self.done = 0
        self.started_at = time.time()
        self.last_message = ""

    def __call__(self, event: dict) -> None:
        if not self.enabled:
            return
        if event.get("event") == "video_start":
            self.last_message = f"video {Path(str(event.get('path'))).name}"
            self._render()
            return
        if event.get("event") != "advance":
            return
        self.done += 1
        path = Path(str(event.get("path", ""))).name
        kind = event.get("kind", "")
        text_chars = event.get("text_chars", 0)
        if kind == "video_frame":
            self.last_message = (
                f"{path} t={event.get('timestamp_seconds')}s "
                f"{event.get('region')} chars={text_chars}"
            )
        else:
            self.last_message = f"{path} chars={text_chars}"
        self._render()

    def finish(self) -> None:
        if not self.enabled:
            return
        self.done = min(max(self.done, 0), self.total)
        self._render(final=True)

    def _render(self, final: bool = False) -> None:
        width = 28
        ratio = min(1.0, self.done / self.total)
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = max(0.1, time.time() - self.started_at)
        speed = self.done / elapsed
        remaining = (self.total - self.done) / speed if speed > 0 else 0.0
        end = "\n" if final else "\r"
        print(
            f"[{bar}] {self.done}/{self.total} {ratio * 100:5.1f}% "
            f"elapsed={elapsed:5.0f}s eta={remaining:5.0f}s {self.last_message[:80]}",
            end=end,
            flush=True,
        )


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
        "skipped_image_count": report.get("skipped_image_count"),
        "video_count": report.get("video_count"),
        "skipped_video_count": report.get("skipped_video_count"),
        "llm_report": str(output_dir / "reports" / "llm_media_batch_summary.json"),
    }
    path = output_dir / "reports" / "crawl_media_ocr_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
