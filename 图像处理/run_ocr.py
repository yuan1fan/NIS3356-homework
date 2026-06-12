from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.social_ocr.pipeline import process_batch, process_image, process_media_batch, process_video


VARIANT_SETS = {
    "full": None,
    "fast": {"original", "gray", "clahe", "clahe_sharpen"},
    "minimal": {"original", "clahe"},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run social media OCR with preprocessing variants and confidence evaluation."
    )
    parser.add_argument("--image", type=str, help="Path to a single image.")
    parser.add_argument("--video", type=str, help="Path to a single video.")
    parser.add_argument("--media-dir", type=str, help="Recursive directory of images and videos.")
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Directory of images.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--platform", type=str, default="weibo", help="Source platform name.")
    parser.add_argument("--limit", type=int, help="Maximum number of images to process in batch mode.")
    parser.add_argument("--image-limit", type=int, help="Maximum number of images in media-dir mode.")
    parser.add_argument("--video-limit", type=int, help="Maximum number of videos in media-dir mode.")
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=None,
        help="Seconds between sampled video frames. Default: auto, 2s for videos under 64s, otherwise duration/32.",
    )
    parser.add_argument("--max-video-frames", type=int, default=32, help="Maximum frames sampled per video.")
    parser.add_argument(
        "--frame-regions",
        default="full,bottom",
        help="Comma-separated video frame regions: full,top,center,bottom.",
    )
    parser.add_argument(
        "--variant-set",
        choices=sorted(VARIANT_SETS),
        default="full",
        help="Preprocessing variants to run. full=8 variants, fast=4 variants, minimal=2 variants.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Paddle inference device, for example cpu or gpu:0. Requires paddlepaddle-gpu for GPU.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable progress output in media-dir mode.")
    args = parser.parse_args()
    variant_names = VARIANT_SETS[args.variant_set]

    frame_regions = tuple(item.strip() for item in args.frame_regions.split(",") if item.strip())

    if args.image:
        result = process_image(
            args.image,
            output_dir=args.output_dir,
            platform=args.platform,
            variant_names=variant_names,
            device=args.device,
        )
        print(f"Detailed JSON: {result['json_path']}")
        print(f"LLM JSON: {result['llm_json_path']}")
        print(f"Visualization: {result['visualization_path']}")
        print(f"Best variant: {result['best_variant']} score={result['best_score']}")
        print(result["for_llm_summary"])
    elif args.video:
        result = process_video(
            args.video,
            output_dir=args.output_dir,
            platform=args.platform,
            frame_interval_seconds=args.frame_interval,
            max_frames=args.max_video_frames,
            frame_regions=frame_regions,
            variant_names=variant_names,
            device=args.device,
        )
        print(f"Detailed JSON: {result['json_path']}")
        print(f"LLM JSON: {result['llm_json_path']}")
        print(f"Sampled frames: {result['frame_sampling']['sampled_frame_count']}")
        print(result["for_llm_summary"])
    elif args.media_dir:
        progress = SimpleProgress(enabled=not args.quiet)
        report = process_media_batch(
            input_dir=Path(args.media_dir),
            output_dir=Path(args.output_dir),
            platform=args.platform,
            image_limit=args.image_limit,
            video_limit=args.video_limit,
            variant_names=variant_names,
            device=args.device,
            frame_interval_seconds=args.frame_interval,
            max_video_frames=args.max_video_frames,
            frame_regions=frame_regions,
            progress_callback=progress,
        )
        progress.finish()
        print(
            f"Processed {report['image_count']} images and "
            f"{report['video_count']} videos. Skipped videos: {report['skipped_video_count']}."
        )
        print(f"Report: {Path(args.output_dir) / 'reports' / 'media_batch_summary.json'}")
        print(f"LLM report: {Path(args.output_dir) / 'reports' / 'llm_media_batch_summary.json'}")
    else:
        results = process_batch(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            platform=args.platform,
            limit=args.limit,
            variant_names=variant_names,
            device=args.device,
        )
        print(f"Processed {len(results)} images.")
        for result in results:
            print(
                f"- {result['image_id']}: {result['best_variant']} "
                f"score={result['best_score']} llm_json={result['llm_json_path']}"
            )


class SimpleProgress:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.count = 0
        self.started_at = time.time()

    def __call__(self, event: dict) -> None:
        if not self.enabled:
            return
        if event.get("event") == "video_start":
            print(f"\nVideo: {Path(str(event.get('path'))).name}")
            return
        if event.get("event") != "advance":
            return
        self.count += 1
        elapsed = time.time() - self.started_at
        if event.get("kind") == "video_frame":
            message = (
                f"{self.count} units | {Path(str(event.get('path'))).name} "
                f"t={event.get('timestamp_seconds')}s {event.get('region')}"
            )
        else:
            message = f"{self.count} units | {Path(str(event.get('path'))).name}"
        print(f"\r{message} | elapsed={elapsed:.0f}s", end="", flush=True)

    def finish(self) -> None:
        if self.enabled:
            print()


if __name__ == "__main__":
    main()
