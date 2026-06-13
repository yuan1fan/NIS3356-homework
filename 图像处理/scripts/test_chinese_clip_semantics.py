from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from src.social_ocr.clip_semantics import DEFAULT_CLIP_MODEL, ChineseClipAnalyzer  # noqa: E402
from src.social_ocr.pipeline import IMAGE_SUFFIXES  # noqa: E402
from src.social_ocr.preprocess import read_image  # noqa: E402
from src.social_ocr.video import VIDEO_LIKE_SUFFIXES, extract_video_frames, is_probable_video_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quickly test Chinese-CLIP visual semantics.")
    parser.add_argument("paths", nargs="+", help="Image, video, or directory paths.")
    parser.add_argument("--model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0, gpu:0, etc.")
    parser.add_argument("--limit", type=int, default=12, help="Max files from each directory.")
    parser.add_argument("--video-frames", type=int, default=4, help="Frames sampled from each video.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyzer = ChineseClipAnalyzer(model_name=args.model, device=args.device)
    results: list[dict] = []
    for path in expand_paths([Path(item) for item in args.paths], args.limit):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            image = read_image(path)
            result = analyzer.analyze_image(image)
            result["source_media"] = str(path)
            results.append(result)
            print_result(path, result)
        elif path.suffix.lower() in VIDEO_LIKE_SUFFIXES and is_probable_video_file(path):
            frames = extract_video_frames(path, interval_seconds=2, max_frames=args.video_frames)
            for frame in frames:
                result = analyzer.analyze_image(frame.image)
                result["source_media"] = str(path)
                result["timestamp_seconds"] = frame.timestamp_seconds
                results.append(result)
                print_result(path, result, suffix=f"t={frame.timestamp_seconds:.1f}s")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved JSON: {output_path}")


def expand_paths(paths: list[Path], limit: int) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            files = [
                item
                for item in sorted(path.rglob("*"))
                if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES | VIDEO_LIKE_SUFFIXES
            ]
            expanded.extend(files[:limit])
        else:
            expanded.append(path)
    return expanded


def print_result(path: Path, result: dict, suffix: str = "") -> None:
    visual_type = result.get("visual_type", {})
    tags = result.get("semantic_tags", [])
    tag_text = ", ".join(f"{item['label']}={item['score']:.3f}" for item in tags[:3])
    title = str(path)
    if suffix:
        title += f" [{suffix}]"
    print(f"{title}")
    print(f"  type: {visual_type.get('label')} ({visual_type.get('confidence'):.3f})")
    print(f"  tags: {tag_text}")
    print(f"  summary: {result.get('visual_summary')}")


if __name__ == "__main__":
    main()
