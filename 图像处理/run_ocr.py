from __future__ import annotations

import argparse
from pathlib import Path

from src.social_ocr.pipeline import process_batch, process_image


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
    parser.add_argument("--input-dir", type=str, default="data/raw", help="Directory of images.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--platform", type=str, default="weibo", help="Source platform name.")
    parser.add_argument("--limit", type=int, help="Maximum number of images to process in batch mode.")
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
    args = parser.parse_args()
    variant_names = VARIANT_SETS[args.variant_set]

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


if __name__ == "__main__":
    main()
