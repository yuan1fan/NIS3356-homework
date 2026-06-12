from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .preprocess import write_image


REGION_COLORS = {
    "header": (255, 170, 0),
    "title": (0, 90, 255),
    "body": (0, 180, 90),
    "side_or_metric": (180, 70, 255),
    "footer": (120, 120, 120),
    "unknown": (255, 255, 255),
}


def draw_text_blocks(image: np.ndarray, blocks: list[dict], output_path: str | Path) -> None:
    canvas = image.copy()
    for block in blocks:
        box = block.get("box") or []
        if len(box) >= 4:
            points = np.array(box, dtype=np.int32)
            color = REGION_COLORS.get(block.get("region_type", "unknown"), REGION_COLORS["unknown"])
            cv2.polylines(canvas, [points], isClosed=True, color=color, thickness=2)
            x = int(min(point[0] for point in box))
            y = int(min(point[1] for point in box))
            label = f"{block.get('confidence', 0.0):.2f}"
            cv2.putText(
                canvas,
                label,
                (x, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    write_image(output_path, canvas)

