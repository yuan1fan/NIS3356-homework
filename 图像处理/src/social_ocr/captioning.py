from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from PIL import Image


DEFAULT_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
CAPTION_TRIGGER_TYPES = (
    "新闻现场照片",
    "普通人物照片",
    "风景或环境照片",
    "表情包或梗图",
    "视频画面截图",
    "其他图片",
)


class ImageCaptioner:
    def __init__(self, model_name: str = DEFAULT_CAPTION_MODEL, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = _resolve_torch_device(device)
        self._torch, self._model, self._processor = _load_caption_model(model_name, self.device)

    def caption_image(self, image: np.ndarray, max_new_tokens: int = 32) -> dict[str, Any]:
        pil_image = _bgr_to_pil(image)
        inputs = self._processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=max_new_tokens)
        caption = self._processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        return {
            "method": "image captioning",
            "model": self.model_name,
            "device": self.device,
            "text": caption,
        }


def should_generate_caption(
    visual_semantics: dict[str, Any] | None,
    trigger_types: tuple[str, ...] = CAPTION_TRIGGER_TYPES,
) -> bool:
    if not visual_semantics:
        return False
    visual_type = visual_semantics.get("visual_type", {})
    label = str(visual_type.get("label") or "")
    return label in set(trigger_types)


def attach_caption_to_visual_semantics(
    visual_semantics: dict[str, Any],
    caption: dict[str, Any],
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = dict(visual_semantics)
    caption_record = dict(caption)
    if source:
        caption_record["source"] = source
    updated["image_caption"] = caption_record
    caption_text = str(caption.get("text") or "").strip()
    if caption_text:
        old_summary = str(updated.get("visual_summary") or "").strip()
        updated["visual_summary"] = f"{old_summary} 一句话视觉描述：{caption_text}".strip()
    return updated


def _bgr_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _resolve_torch_device(device: str) -> str:
    requested = device
    if device.startswith("gpu"):
        suffix = device.split(":", 1)[1] if ":" in device else "0"
        requested = f"cuda:{suffix}"
    if requested.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return requested
        except Exception:  # noqa: BLE001
            pass
        return "cpu"
    return "cpu"


@lru_cache(maxsize=2)
def _load_caption_model(model_name: str, device: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Image captioning dependencies are missing. Install torch and transformers, "
            "or disable captioning."
        ) from exc

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForMultimodalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return torch, model, processor
