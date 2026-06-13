from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image


DEFAULT_CLIP_MODEL = "OFA-Sys/chinese-clip-vit-base-patch16"

VISUAL_TYPE_LABELS = (
    "聊天记录截图",
    "新闻报道截图",
    "微博或社交媒体截图",
    "新闻现场照片",
    "普通人物照片",
    "风景或环境照片",
    "信息图表或数据图",
    "宣传海报或公告图",
    "商品广告图片",
    "表情包或梗图",
    "证件票据截图",
    "视频画面截图",
    "其他图片",
)

SEMANTIC_TAG_LABELS = (
    "诈骗",
    "违法犯罪",
    "突发事故",
    "自然灾害",
    "公共安全",
    "食品安全",
    "医疗健康",
    "教育考试",
    "体育赛事",
    "娱乐明星",
    "财经消费",
    "企业品牌",
    "政策公告",
    "社会民生",
    "争议舆情",
    "谣言辟谣",
    "广告营销",
    "日常生活",
)


class ChineseClipAnalyzer:
    def __init__(
        self,
        model_name: str = DEFAULT_CLIP_MODEL,
        device: str = "cpu",
        visual_type_labels: Iterable[str] = VISUAL_TYPE_LABELS,
        semantic_tag_labels: Iterable[str] = SEMANTIC_TAG_LABELS,
    ) -> None:
        self.model_name = model_name
        self.device = _resolve_torch_device(device)
        self.visual_type_labels = tuple(visual_type_labels)
        self.semantic_tag_labels = tuple(semantic_tag_labels)
        self._torch, self._model, self._processor = _load_chinese_clip(model_name, self.device)
        self._type_prompts = [f"这是一张{label}" for label in self.visual_type_labels]
        self._tag_prompts = [f"这张图片与{label}有关" for label in self.semantic_tag_labels]

    def analyze_image(self, image: np.ndarray, ocr_text: str = "") -> dict[str, Any]:
        pil_image = _bgr_to_pil(image)
        type_scores = self._rank_labels(pil_image, self.visual_type_labels, self._type_prompts)
        tag_scores = self._rank_labels(pil_image, self.semantic_tag_labels, self._tag_prompts)
        top_type = type_scores[0] if type_scores else {"label": "其他图片", "score": 0.0}
        top_tags = [item for item in tag_scores[:5] if item["score"] >= 0.08]
        if not top_tags and tag_scores:
            top_tags = tag_scores[:3]
        return {
            "method": "Chinese-CLIP zero-shot image-text matching",
            "model": self.model_name,
            "device": self.device,
            "visual_type": {
                "label": top_type["label"],
                "confidence": top_type["score"],
                "candidates": type_scores[:5],
            },
            "semantic_tags": top_tags,
            "visual_summary": _build_visual_summary(top_type["label"], top_tags, ocr_text),
        }

    def _rank_labels(
        self,
        pil_image: Image.Image,
        labels: tuple[str, ...],
        prompts: list[str],
    ) -> list[dict[str, Any]]:
        inputs = self._processor(
            text=prompts,
            images=pil_image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=0).detach().cpu().tolist()
        ranked = sorted(
            (
                {"label": label, "score": round(float(score), 4)}
                for label, score in zip(labels, probs)
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        return ranked


def summarize_video_visual_semantics(frame_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    frame_semantics = [
        item.get("visual_semantics")
        for item in frame_results
        if isinstance(item.get("visual_semantics"), dict)
    ]
    if not frame_semantics:
        return None

    type_counter: Counter[str] = Counter()
    type_scores: dict[str, list[float]] = {}
    tag_counter: Counter[str] = Counter()
    tag_scores: dict[str, list[float]] = {}
    method = frame_semantics[0].get("method", "Chinese-CLIP zero-shot image-text matching")
    model = frame_semantics[0].get("model")
    device = frame_semantics[0].get("device")

    for semantics in frame_semantics:
        visual_type = semantics.get("visual_type", {})
        label = visual_type.get("label")
        if label:
            type_counter[str(label)] += 1
            type_scores.setdefault(str(label), []).append(float(visual_type.get("confidence") or 0.0))
        for tag in semantics.get("semantic_tags", []):
            tag_label = tag.get("label")
            if tag_label:
                tag_counter[str(tag_label)] += 1
                tag_scores.setdefault(str(tag_label), []).append(float(tag.get("score") or 0.0))

    top_type_label = type_counter.most_common(1)[0][0]
    frame_count = len(frame_semantics)
    type_candidates = [
        {
            "label": label,
            "score": round(sum(type_scores.get(label, [0.0])) / max(1, len(type_scores.get(label, []))), 4),
            "frame_count": count,
        }
        for label, count in type_counter.most_common(5)
    ]
    top_tags = [
        {
            "label": label,
            "score": round(sum(tag_scores.get(label, [0.0])) / max(1, len(tag_scores.get(label, []))), 4),
            "frame_count": count,
        }
        for label, count in tag_counter.most_common(5)
    ]
    return {
        "method": f"{method}; video frame vote",
        "model": model,
        "device": device,
        "visual_type": {
            "label": top_type_label,
            "confidence": round(type_counter[top_type_label] / frame_count, 4),
            "candidates": type_candidates,
        },
        "semantic_tags": top_tags,
        "visual_summary": _build_video_summary(top_type_label, top_tags, frame_count),
    }


def _build_visual_summary(visual_type: str, tags: list[dict[str, Any]], ocr_text: str) -> str:
    tag_text = "、".join(item["label"] for item in tags[:3]) if tags else "未形成稳定标签"
    text_hint = "含可识别文字" if str(ocr_text or "").strip() else "文字较少或无可识别文字"
    return f"Chinese-CLIP 判断该媒体更接近“{visual_type}”，视觉语义标签偏向：{tag_text}；{text_hint}。"


def _build_video_summary(visual_type: str, tags: list[dict[str, Any]], frame_count: int) -> str:
    tag_text = "、".join(item["label"] for item in tags[:3]) if tags else "未形成稳定标签"
    return f"Chinese-CLIP 对 {frame_count} 个抽帧/区域投票后，视频整体更接近“{visual_type}”，视觉语义标签偏向：{tag_text}。"


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
def _load_chinese_clip(model_name: str, device: str) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import ChineseCLIPModel, ChineseCLIPProcessor
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Chinese-CLIP dependencies are missing. Install torch and transformers, "
            "or disable visual semantics."
        ) from exc

    model = ChineseCLIPModel.from_pretrained(model_name)
    processor = ChineseCLIPProcessor.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return torch, model, processor
