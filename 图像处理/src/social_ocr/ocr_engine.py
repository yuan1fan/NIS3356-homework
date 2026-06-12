from __future__ import annotations

import inspect
import os
import sysconfig
import tempfile
from pathlib import Path
from typing import Any

from .postprocess import OcrResult, TextBlock, infer_region_type, normalize_text
from .preprocess import write_image


_DLL_DIR_HANDLES = []


class PaddleOcrEngine:
    """Thin adapter around PaddleOCR with compatibility for common 2.x/3.x outputs."""

    def __init__(
        self,
        lang: str = "ch",
        text_detection_model_name: str = "PP-OCRv5_mobile_det",
        text_recognition_model_name: str = "PP-OCRv5_mobile_rec",
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
        device: str = "cpu",
        enable_mkldnn: bool = False,
        cpu_threads: int = 4,
    ) -> None:
        if device.startswith("cpu"):
            # Windows CPU inference can hit PaddlePaddle oneDNN/PIR incompatibilities.
            # Disabling oneDNN is slower but more stable for course-project machines.
            os.environ.setdefault("FLAGS_use_mkldnn", "0")
            os.environ.setdefault("FLAGS_use_onednn", "0")
        elif os.name == "nt":
            _add_windows_nvidia_dll_dirs()
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is not installed. Install with: python -m pip install paddleocr paddlepaddle"
            ) from exc

        init_options = {
            "text_detection_model_name": text_detection_model_name,
            "text_recognition_model_name": text_recognition_model_name,
            "use_angle_cls": use_textline_orientation,
            "use_textline_orientation": use_textline_orientation,
            "use_doc_orientation_classify": use_doc_orientation_classify,
            "use_doc_unwarping": use_doc_unwarping,
            "device": device,
            "enable_mkldnn": enable_mkldnn,
            "cpu_threads": cpu_threads,
            "show_log": False,
        }
        if not text_detection_model_name and not text_recognition_model_name:
            init_options["lang"] = lang
        kwargs = self._filter_init_kwargs(PaddleOCR, init_options)
        self.ocr = PaddleOCR(**kwargs)

    @staticmethod
    def _filter_init_kwargs(cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(cls)
            explicit_parameters = {
                key
                for key, param in signature.parameters.items()
                if param.kind != inspect.Parameter.VAR_KEYWORD
            }
            if explicit_parameters:
                common_parameters = {
                    "device",
                    "engine",
                    "engine_config",
                    "enable_hpi",
                    "use_tensorrt",
                    "precision",
                    "enable_mkldnn",
                    "mkldnn_cache_capacity",
                    "cpu_threads",
                    "enable_cinn",
                }
                accepted = explicit_parameters | common_parameters
                return {key: value for key, value in kwargs.items() if key in accepted}
            return kwargs
        except (TypeError, ValueError):
            return kwargs

    def recognize(self, image, variant_name: str) -> OcrResult:
        # PaddleOCR 3.x accepts paths reliably across pipeline variants.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            write_image(temp_path, image)
            raw = self._run_ocr(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

        blocks = parse_paddle_result(raw, image.shape[1], image.shape[0])
        return OcrResult(variant_name=variant_name, text_blocks=blocks, raw_result=raw)

    def _run_ocr(self, image_path: Path) -> Any:
        if hasattr(self.ocr, "predict"):
            return self.ocr.predict(str(image_path))
        return self.ocr.ocr(str(image_path), cls=True)


def _add_windows_nvidia_dll_dirs() -> None:
    site_packages = Path(sysconfig.get_paths()["purelib"])
    candidates = [
        site_packages / "nvidia" / "cu13" / "bin" / "x86_64",
        site_packages / "nvidia" / "cudnn" / "bin",
    ]
    for path in candidates:
        if path.exists():
            path_str = str(path)
            if path_str not in os.environ.get("PATH", ""):
                os.environ["PATH"] = path_str + os.pathsep + os.environ.get("PATH", "")
            if path_str not in {str(handle) for handle in _DLL_DIR_HANDLES}:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(path_str))


def parse_paddle_result(raw: Any, image_width: int, image_height: int) -> list[TextBlock]:
    blocks: list[TextBlock] = []

    for page in _as_pages(raw):
        if isinstance(page, dict):
            blocks.extend(_parse_page_dict(page, image_width, image_height))
        else:
            blocks.extend(_parse_legacy_items(page, image_width, image_height))

    blocks = [block for block in blocks if block.text]
    return sorted(blocks, key=lambda block: _sort_key(block.box))


def _as_pages(raw: Any) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    return [raw]


def _parse_page_dict(page: dict[str, Any], image_width: int, image_height: int) -> list[TextBlock]:
    texts = _pick_page_field(page, ("rec_texts", "texts", "text", "recognized_text"))
    scores = _pick_page_field(page, ("rec_scores", "scores", "confidence"))
    boxes = _pick_page_field(page, ("rec_polys", "dt_polys", "rec_boxes", "boxes"))

    if isinstance(texts, str):
        texts = [texts]
    if isinstance(scores, (int, float)):
        scores = [float(scores)]
    if hasattr(texts, "tolist"):
        texts = texts.tolist()
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if hasattr(boxes, "tolist") and _safe_len(boxes) == 0:
        boxes = []

    blocks: list[TextBlock] = []
    for index, text in enumerate(texts):
        box = _normalize_box(boxes[index] if index < _safe_len(boxes) else [])
        confidence = _safe_float(scores[index] if index < _safe_len(scores) else 0.0)
        normalized = normalize_text(str(text))
        blocks.append(
            TextBlock(
                text=normalized,
                confidence=confidence,
                box=box,
                region_type=infer_region_type(box, image_width, image_height),
            )
        )
    return blocks


def _pick_page_field(page: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key not in page:
            continue
        value = page.get(key)
        if _has_items(value):
            return value
    return []


def _has_items(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (int, float)):
        return True
    if hasattr(value, "size"):
        return int(value.size) > 0
    try:
        return len(value) > 0
    except TypeError:
        return True


def _safe_len(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


def _parse_legacy_items(page: Any, image_width: int, image_height: int) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    if not isinstance(page, list):
        return blocks

    for item in page:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        box = _normalize_box(item[0])
        text = ""
        confidence = 0.0
        payload = item[1]
        if isinstance(payload, (list, tuple)) and payload:
            text = str(payload[0])
            if len(payload) > 1:
                confidence = _safe_float(payload[1])
        elif isinstance(payload, str):
            text = payload
        blocks.append(
            TextBlock(
                text=normalize_text(text),
                confidence=confidence,
                box=box,
                region_type=infer_region_type(box, image_width, image_height),
            )
        )
    return blocks


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _normalize_box(box: Any) -> list[list[float]]:
    if box is None:
        return []
    if hasattr(box, "tolist"):
        box = box.tolist()

    if isinstance(box, (list, tuple)) and len(box) == 4 and all(
        isinstance(value, (int, float)) for value in box
    ):
        x_min, y_min, x_max, y_max = [float(value) for value in box]
        return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]

    normalized: list[list[float]] = []
    if isinstance(box, (list, tuple)):
        for point in box:
            if hasattr(point, "tolist"):
                point = point.tolist()
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                normalized.append([float(point[0]), float(point[1])])
    return normalized


def _sort_key(box: list[list[float]]) -> tuple[float, float]:
    if not box:
        return (0.0, 0.0)
    xs = [point[0] for point in box]
    ys = [point[1] for point in box]
    return (min(ys), min(xs))
