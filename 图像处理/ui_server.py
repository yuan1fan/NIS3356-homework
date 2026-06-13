from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import sys
import time
import uuid
import warnings
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

warnings.filterwarnings("ignore", message="'cgi' is deprecated.*", category=DeprecationWarning)
import cgi  # noqa: E402


MODULE_ROOT = Path(__file__).resolve().parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from src.social_ocr.pipeline import IMAGE_SUFFIXES, make_llm_result, process_image, process_video  # noqa: E402
from src.social_ocr.video import VIDEO_LIKE_SUFFIXES, is_probable_video_file  # noqa: E402


VARIANT_SETS = {
    "full": None,
    "fast": {"original", "gray", "clahe", "clahe_sharpen"},
    "minimal": {"original", "clahe"},
}

UPLOAD_ROOT = MODULE_ROOT / "ui_uploads"
OUTPUT_ROOT = MODULE_ROOT / "ui_outputs"
UI_ROOT = MODULE_ROOT / "ui"


class OcrUiHandler(BaseHTTPRequestHandler):
    server_version = "SocialOcrUI/1.0"

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if path in {"", "/"}:
            self._send_file(UI_ROOT / "index.html")
            return
        if path.startswith("/ui/"):
            self._send_file(UI_ROOT / path.removeprefix("/ui/"))
            return
        if path.startswith("/outputs/"):
            self._send_output_file(path.removeprefix("/outputs/"))
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path != "/api/ocr":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        try:
            payload = self._handle_ocr_request()
            self._send_json(payload)
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[ui] {self.address_string()} - {format % args}")

    def _handle_ocr_request(self) -> dict[str, Any]:
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
            keep_blank_values=True,
        )

        files = form["files"] if "files" in form else []
        if not isinstance(files, list):
            files = [files]
        files = [item for item in files if getattr(item, "filename", "")]
        if not files:
            raise ValueError("No files uploaded.")

        mode = _field_value(form, "mode", "cpu")
        device = "gpu:0" if mode == "gpu" else "cpu"
        variant_set = _field_value(form, "variant_set", "minimal")
        platform = _field_value(form, "platform", "weibo_hotsearch")
        max_video_frames = int(_field_value(form, "max_video_frames", "32") or "32")
        frame_interval_raw = _field_value(form, "frame_interval", "")
        frame_interval = float(frame_interval_raw) if frame_interval_raw else None
        frame_regions = _field_values(form, "frame_regions") or ["full", "bottom"]
        enable_clip = _field_value(form, "enable_clip", "") == "on"
        clip_model = _field_value(form, "clip_model", "OFA-Sys/chinese-clip-vit-base-patch16")
        enable_caption = _field_value(form, "enable_caption", "") == "on"
        caption_model = _field_value(form, "caption_model", "Salesforce/blip-image-captioning-base")

        run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        upload_dir = UPLOAD_ROOT / run_id
        output_dir = OUTPUT_ROOT / run_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []

        for item in files:
            source_path = _save_upload(item, upload_dir)
            try:
                result = _process_uploaded_file(
                    source_path=source_path,
                    output_dir=output_dir,
                    platform=platform,
                    variant_names=VARIANT_SETS[variant_set],
                    device=device,
                    frame_interval=frame_interval,
                    max_video_frames=max_video_frames,
                    frame_regions=tuple(frame_regions),
                    enable_clip=enable_clip,
                    clip_model=clip_model,
                    enable_caption=enable_caption,
                    caption_model=caption_model,
                )
                results.append(_make_ui_result(result, source_path))
            except Exception as exc:  # noqa: BLE001
                errors.append({"filename": source_path.name, "error": str(exc)})

        if not results and errors:
            raise ValueError("; ".join(f"{item['filename']}: {item['error']}" for item in errors))

        summary = _summarize_results(results)
        return {
            "run_id": run_id,
            "mode": mode,
            "device": device,
            "variant_set": variant_set,
            "frame_regions": frame_regions,
            "enable_clip": enable_clip,
            "clip_model": clip_model,
            "enable_caption": enable_caption,
            "caption_model": caption_model,
            "summary": summary,
            "results": results,
            "errors": errors,
        }

    def _send_file(self, path: Path) -> None:
        safe_path = path.resolve()
        if not _is_relative_to(safe_path, UI_ROOT.resolve()) or not safe_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        content_type = mimetypes.guess_type(str(safe_path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(safe_path.stat().st_size))
        self.end_headers()
        with safe_path.open("rb") as file:
            shutil.copyfileobj(file, self.wfile)

    def _send_output_file(self, relative_path: str) -> None:
        path = (OUTPUT_ROOT / relative_path).resolve()
        if not _is_relative_to(path, OUTPUT_ROOT.resolve()) or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with path.open("rb") as file:
            shutil.copyfileobj(file, self.wfile)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _field_value(form: cgi.FieldStorage, name: str, default: str = "") -> str:
    if name not in form:
        return default
    field = form[name]
    if isinstance(field, list):
        field = field[0]
    return str(field.value or default)


def _field_values(form: cgi.FieldStorage, name: str) -> list[str]:
    if name not in form:
        return []
    field = form[name]
    if not isinstance(field, list):
        field = [field]
    return [str(item.value) for item in field if str(item.value)]


def _save_upload(item: cgi.FieldStorage, upload_dir: Path) -> Path:
    filename = Path(str(item.filename)).name
    safe_name = _safe_filename(filename)
    path = upload_dir / safe_name
    with path.open("wb") as output:
        shutil.copyfileobj(item.file, output)
    return path


def _safe_filename(filename: str) -> str:
    stem = Path(filename).stem or "upload"
    suffix = Path(filename).suffix.lower()
    safe_stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)
    return f"{safe_stem[:80]}_{uuid.uuid4().hex[:8]}{suffix}"


def _process_uploaded_file(
    source_path: Path,
    output_dir: Path,
    platform: str,
    variant_names: set[str] | None,
    device: str,
    frame_interval: float | None,
    max_video_frames: int,
    frame_regions: tuple[str, ...],
    enable_clip: bool,
    clip_model: str,
    enable_caption: bool,
    caption_model: str,
) -> dict[str, Any]:
    suffix = source_path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return process_image(
            source_path,
            output_dir=output_dir,
            platform=platform,
            variant_names=variant_names,
            device=device,
            enable_clip=enable_clip,
            clip_model=clip_model,
            enable_caption=enable_caption,
            caption_model=caption_model,
        )
    if suffix in VIDEO_LIKE_SUFFIXES and is_probable_video_file(source_path):
        return process_video(
            source_path,
            output_dir=output_dir,
            platform=platform,
            frame_interval_seconds=frame_interval,
            max_frames=max_video_frames,
            frame_regions=frame_regions,
            variant_names=variant_names,
            device=device,
            enable_clip=enable_clip,
            clip_model=clip_model,
            enable_caption=enable_caption,
            caption_model=caption_model,
        )
    raise ValueError(f"Unsupported or unreadable media file: {source_path.name}")


def _make_ui_result(result: dict[str, Any], source_path: Path) -> dict[str, Any]:
    llm_result = make_llm_result(result)
    previews = _preview_items(result)
    return {
        **llm_result,
        "filename": source_path.name,
        "source_media": str(source_path),
        "previews": previews,
    }


def _preview_items(result: dict[str, Any]) -> list[dict[str, str]]:
    paths: list[Path] = []
    if result.get("visualization_path"):
        paths.append(Path(result["visualization_path"]))
    if result.get("media_type") == "video":
        for frame_result in result.get("frame_results", [])[:4]:
            if frame_result.get("visualization_path"):
                paths.append(Path(frame_result["visualization_path"]))
    previews = []
    for path in paths:
        try:
            url = "/outputs/" + path.resolve().relative_to(OUTPUT_ROOT.resolve()).as_posix()
        except ValueError:
            continue
        previews.append({"url": url, "label": path.name})
    return previews


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"avg_confidence": 0.0, "char_count": 0, "needs_review_count": 0}
    qualities = [item.get("ocr_quality", {}) for item in results]
    confidences = [
        float(quality.get("avg_confidence") or 0.0)
        for quality in qualities
        if quality.get("avg_confidence") is not None
    ]
    char_count = sum(int(quality.get("char_count") or 0) for quality in qualities)
    needs_review_count = sum(1 for quality in qualities if quality.get("needs_review"))
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return {
        "avg_confidence": round(avg_confidence, 4),
        "char_count": char_count,
        "needs_review_count": needs_review_count,
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the local OCR module UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), OcrUiHandler)
    print(f"OCR UI running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
