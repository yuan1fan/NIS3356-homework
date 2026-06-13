from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .preprocess import read_image, write_image


class TorchVisualWorker:
    """Run Torch-based visual models in a separate process.

    PaddleOCR and PyTorch CUDA wheels can conflict on Windows when they are
    imported into the same process. This worker keeps Chinese-CLIP/captioning
    on GPU while the main process stays dedicated to PaddleOCR.
    """

    def __init__(
        self,
        clip_model: str,
        caption_model: str,
        device: str = "cpu",
        python_executable: str | Path | None = None,
    ) -> None:
        self.clip_model = clip_model
        self.caption_model = caption_model
        self.device = _torch_device_name(device)
        self.python_executable = str(python_executable or _default_torch_python())
        self._process = _start_worker_process(self.python_executable)

    def analyze_image(self, image: np.ndarray, ocr_text: str = "") -> dict[str, Any]:
        return self._request_with_image(
            "analyze",
            image,
            {
                "ocr_text": ocr_text,
                "clip_model": self.clip_model,
            },
        )

    def caption_image(self, image: np.ndarray) -> dict[str, Any]:
        return self._request_with_image(
            "caption",
            image,
            {
                "caption_model": self.caption_model,
            },
        )

    def close(self) -> None:
        process = self._process
        if process.poll() is not None:
            return
        try:
            self._send({"task": "close"})
        except Exception:  # noqa: BLE001
            process.terminate()

    def _request_with_image(
        self,
        task: str,
        image: np.ndarray,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            write_image(temp_path, image)
            response = self._send(
                {
                    "task": task,
                    "image_path": str(temp_path),
                    "device": self.device,
                    **payload,
                }
            )
            return response
        finally:
            temp_path.unlink(missing_ok=True)

    def _send(self, payload: dict[str, Any]) -> dict[str, Any]:
        process = self._process
        if process.poll() is not None:
            raise RuntimeError(f"Torch visual worker exited with code {process.returncode}.")
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("Torch visual worker did not return a response.")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "Torch visual worker failed."))
        return response.get("result", {})

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def _default_torch_python() -> Path:
    env_path = os.environ.get("SOCIAL_OCR_TORCH_PYTHON")
    if env_path:
        return Path(env_path)
    module_root = Path(__file__).resolve().parents[2]
    candidate = module_root / ".venv-paddle-gpu" / "Scripts" / "python.exe"
    return candidate if candidate.exists() else Path(sys.executable)


def _start_worker_process(python_executable: str) -> subprocess.Popen[str]:
    module_root = Path(__file__).resolve().parents[2]
    return subprocess.Popen(
        [
            python_executable,
            "-u",
            "-m",
            "src.social_ocr.torch_visual_worker",
            "--worker",
        ],
        cwd=str(module_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )


def _torch_device_name(device: str) -> str:
    if device.startswith("gpu"):
        suffix = device.split(":", 1)[1] if ":" in device else "0"
        return f"cuda:{suffix}"
    return device


def _worker_main() -> None:
    analyzers: dict[tuple[str, str], Any] = {}
    captioners: dict[tuple[str, str], Any] = {}
    for line in sys.stdin:
        try:
            request = json.loads(line)
            task = request.get("task")
            if task == "close":
                _write_response({})
                return
            device = str(request.get("device") or "cpu")
            image = read_image(request["image_path"])
            if task == "analyze":
                model_name = str(request.get("clip_model") or "OFA-Sys/chinese-clip-vit-base-patch16")
                key = (model_name, device)
                if key not in analyzers:
                    from .clip_semantics import ChineseClipAnalyzer

                    analyzers[key] = ChineseClipAnalyzer(model_name=model_name, device=device)
                result = analyzers[key].analyze_image(image, str(request.get("ocr_text") or ""))
            elif task == "caption":
                model_name = str(request.get("caption_model") or "Salesforce/blip-image-captioning-base")
                key = (model_name, device)
                if key not in captioners:
                    from .captioning import ImageCaptioner

                    captioners[key] = ImageCaptioner(model_name=model_name, device=device)
                result = captioners[key].caption_image(image)
            else:
                raise ValueError(f"Unsupported worker task: {task}")
            _write_response(result)
        except Exception as exc:  # noqa: BLE001
            _write_response({}, ok=False, error=str(exc))


def _write_response(result: dict[str, Any], ok: bool = True, error: str | None = None) -> None:
    print(
        json.dumps(
            {
                "ok": ok,
                "result": result,
                "error": error,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__" and "--worker" in sys.argv:
    _worker_main()
