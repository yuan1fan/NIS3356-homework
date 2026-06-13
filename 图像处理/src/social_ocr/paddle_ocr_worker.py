from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .ocr_engine import parse_paddle_result
from .postprocess import OcrResult
from .preprocess import write_image


class PaddleOcrWorker:
    """Run PaddleOCR in a dedicated GPU process.

    This keeps the main application free of Paddle CUDA DLLs, which avoids
    Windows conflicts with Torch CUDA when visual semantic models are enabled.
    """

    def __init__(self, device: str = "gpu:0", python_executable: str | Path | None = None) -> None:
        self.device = device
        self.python_executable = str(python_executable or _default_ocr_python())
        self._process = _start_worker_process(self.python_executable)

    def recognize(self, image, variant_name: str) -> OcrResult:
        temp_path = Path(self._send_image(image))
        try:
            response = self._send(
                {
                    "task": "recognize",
                    "image_path": str(temp_path),
                    "variant_name": variant_name,
                    "device": self.device,
                }
            )
            blocks = parse_paddle_result(
                response.get("raw_result"),
                int(response.get("image_width") or image.shape[1]),
                int(response.get("image_height") or image.shape[0]),
            )
            return OcrResult(
                variant_name=variant_name,
                text_blocks=blocks,
                raw_result=response.get("raw_result"),
            )
        finally:
            temp_path.unlink(missing_ok=True)

    def close(self) -> None:
        process = self._process
        if process.poll() is not None:
            return
        try:
            self._send({"task": "close"})
        except Exception:  # noqa: BLE001
            process.terminate()

    def _send_image(self, image) -> str:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            path = Path(temp_file.name)
        write_image(path, image)
        return str(path)

    def _send(self, payload: dict[str, Any]) -> dict[str, Any]:
        process = self._process
        if process.poll() is not None:
            raise RuntimeError(f"Paddle OCR worker exited with code {process.returncode}.")
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        while line and not line.lstrip().startswith("{"):
            line = process.stdout.readline()
        if not line:
            raise RuntimeError("Paddle OCR worker did not return a response.")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error") or "Paddle OCR worker failed."))
        return response.get("result", {})

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def _default_ocr_python() -> Path:
    env_path = os.environ.get("SOCIAL_OCR_PADDLE_PYTHON")
    if env_path:
        return Path(env_path)
    module_root = Path(__file__).resolve().parents[2]
    candidate = module_root / ".venv-ocr-gpu" / "Scripts" / "python.exe"
    return candidate if candidate.exists() else Path(sys.executable)


def _start_worker_process(python_executable: str) -> subprocess.Popen[str]:
    module_root = Path(__file__).resolve().parents[2]
    return subprocess.Popen(
        [
            python_executable,
            "-u",
            "-m",
            "src.social_ocr.paddle_ocr_worker",
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


def _worker_main() -> None:
    engines: dict[str, Any] = {}
    for line in sys.stdin:
        try:
            request = json.loads(line)
            task = request.get("task")
            if task == "close":
                _write_response({})
                return
            if task != "recognize":
                raise ValueError(f"Unsupported worker task: {task}")
            device = str(request.get("device") or "gpu:0")
            if device not in engines:
                from .ocr_engine import PaddleOcrEngine

                engines[device] = PaddleOcrEngine(device=device)
            image_path = Path(request["image_path"])
            raw = engines[device]._run_ocr(image_path)
            import cv2

            image = cv2.imdecode(
                __import__("numpy").fromfile(str(image_path), dtype=__import__("numpy").uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                height = width = 0
            else:
                height, width = image.shape[:2]
            _write_response(
                {
                    "raw_result": _jsonable(raw),
                    "image_width": width,
                    "image_height": height,
                }
            )
        except Exception as exc:  # noqa: BLE001
            _write_response({}, ok=False, error=str(exc))


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
