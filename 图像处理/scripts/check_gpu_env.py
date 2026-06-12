from __future__ import annotations

import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from src.social_ocr.ocr_engine import _add_windows_nvidia_dll_dirs  # noqa: E402


def main() -> None:
    _add_windows_nvidia_dll_dirs()

    import paddle  # noqa: PLC0415

    print(f"Paddle version: {paddle.__version__}")
    print(f"Compiled with CUDA: {paddle.device.is_compiled_with_cuda()}")
    if not paddle.device.is_compiled_with_cuda():
        raise SystemExit("Current Paddle installation is CPU-only.")

    paddle.set_device("gpu:0")
    x = paddle.randn([256, 256])
    y = x @ x
    print(f"Selected device: {paddle.device.get_device()}")
    print(f"Tensor place: {y.place}")
    print("GPU environment check passed.")


if __name__ == "__main__":
    main()
