from __future__ import annotations

import sys
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from src.social_ocr.paddle_ocr_worker import _default_ocr_python  # noqa: E402
from src.social_ocr.torch_visual_worker import _default_torch_python  # noqa: E402


def main() -> None:
    print(f"Paddle OCR worker Python: {_default_ocr_python()}")
    print(f"Torch visual worker Python: {_default_torch_python()}")
    check_paddle_worker()
    check_torch_worker()
    print("Full GPU environment check passed.")


def check_paddle_worker() -> None:
    import subprocess

    code = (
        "from src.social_ocr.ocr_engine import _add_windows_nvidia_dll_dirs; "
        "_add_windows_nvidia_dll_dirs(); "
        "import paddle; "
        "print(f'Paddle version: {paddle.__version__}'); "
        "print(f'Paddle CUDA: {paddle.device.is_compiled_with_cuda()}'); "
        "paddle.set_device('gpu:0'); "
        "x=paddle.randn([256,256]); y=x@x; "
        "print(f'Paddle selected device: {paddle.device.get_device()}'); "
        "print(f'Paddle tensor place: {y.place}')"
    )
    subprocess.run([str(_default_ocr_python()), "-c", code], cwd=MODULE_ROOT, check=True)


def check_torch_worker() -> None:
    import subprocess

    code = (
        "import torch; "
        "print(f'Torch version: {torch.__version__}'); "
        "print(f'Torch CUDA: {torch.cuda.is_available()} {torch.version.cuda}'); "
        "x=torch.randn((256,256), device='cuda:0'); y=x@x; "
        "print(f'Torch device: {y.device}')"
    )
    subprocess.run([str(_default_torch_python()), "-c", code], cwd=MODULE_ROOT, check=True)


if __name__ == "__main__":
    main()
