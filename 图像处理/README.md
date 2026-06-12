# 社交媒体热点图像 OCR 与结构化模块

本模块用于课程设计中的数字图像处理部分，核心功能是基于 PaddleOCR 对微博/小红书等社交媒体图片进行文字识别，并通过多版本图像增强自动选择最佳 OCR 结果，输出 JSON 给 NLP、ASR、数据分析或大模型汇总模块使用。

## 功能

- 图像预处理：灰度化、CLAHE 对比度增强、锐化、去噪、二值化、放大、组合增强。
- OCR：调用 PaddleOCR 识别中文社媒图片和视频抽帧文字。
- 自动择优：对每个预处理版本分别 OCR，根据平均置信度、文本长度、文本块数量、低置信度比例和文本区域占比综合打分。
- 视频文本提取：对微博视频按时间间隔抽帧，并可分别识别整帧、顶部、中部、底部区域，适配字幕、标题贴片和画面文字。
- 真实爬取媒体处理：支持递归读取 `数据爬取/outputs/.../media` 中的图片、`.mp4` 视频和可识别为视频的 `.bin` 文件。
- 置信度评估：输出平均置信度、最低置信度、低置信度比例、是否需要人工复核。
- JSON 输出：详细 JSON 用于调试和画框，精简 LLM JSON 用于交给大模型汇总。
- 可视化：在图片上画出 OCR 文本框和置信度。

## 快速运行

```powershell
python -m pip install -r requirements.txt
python scripts/create_sample_images.py
python scripts/fetch_hotlist_sample.py
python run_ocr.py --input-dir data/raw --output-dir outputs --platform weibo
```

单张图片：

```powershell
python run_ocr.py --image data/raw/sample_weibo_hotsearch.png --output-dir outputs --platform weibo
```

单个视频抽帧 OCR：

```powershell
python run_ocr.py `
  --video "..\数据爬取\outputs\20260612_141245\media\02_老外也疑惑中国为什么不参加世界杯\videos\video_3a2c8bc4eeba.mp4" `
  --output-dir outputs_video_test `
  --platform weibo_hotsearch `
  --variant-set minimal `
  --frame-regions full,bottom
```

处理微博爬取结果中的图片和视频：

```powershell
python scripts/process_latest_crawl_media.py
python scripts/process_latest_crawl_media.py --mode gpu
```

该脚本会自动查找 `..\数据爬取\outputs` 中最新的爬取结果目录，并处理其中 `media/` 下的所有图片和视频。输出默认写入：

```text
outputs_from_crawl/<爬取结果目录名>/
```

快速试跑：

```powershell
python scripts/process_latest_crawl_media.py --dry-run
python scripts/process_latest_crawl_media.py --image-limit 5 --video-limit 2
python scripts/process_latest_crawl_media.py --mode gpu --image-limit 5 --video-limit 2
```

脚本默认会显示进度条和预计剩余时间；如需静默运行可加 `--quiet`。

也可以手动指定某次爬取结果：

```powershell
python scripts/process_latest_crawl_media.py `
  --crawl-run-dir "..\数据爬取\outputs\20260612_141245" `
  --output-dir outputs_weibo_media
```

默认抽帧规则：视频不足 64 秒时每 2 秒抽一帧；视频达到 64 秒及以上时使用 `视频时长 / 32` 作为间隔，最多抽 32 帧。可以用 `--frame-interval` 和 `--max-video-frames` 手动覆盖。最终给大模型的 `ocr_text_compact` 会对相邻帧重复文本自动去重。

说明：`.bin` 文件会先做文件头判断。HTML 播放页不会当作视频处理；只有可被识别为 MP4/WebM/AVI 等视频容器的数据才会进入抽帧 OCR。

抓取真实网页截图数据：

```powershell
python scripts/collect_real_web_screenshots.py --output-dir data/real_raw --count 40
```

真实数据抽样 OCR：

```powershell
python run_ocr.py --input-dir data/real_raw --output-dir outputs_real_minimal --platform real_web --limit 6 --variant-set minimal
```

指定推理设备：

```powershell
python run_ocr.py --image data/raw/sample_weibo_hotsearch.png --mode cpu
python run_ocr.py --image data/raw/sample_weibo_hotsearch.png --mode gpu
python run_ocr.py --image data/raw/sample_weibo_hotsearch.png --device gpu:1
```

`--mode gpu` 默认使用 `gpu:0`；如果要指定第二张显卡或特殊 Paddle 设备名，可以用 `--device` 覆盖。

GPU 需要安装 CUDA 版 PaddlePaddle。建议为 PaddleOCR 单独创建干净环境，避免和其他深度学习框架的 CUDA/cuDNN DLL 冲突。

## 输出

- `outputs/json/*.json`：每张图片的结构化 OCR 结果。
- `outputs/llm_json/*.json`：面向大模型的精简 OCR 结果，不包含坐标和多版本细节。
- `outputs/visualizations/*.png`：最佳版本 OCR 框可视化。
- `outputs/variants/<image_id>/*.png`：不同预处理版本图片。
- `outputs/video_frames/<video_id>/*.jpg`：视频抽取的原始帧。
- `outputs/frame_ocr/`：每个视频帧/区域的 OCR 详细 JSON、LLM JSON 和可视化。
- `outputs/reports/batch_summary.json`：批处理汇总。
- `outputs/reports/llm_batch_summary.json`：面向大模型的批处理精简汇总。
- `outputs/reports/media_batch_summary.json`：混合图片/视频媒体处理汇总。
- `outputs/reports/llm_media_batch_summary.json`：面向大模型的混合媒体精简汇总。

说明：`outputs/json` 会保留文本框坐标、区域类型和所有预处理版本结果，文件会比较大，主要用于调试、可视化和课程报告证明。给大模型使用时优先读取 `outputs/llm_json` 或 `outputs/reports/llm_batch_summary.json` 中的 `ocr_text_compact`。

完整批处理会对每张图运行 8 个预处理版本，CPU 上耗时会明显高于单张 OCR。调试时建议先用 `--image` 跑单张图片。

`--variant-set` 可选：

- `full`：8 个版本，适合最终对比实验。
- `fast`：4 个版本，适合中等规模抽样。
- `minimal`：原图 + CLAHE，适合几十张真实截图的初筛。

GPU 环境建议：

```powershell
cd 图像处理
python -m venv .venv-paddle-gpu
.\.venv-paddle-gpu\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-gpu.txt
python scripts/check_gpu_env.py
python scripts/process_latest_crawl_media.py --mode gpu --dry-run
python scripts/process_latest_crawl_media.py --mode gpu --variant-set minimal
```

注意：GPU 环境中不要再执行 `python -m pip install -r requirements.txt`，因为普通依赖文件包含 CPU 版 `paddlepaddle`。如需 GPU 环境依赖，请使用 `requirements-gpu.txt`。

本机 `.venv-paddle-gpu` 已验证 `paddlepaddle-gpu==3.3.1` 可以识别 `gpu:0` 并执行张量运算。完整 OCR 是否明显加速取决于图片/视频数量、预处理版本数和视频抽帧规模；几十张图片加大量视频帧通常会比 CPU 快不少，但视频解码和图片预处理仍有一部分在 CPU 上执行。

## 已验证命令

```powershell
python -m pytest tests -q
python scripts/create_sample_images.py
python scripts/fetch_hotlist_sample.py
python scripts/collect_real_web_screenshots.py --output-dir data/real_raw --count 40
python run_ocr.py --input-dir data/raw --output-dir outputs --platform social_media
python run_ocr.py --input-dir data/real_raw --output-dir outputs_real_minimal --platform real_web --limit 6 --variant-set minimal
```

当前真实截图数据集位于 `data/real_raw`，共 37 张，来源包括百度热榜、澎湃新闻、人民网、央视新闻、腾讯新闻、IT之家。来源元数据保存在 `data/real_raw/metadata.jsonl`。

## 创新点

1. 多版本图像增强自动择优 OCR，不固定使用单一预处理流程。
2. 将 OCR 结果扩展为结构化图像信息，包括坐标、区域类型、置信度和质量评估。
3. 同时输出详细 OCR JSON 和 LLM 精简 JSON，其中 `ocr_text_compact` 可直接交给大模型，避免读取大量坐标噪声。
4. 对低质量、压缩、模糊社媒图像进行 OCR 质量评估并标记是否需要人工复核。
