# 社交媒体热点图像 OCR 与结构化模块

本模块用于课程设计中的数字图像处理部分，核心功能是基于 PaddleOCR 对微博/小红书等社交媒体图片进行文字识别，并通过多版本图像增强自动选择最佳 OCR 结果，输出 JSON 给 NLP、ASR、数据分析或大模型汇总模块使用。

## 功能

- 图像预处理：灰度化、CLAHE 对比度增强、锐化、去噪、二值化、放大、组合增强。
- OCR：调用 PaddleOCR 识别中文社媒图片文字。
- 自动择优：对每个预处理版本分别 OCR，根据平均置信度、文本长度、文本块数量、低置信度比例和文本区域占比综合打分。
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
python run_ocr.py --image data/raw/sample_weibo_hotsearch.png --device cpu
python run_ocr.py --image data/raw/sample_weibo_hotsearch.png --device gpu:0
```

GPU 需要安装 CUDA 版 PaddlePaddle。当前项目代码已支持 `--device gpu:0`，但如果同一 Python 环境里同时安装了 PyTorch CUDA 和 Paddle CUDA，Windows 下可能出现 CUDA/cuDNN DLL 冲突。遇到这种情况，建议为 PaddleOCR 单独创建干净环境。

## 输出

- `outputs/json/*.json`：每张图片的结构化 OCR 结果。
- `outputs/llm_json/*.json`：面向大模型的精简 OCR 结果，不包含坐标和多版本细节。
- `outputs/visualizations/*.png`：最佳版本 OCR 框可视化。
- `outputs/variants/<image_id>/*.png`：不同预处理版本图片。
- `outputs/reports/batch_summary.json`：批处理汇总。
- `outputs/reports/llm_batch_summary.json`：面向大模型的批处理精简汇总。

说明：`outputs/json` 会保留文本框坐标、区域类型和所有预处理版本结果，文件会比较大，主要用于调试、可视化和课程报告证明。给大模型使用时优先读取 `outputs/llm_json` 或 `outputs/reports/llm_batch_summary.json`。

完整批处理会对每张图运行 8 个预处理版本，CPU 上耗时会明显高于单张 OCR。调试时建议先用 `--image` 跑单张图片。

`--variant-set` 可选：

- `full`：8 个版本，适合最终对比实验。
- `fast`：4 个版本，适合中等规模抽样。
- `minimal`：原图 + CLAHE，适合几十张真实截图的初筛。

GPU 环境建议：

```powershell
python -m venv .venv-paddle-gpu
.\.venv-paddle-gpu\Scripts\Activate.ps1
python -m pip install paddleocr
python -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
python run_ocr.py --image data/raw/sample_weibo_hotsearch.png --device gpu:0 --variant-set minimal
```

本机曾验证 `paddlepaddle-gpu==3.3.1` 可以识别 `gpu:0` 并执行张量运算，但在当前全局环境中 PaddleOCR/PaddleX 初始化 GPU 推理器会触发 CUDA/cuDNN DLL 冲突，因此默认仍使用 CPU。

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

## 适合写进报告的创新点

1. 多版本图像增强自动择优 OCR，不固定使用单一预处理流程。
2. 将 OCR 结果扩展为结构化图像信息，包括坐标、区域类型、置信度和质量评估。
3. 同时输出详细 OCR JSON 和 LLM 精简 JSON，避免大模型读取大量坐标噪声。
4. 对低质量、压缩、模糊社媒图像进行 OCR 质量评估并标记是否需要人工复核。
