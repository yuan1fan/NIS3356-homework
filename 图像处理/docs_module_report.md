# 数字图像处理模块说明

## 模块名称

基于 PaddleOCR 的社交媒体热点图像文字识别与结构化分析模块。

## 已实现内容

- PaddleOCR 中文 OCR 识别。
- 多版本图像预处理：原图、灰度、CLAHE、锐化、去噪、自适应二值化、放大、CLAHE+锐化。
- 多版本 OCR 自动择优：根据平均置信度、文本长度、文本块数量、低置信度比例、文本区域占比综合评分。
- OCR 置信度评估：平均置信度、最低置信度、低置信度比例、是否需要人工复核。
- JSON 输出：包含识别文本、文本框坐标、置信度、区域类型、质量评价和大模型摘要字段。
- LLM 精简 JSON：只保留 OCR 文本、置信度、最佳预处理版本和少量统计字段，供大模型汇总分析。
- OCR 可视化：在最佳预处理图像上绘制文本框和置信度。
- 样例数据：仿微博热搜图、仿小红书封面图、压缩低质微博图、可选在线百度热榜样例。

## 处理流程

1. 读取图片并限制最长边，降低超大图对推理速度的影响。
2. 生成多个预处理版本。
3. 对每个版本调用 PaddleOCR。
4. 解析 OCR 文本、坐标和置信度。
5. 按空间位置推断文本区域类型，如 header、title、body、footer。
6. 计算质量指标和综合评分。
7. 选择最佳版本并输出 JSON、可视化图片和批处理汇总。

## 综合评分公式

综合评分由以下因素组成：

- 平均 OCR 置信度：55%。
- 有效文本长度：20%。
- 文本块数量：10%。
- 文本区域占比：10%。
- 低置信度文本比例惩罚：5%。

该设计避免只看置信度造成“漏识别但高置信”的问题，也避免只看文本长度导致噪声文本过多。

## 创新点

1. 多版本图像增强自动择优，不固定依赖某一种预处理方法。
2. 把 OCR 从纯文本识别扩展为结构化图像理解，保留文本框、区域类型和质量指标。
3. 输出 `ocr_text_compact` 字段，方便直接接入小组的大模型汇总分析。
4. 对低质、压缩、模糊图片进行质量评估，能判断是否需要人工复核。

## 测试结果

当前已通过：

- `python -m pytest tests -q`
- `python scripts/create_sample_images.py`
- `python scripts/fetch_hotlist_sample.py`
- `python scripts/collect_real_web_screenshots.py --output-dir data/real_raw --count 40`
- `python run_ocr.py --input-dir data/raw --output-dir outputs --platform social_media`
- `python run_ocr.py --input-dir data/real_raw --output-dir outputs_real_minimal --platform real_web --limit 6 --variant-set minimal`

批处理样例结果：

| 图片 | 最佳预处理版本 | 综合评分 | 平均置信度 | 是否需要复核 |
|---|---|---:|---:|---|
| sample_baidu_hotlist_online | original | 0.9982 | 0.9968 | 否 |
| sample_weibo_hotsearch | gray | 0.9852 | 0.9981 | 否 |
| sample_xiaohongshu_cover | clahe_sharpen | 0.8750 | 0.9520 | 否 |
| sample_weibo_hotsearch_compressed | clahe | 0.9936 | 0.9884 | 否 |

其中 `sample_baidu_hotlist_online` 由 `scripts/fetch_hotlist_sample.py` 从百度实时热榜页面抓取标题后渲染成 OCR 测试图像，用于模拟热点榜单截图输入。

## 真实网页截图数据

除生成样例外，已使用 Playwright 从公开网页抓取真实热点/新闻页面截图，共 37 张，保存于 `data/real_raw`。每张图片对应的来源、URL、页面标题、截图时间、滚动位置保存在 `data/real_raw/metadata.jsonl`。

来源分布：

| 来源 | 数量 |
|---|---:|
| 人民网 | 8 |
| 百度实时热榜 | 7 |
| 央视新闻 | 7 |
| IT之家 | 7 |
| 澎湃新闻 | 5 |
| 腾讯新闻 | 3 |

真实数据 OCR 抽样验证采用 `minimal` 模式，即原图和 CLAHE 两个版本自动择优。该模式用于几十张截图的初筛，完整实验仍可使用 `full` 模式。

| 真实截图样本 | 最佳预处理版本 | 综合评分 | 平均置信度 |
|---|---|---:|---:|
| real_baidu_realtime_hotlist_01 | clahe | 0.9428 | 0.9663 |
| real_baidu_realtime_hotlist_02 | clahe | 0.9288 | 0.9476 |
| real_baidu_realtime_hotlist_03 | original | 0.9408 | 0.9571 |
| real_baidu_realtime_hotlist_04 | original | 0.9285 | 0.9432 |
| real_baidu_realtime_hotlist_05 | original | 0.9351 | 0.9565 |
| real_baidu_realtime_hotlist_06 | original | 0.9488 | 0.9744 |

真实网页截图文字密度较高，CPU 版 PaddleOCR 推理时间明显增加。因此项目中提供三种预处理规模：`full` 用于最终实验对比，`fast` 用于中等规模抽样，`minimal` 用于真实截图批量初筛。

## GPU 运行说明

代码已支持通过 `--device` 指定 PaddleOCR 推理设备：

```powershell
python run_ocr.py --image data/raw/sample_weibo_hotsearch.png --device cpu
python run_ocr.py --image data/raw/sample_weibo_hotsearch.png --device gpu:0
```

本机有 NVIDIA GeForce RTX 5060 Laptop GPU，安装 `paddlepaddle-gpu==3.3.1` 后已验证 PaddlePaddle 可在 `gpu:0` 上完成张量运算，说明驱动和 GPU 基础能力可用。但当前全局 Python 环境同时存在 PyTorch CUDA 12.8 与 Paddle CUDA 13.0，PaddleOCR/PaddleX 初始化 GPU 推理器时会触发 Windows CUDA/cuDNN DLL 冲突。因此最终交付版本默认使用 CPU，GPU 参数保留为可选能力。

若要稳定使用 GPU，建议单独创建干净环境，只安装 PaddleOCR 和 CUDA 版 PaddlePaddle：

```powershell
python -m venv .venv-paddle-gpu
.\.venv-paddle-gpu\Scripts\Activate.ps1
python -m pip install paddleocr
python -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/
python run_ocr.py --image data/raw/sample_weibo_hotsearch.png --device gpu:0 --variant-set minimal
```

## 输出目录

- `outputs/json/`：单图 JSON 结果。
- `outputs/llm_json/`：面向大模型的精简 JSON 结果。
- `outputs/visualizations/`：最佳 OCR 文本框可视化。
- `outputs/variants/`：每张图的不同预处理版本。
- `outputs/reports/batch_summary.json`：批处理汇总。
- `outputs/reports/llm_batch_summary.json`：面向大模型的批量精简汇总。

详细 JSON 保留坐标和多版本信息，主要用于调试、可视化和课程报告展示；大模型汇总时推荐只使用 `llm_json` 或 `llm_batch_summary.json`，避免坐标信息造成上下文浪费。
