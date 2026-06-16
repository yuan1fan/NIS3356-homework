# 数据分析模块

本目录是课程设计的初版数据分析模块。它通过文件接口整合微博代表帖正文、NLP 结果、OCR 文本和视觉语义摘要，生成统一 JSONL 数据集，并执行可复现的单批次热点统计分析。

模块不会调用爬虫、NLP、图像处理或 ASR 的内部函数，只读取这些模块已经生成的结果文件。

## 当前状态

已接入：

- 爬虫代表帖正文、热搜主题、互动指标和本地媒体路径；
- NLP 分类、情感、关键词、摘要等结果；
- 图片和视频的 OCR 文本；
- 图片和视频的视觉语义摘要。

暂未接入：

- ASR 语音转写。统一数据中已预留 `asr_texts` 和 `asr_count`，ASR 缺失
  不会阻塞当前流程。

## 环境要求

- Python 3.10 或更高版本；
- 当前构建和统计脚本只使用 Python 标准库；
- 不需要安装 BERT、BERTopic、Transformers 等额外依赖。

## 第一步：构建整合数据

在仓库的 `数据分析/` 目录执行：

```powershell
python build_integrated_dataset.py `
  --crawler-jsonl "<爬虫结果目录>\representative_posts.jsonl" `
  --ocr-json "..\图像处理\outputs_from_crawl\v2_latest_gpu_run\reports\llm_media_batch_summary.json" `
  --nlp-jsonl "..\NLP\输出\nlp_pipeline_results.jsonl" `
  --crawl-run-id "20260612_141245" `
  --output-dir "integrated_outputs"
```

尖括号中的爬虫结果路径需要按本机实际位置修改。脚本本身不包含任何
写死的本机盘符绝对路径。

输入文件：

| 参数 | 内容 |
| --- | --- |
| `--crawler-jsonl` | `representative_posts.jsonl`，作为 50 条父记录主表 |
| `--ocr-json` | 图像处理批量结果，包含 OCR 和 `visual_semantics` |
| `--nlp-jsonl` | NLP 流水线结果 |
| `--crawl-run-id` | 本批爬取任务编号，用于生成稳定父键 |
| `--output-dir` | 正式整合结果目录，默认 `integrated_outputs/` |

构建输出：

- `integrated_outputs/integrated_records.jsonl`：统一多模块数据集；
- `integrated_outputs/integration_build_report.md`：记录数和对齐校验报告。

每条统一记录主要包含：

- `parent_document_id`、`topic`、`topic_rank`、`post_id`；
- `raw_post_text`；
- `ocr_texts`；
- `visual_summaries`；
- `asr_texts`；
- `nlp_result`；
- `merged_text`；
- `available_modalities`、`missing_modalities` 和 `multimodal_score`；
- `fused_text`：按来源保存正文、OCR、视觉摘要、ASR 及融合文本；
- `safety_indicators`：用于 PRE 阶段解释性筛选的候选关注标记；
- `cross_modal_analysis`：OCR/视觉语义的补充价值、与正文的粗略重合度及
  一致性辅助分析；
- `media_info`、`metrics` 和 `quality`。

`merged_text` 按微博正文、OCR、视觉语义摘要、ASR 的顺序合并，并避免
重复加入完全相同的文本。该字段继续保留用于向后兼容；`fused_text` 在此
基础上进一步记录实际参与融合的文本来源。

`multimodal_score` 只表示正文、NLP、OCR、视觉语义和 ASR 的信息完整度，
不表示内容安全风险。`safety_indicators.needs_review` 仅表示建议重点关注，
不是最终安全判定。

## 第二步：运行初步分析

完成整合后执行：

```powershell
python analyze_integrated_dataset.py `
  --input "integrated_outputs\integrated_records.jsonl" `
  --output-dir "analysis_outputs"
```

分析输出：

- `analysis_outputs/data_analysis_report.md`：由脚本自动生成的完整分析报告；
- `analysis_outputs/category_summary.csv`：类别分布、平均互动量和代表热搜；
- `analysis_outputs/sentiment_summary.csv`：总体情感及类别下的情感分布；
- `analysis_outputs/media_engagement_summary.csv`：媒体数量与互动量比较；
- `analysis_outputs/multimodal_coverage_summary.csv`：OCR、视觉语义覆盖及
  文本增量；
- `analysis_outputs/analysis_metrics.json`：全部分析指标和趋势检测状态。
- `analysis_outputs/multimodal_fusion_report.md`：多模态完整度、融合来源和
  候选关注内容的课程展示报告；
- `analysis_outputs/analysis_report.html`：适合浏览和截图的静态分析看板；
- `analysis_outputs/record_browser.html`：内嵌全部统一记录的静态结果浏览器，
  可从左侧切换微博并查看多模态证据、融合结果和候选分析；
- `analysis_outputs/record_browser_tabs.html`：标签页版静态结果浏览器，
  用于课堂演示和 PPT 截图，包含总览、多模态证据、融合结果、跨模态分析
  和原始文本；
- `analysis_outputs/charts/`：模态覆盖率、完整度分布和候选关注内容统计图。
  其中还包括 `cross_modal_consistency_distribution.png`，用于展示
  `consistent`、`partial`、`weak`、`unknown` 的记录分布。

当前分析包括：

- 爬虫、OCR、视觉语义、NLP 和 ASR 的覆盖率；
- 热点类别、情感分布及类别和情感交叉统计；
- 点赞、评论、转发、加权互动量的描述统计；
- 不同类别和不同媒体分组的互动量比较；
- OCR 与视觉摘要对合并文本的增量统计；
- OCR/视觉语义是否补充正文信息，以及图像侧信息与正文的粗略一致性；
- 从数据中自动抽取的多模态代表样例。

所有报告结论均由 `analyze_integrated_dataset.py` 从
`integrated_records.jsonl` 自动计算。脚本不调用大模型 API，也不依赖人工
填写分析数字。

静态图表使用本地 `matplotlib` 生成。`analysis_report.html` 不使用
JavaScript、外部 CDN 或网络资源，图像通过相对路径引用
`analysis_outputs/charts/`。

结果浏览器可单独生成：

```powershell
python generate_record_browser.py
python generate_record_browser.py --layout tabs
```

结果浏览器页面使用内嵌原生 JavaScript 完成记录切换，不需要启动后端服务。

## 统一接口原则

数据分析模块只接收其他模块的输出文件，不依赖其他模块内部代码。推荐的
完整数据流见 `data_flow.md`。

后续接入 ASR 时，应至少提供可与爬虫媒体路径对应的
`source_video_path`、`transcript_text`，以及可选的 `segments`、
`language` 和置信度信息。

## 当前局限

- 当前只有单一批次、50 条微博代表帖；
- 每个热搜仅有一条代表帖，不能代表全部讨论；
- ASR 尚未接入，视频中的语音信息暂未进入统一文本；
- NLP 分类和情感结果尚未进行人工标注评估；
- 本批 50 条代表帖均包含媒体，因此有媒体/无媒体互动对比不具备区分意义，
  相关结果仅保留为程序接口验证；
- 互动量统计依赖爬虫返回字段。当前批次中部分互动字段数值较低或缺失，
  因此互动量相关分析仅作为初步参考；
- 当前统计和主题明细用于课程 Demo，不应视为正式舆情结论。

当前只进行单批次热点统计。脚本已预留 `analyze_trends(records)`：

- 只有一个 `crawl_run_id` 时输出 `insufficient_batches`；
- 趋势分析和预测至少需要多个 `crawl_run_id`；
- 后续可比较不同批次的类别占比、情感占比、互动量和主题变化。

## 本地归档

早期实验脚本和旧版结果保存在 `_archive_local/`。该目录已被
`数据分析/.gitignore` 排除，不应提交到 GitHub。
