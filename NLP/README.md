# NLP 文本智能分析平台

> 传统规则 + 统计方法 + LLM

---

## 目录

1. [项目简介](#1-项目简介)
2. [系统架构](#2-系统架构)
3. [创新点说明](#3-创新点说明)
4. [环境安装](#4-环境安装)
5. [一键启动](#5-一键启动)
    - [5.1 批处理流水线](#51-批处理流水线)
    - [5.2 Web 可视化界面](#52-web-可视化界面)
6. [8 个子模块](#6-8-个子模块)
7. [输出格式](#7-输出格式)
8. [传统模式 vs LLM 模式](#8-传统模式-vs-llm-模式)
9. [项目结构](#9-项目结构)
10. [常见问题 FAQ](#10-常见问题-faq)

---

## 1. 项目简介

### 1.1 是什么

**NLP 文本智能分析平台**（NLP Text Intelligence Analysis Platform）是一个面向微博/新闻等网络媒体文本的多维度 NLP 分析系统，采用**传统规则 + 统计方法**与**大语言模型**双引擎架构，支持 8 个 NLP 子模块的独立或批量分析，并提供 Web 可视化界面。

### 1.2 解决什么问题

- **海量网络文本理解**：每天产生的微博、新闻、评论等文本需要自动化分析
- **热点事件检测**：从文本中提取事件要素（时间、地点、主体、产品）
- **舆情情感追踪**：自动判断文本正/负/中性情感倾向及细粒度情绪
- **实体信息抽取**：识别人名、地名、机构名、时间、金额等关键实体
- **主题发现与分类**：自动将文本归类并发现潜在主题
- **传统算法与 LLM 对比验证**：同一套接口下对比规则方法与大模型效果

### 1.3 工作流程

```
原始文本（微博/新闻/评论）
  ↓
[预处理] 去噪 → 繁转简 → jieba 分词 → 去停用词
  ↓
  ├── 关键词提取 (TF-IDF / TextRank)
  ├── 命名实体识别 (BERT / jieba.posseg 自动降级)
  ├── 情感分析 (词典 + 规则 / DeepSeek)
  ├── 事件抽取 (36 类触发词 + NER 槽位)
  ├── 文本分类 (朴素贝叶斯 / 逻辑回归)
  ├── 主题发现 (LDA Gibbs Sampling / K-means)
  └── 自动摘要 (TextRank 句子排序)
  ↓
结构化 JSON 输出 / 可视化 Web UI
```

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                   Web UI (Gradio)                        │
│  ┌──────────────────────────────────────────────────┐   │
│  │  传统模式（规则/统计）  │  LLM 模式 (DeepSeek)     │   │
│  └──────────┬────────────────────┬──────────────────┘   │
└─────────────┼────────────────────┼──────────────────────┘
              │                    │
┌─────────────▼────────────────────▼──────────────────────┐
│                    NLP 核心引擎                          │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ │
│  │预处理│ │关键词│ │  NER │ │情感  │ │事件  │ │分类  │ │
│  │      │ │提取  │ │识别  │ │分析  │ │抽取  │ │      │ │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ │
│  ┌──────┐ ┌──────┐                                      │
│  │主题  │ │摘要  │                                      │
│  │发现  │ │生成  │                                      │
│  └──────┘ └──────┘                                      │
└──────────────────────────────────────────────────────────┘
```

| 组件 | 传统模式 | LLM 模式 |
|------|---------|---------|
| 关键词提取 | TF-IDF / TextRank 从零实现 | DeepSeek 语义提取 |
| 命名实体识别 | BERT (ckiplab) / jieba.posseg | DeepSeek 语义识别 |
| 情感分析 | 词典(330词) + 否定/程度/转折规则 | DeepSeek 情感判断 |
| 事件抽取 | 36 类事件、286 触发词 + NER 槽位 | DeepSeek 语义抽取 |
| 文本分类 | 朴素贝叶斯 / 逻辑回归 | DeepSeek 分类 |
| 主题发现 | LDA (Collapsed Gibbs) / K-means | 不支持 |
| 自动摘要 | TextRank 句子图排序 | DeepSeek 生成式摘要 |

---

## 3. 创新点说明

### 3.1 如何契合"多模态融合与信息内容安全"课程要求

| 课程要求 | 本项目实现 |
|---------|-----------|
| 多模态信息融合 | 文本 NLP + 图像 OCR 跨模态数据交叉验证 |
| 信息内容安全检测 | 命名实体识别 + 事件抽取联合发现安全事件 |
| 文本处理与分析 | 8 个独立 NLP 模块，从零实现核心算法 |
| 传统算法与深度学习对比 | 同一接口下规则方法 ↔ BERT ↔ DeepSeek LLM 三路对比 |
| 工程实现 | Gradio 可视化 UI + 批处理流水线 + 一键部署 |

### 3.2 核心创新

1. **三引擎混合架构**：规则方法（离线、快速、可解释）→ BERT（高精度 NER）→ DeepSeek LLM（语义理解），按场景自动选择或对比
2. **全部核心算法从零实现**：TF-IDF、TextRank、LDA Gibbs Sampling、朴素贝叶斯等均不依赖 sklearn / gensim，完全手写
3. **BERT 自动降级**：NER 模块启动时自动检测模型缓存；未检测到 BERT 模型则无缝降级到 jieba.posseg，零配置切换
4. **LLM 懒加载**：transformers 库在模块首次真正需要时才导入，避免 import 时卡死（针对 Windows 下 transformers 目录扫描问题）
5. **36 类事件触发词系统**：覆盖产品发布/召回、处罚整治、诉讼维权、事故灾害、社会治安等网络媒体核心事件类型

---

## 4. 环境安装

### 4.1 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| Python | ≥ 3.9 | 3.10–3.12 |
| 内存 | 2 GB | 8 GB+（BERT 模式需要）|
| 磁盘 | 500 MB | 5 GB（含 BERT 模型）|
| GPU | 可选 | NVIDIA GPU + CUDA（BERT 加速）|

### 4.2 安装步骤

#### 基础环境（传统模式，仅 jieba）

```bash
# 进入项目目录
cd /path/to/project

# 安装依赖
pip install jieba opencc-python-reimplemented gradio

# 验证安装
python -c "import jieba; print('jieba OK')"
```

#### Conda 环境（BERT-NER + GPU 加速）

```bash
# 创建 conda 环境（推荐 Python 3.10+）
conda create -n nlp python=3.10
conda activate nlp

# 安装 GPU 依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers huggingface_hub

# 验证
python -c "from transformers import AutoModelForTokenClassification; print('transformers OK')"
```

#### DeepSeek LLM 模式

```bash
# 设置环境变量
set DEEPSEEK_API_KEY=sk-your-key-here

# 验证
python -c "import sys; sys.path.insert(0, 'NLP LLM'); from deepseek_nlp import DeepSeekNLP; print('DeepSeek OK')"
```

### 4.3 BERT 模型缓存（可选）

预训练 NER 模型 `ckiplab/bert-base-chinese-ner`（~407MB）已缓存在 `_model_cache/` 目录：

```
_model_cache/models--ckiplab--bert-base-chinese-ner/
└── snapshots/{hash}/
    ├── config.json
    ├── pytorch_model.bin
    ├── vocab.txt
    └── tokenizer_config.json
```

如目录不存在，NER 模块会自动降级到 jieba.posseg 模式。

---

## 5. 一键启动

### 5.1 批处理流水线

```bash
# 传统模式：50 条微博完整流水线
cd /path/to/project
python "输出/nlp_pipeline.py"

# LLM 模式（需配置 API Key）
set DEEPSEEK_API_KEY=sk-xxx
python "LLM 输出/nlp_pipeline_llm.py"
```

输出文件：

| 文件 | 格式 | 说明 |
|------|------|------|
| `输出/nlp_pipeline_results.jsonl` | JSONL | 50 条 × 8 模块分析结果 |
| `输出/nlp_topics.json` | JSON | LDA 8 主题 × Top-10 词 |
| `输出/nlp_pipeline_report.md` | Markdown | 语料级统计报告 |
| `LLM 输出/nlp_pipeline_results1.jsonl` | JSONL | 50 条 × 6 模块 LLM 分析结果 |
| `LLM 输出/nlp_pipeline_report.md` | Markdown | LLM 语料级统计报告 |

### 5.2 Web 可视化界面

```bash
# 启动 Web UI（默认 7865 端口）
cd UI/NLP
python app.py --port 7865

# 打开浏览器访问 http://127.0.0.1:7865
```

#### Web UI 功能

| 功能 | 说明 |
|------|------|
| 传统模式 | 使用 NLP/ 目录下的 8 个规则/统计 NLP 模块 |
| LLM 模式 | 使用 DeepSeek API 进行智能 NLP 分析 |
| 单文本分析 | 输入单条文本，选择需要运行的模块 |
| 模块选择 | 自由勾选 8 个子模块中的任意组合 |
| 关键词方法切换 | TextRank（单文档） / TF-IDF（需语料）|
| 结果概览 | 总览标签页展示各模块核心结果 |
| 详细结果 | 每个模块独立标签页展示完整分析 |
| 报告导出 | 导出完整分析报告（.txt）|

#### 示例文本

```text
2026年6月12日，公安部召开专题新闻发布会。会上，北京市公安局刑侦总队政治处主任
李小燕公布了在北京发生的两起典型案例。其中一起是400余名老年人遭健康养生诈骗。
北京警方近期打掉一个专门针对老年人的诈骗团伙，抓获31名犯罪嫌疑人，
涉及朝阳、顺义、平谷、密云4个区20家门店。
```

---

## 6. 8 个子模块

| # | 模块 | 文件 | 方法 | 依赖 |
|---|------|------|------|------|
| 1 | **文本预处理** | `preprocessing.py` | 14 步清洗流水线：HTML → URL → @ → # → emoji → 符号 → 繁转简 → 分词 → 去停用词 | jieba, opencc |
| 2 | **关键词提取** | `keyword_extraction.py` | TF-IDF（语料级，从零实现）+ TextRank（单文档，从零实现） | preprocessing |
| 3 | **命名实体识别** | `named_entity_recognition.py` | BERT (ckiplab, 自动降级到 jieba.posseg) + 内置词典（百家姓/省市/机构后缀/产品后缀） | jieba; 可选 torch+transformers |
| 4 | **情感分析** | `sentiment_analysis.py` | 词典(330 个正负面词) + 否定/程度副词/转折规则 + 表情符号 + 细粒度情绪 | preprocessing |
| 5 | **文本分类** | `text_classification.py` | 朴素贝叶斯 Multinomial NB / 逻辑回归 LR（一对其余），均从零实现 | preprocessing |
| 6 | **事件抽取** | `event_extraction.py` | 36 类事件、286 个触发词 + NER 槽位填充（时间/地点/主体/产品）+ 句式匹配 | NER 模块 |
| 7 | **主题发现** | `topic_discovery.py` | LDA (Collapsed Gibbs Sampling) / K-means (Cosine 距离)，均从零实现 | preprocessing |
| 8 | **自动摘要** | `text_summarization.py` | TextRank 句子图排序 / TF-IDF 句子评分 + 多样性惩罚，从零实现 | preprocessing |

> 全部 8 个模块的核心算法均从零实现，不依赖 sklearn / gensim / spaCy 等外部 ML 库。

---

## 7. 输出格式

### JSONL（批处理流水线）

```json
{
  "id": 0,
  "topic": "公安部新闻发布会",
  "original": "2026年6月12日，公安部召开专题新闻发布会...",
  "preprocessed": "2026年 6月 12日 公安部 召开 专题 新闻发布会 ...",
  "keywords": [{"word": "诈骗", "score": 0.85}, {"word": "老年人", "score": 0.72}, ...],
  "entities": [
    {"text": "公安部", "type": "机构名", "start": 10, "end": 13},
    {"text": "李小燕", "type": "人名", "start": 26, "end": 29},
    {"text": "北京", "type": "地名", "start": 34, "end": 36}
  ],
  "sentiment": {"sentiment": "负面", "score": -0.65},
  "events": [{"事件类型": "处罚整治", "主体": "北京警方", "对象": "诈骗团伙"}],
  "classification": {"label": "社会", "probabilities": {"社会": 0.92, "生活": 0.05, ...}},
  "topic_distribution": [{"topic_id": 2, "words": ["诈骗", "老年人", "警方"], "probability": 0.87}],
  "summary_sentences": ["北京警方打掉一个专门针对老年人的诈骗团伙。"]
}
```

---

## 8. 传统模式 vs LLM 模式

| 维度 | 传统模式 (NLP/) | LLM 模式 (NLP LLM/) |
|------|----------------|-------------------|
| 运行命令 | `python 输出/nlp_pipeline.py` | `python "LLM 输出/nlp_pipeline_llm.py"` |
| Web UI | 传统模式标签页 | LLM 模式标签页 |
| 依赖 | jieba + 本地规则 | DeepSeek API Key + 网络 |
| 速度 | ~16 秒 / 50 篇 | ~5 分钟 / 50 篇（含 API 间隔） |
| 精度 | 规则定义、可控 | 语义理解、灵活 |
| 成本 | 免费、离线 | API 调用计费、需网络 |
| 覆盖 | 8 个独立模块 | 6 个模块（不支持预处理和主题发现） |
| 接口兼容 | 各模块独立调用 | `nlp.analyze(text)` 一次调用 |

---

## 9. 项目结构

```
项目根目录/
│
├── NLP/                          # 传统 NLP 模块源码（8 个模块）
│   ├── preprocessing.py               # 文本预处理（jieba 分词）
│   ├── keyword_extraction.py          # 关键词提取（TF-IDF / TextRank）
│   ├── named_entity_recognition.py    # 命名实体识别（BERT / jieba）
│   ├── sentiment_analysis.py          # 情感分析（词典 + 规则）
│   ├── text_classification.py         # 文本分类（朴素贝叶斯 / LR）
│   ├── event_extraction.py            # 事件抽取（36 类触发词）
│   ├── topic_discovery.py             # 主题发现（LDA / K-means）
│   ├── text_summarization.py          # 自动摘要（TextRank）
│   ├── stopwords.txt                  # 停用词表
│   └── 文本*模块说明.md (8份)        # 各模块详细说明
│
├── NLP LLM/                      # DeepSeek API 驱动模块
│   ├── deepseek_nlp.py                # DeepSeekNLP 类（ner/sentiment/keyword...）
│   └── README.md                      # LLM 模块说明
│
├── UI/
│   └── NLP/
│       ├── app.py                     # Gradio Web 界面（传统 + LLM 双模式）
│       ├── requirements.txt           # UI 依赖
│       └── run.bat                    # 一键启动脚本
│
├── 输出/                          # 传统方法批处理输出
│   ├── nlp_pipeline.py                # 全流水线脚本
│   ├── nlp_pipeline_results.jsonl     # 50 条文档 × 8 模块分析结果
│   ├── nlp_topics.json                # LDA 主题模型
│   └── nlp_pipeline_report.md         # 语料级统计报告
│
├── LLM 输出/                      # LLM 方法批处理输出
│   ├── nlp_pipeline_llm.py            # LLM 流水线脚本
│   ├── nlp_pipeline_results1.jsonl    # 50 条文档 × 6 模块分析结果
│   ├── nlp_pipeline_report.md         # LLM 语料级统计报告
│   └── README.md                      # LLM 输出说明
│
├── 数据/                          # 原始微博数据
│   └── 20260612_141245/
│       ├── representative_posts.jsonl # 50 条代表性微博
│       └── hot_topics_snapshot.jsonl  # 热搜榜快照
│
├── _model_cache/                  # BERT 模型缓存
│   └── models--ckiplab--bert-base-chinese-ner/
│       └── snapshots/{hash}/
│           ├── config.json
│           ├── pytorch_model.bin
│           └── vocab.txt
│
├── 图像处理/                       # 图像 OCR 模块（PaddleOCR）
│   └── ...（详见 图像处理/README.md）
│
├── gpt说明.md                      # 项目方案说明书
└── README.md                       # 本文档
```

---

## 10. 常见问题 FAQ

### Q1: 启动报错 `ModuleNotFoundError: No module named 'preprocessing'`

**原因**：`preprocessing.py` 不在 Python 路径中。
```bash
# 确保从项目根目录运行，或设置 PYTHONPATH
set PYTHONPATH=E:\path\to\project\NLP
```

### Q2: NER 模块导入时卡死 / KeyboardInterrupt

**原因**：Windows 下 transformers 库初始化时扫描所有模型目录。

**解决**：已修复。当前版本将 transformers 导入改为懒加载，只在首次真正调用 BERT 模型时才导入。如果仍然卡住，检查 `_model_cache/` 目录是否损坏。

### Q3: 启动报错 `No module named 'jieba'`

```bash
pip install jieba
```

### Q4: BERT-NER 没有效果

**原因**：`_model_cache/` 中未检测到 BERT 模型。

**检查**：
```bash
# 查看缓存目录
dir _model_cache\models--ckiplab--bert-base-chinese-ner\snapshots\
```
如果目录为空，NER 模块会自动降级到 jieba.posseg。如需 BERT 模式，手动下载模型到上述目录。

### Q5: LLM 模式返回空白结果

**原因**：未配置 DeepSeek API Key，或 API 调用失败。

**解决**：在 Web UI 中切换至 LLM 模式后，输入 API Key 或设置环境变量：
```bash
set DEEPSEEK_API_KEY=sk-xxx
```

### Q6: Web UI 中模块选择不是两列显示

**原因**：CSS 异常。

**解决**：已修复。确保 app.py 中 `.module-grid` 包含 `display: grid` 属性。

### Q7: Gradio 界面启动报端口占用

```bash
# 指定其他端口
python app.py --port 7866
```

### Q8: 批处理流水线乱码

```bash
# 确保控制台支持 UTF-8
chcp 65001
python 输出/nlp_pipeline.py
```

### Q9: 如何只用 TF-IDF 模式？

```bash
# 命令行批处理：直接设置 keyword_method
# Web UI：在高级设置中选择 "TF-IDF(需语料)"
```
首次运行流水线时会自动加载语料训练 TF-IDF 模型。Web UI 中单文档模式默认使用 TextRank。

### Q10: 如何对比传统方法与 LLM 的差异？

```bash
# 1. 运行传统模式流水线
python 输出/nlp_pipeline.py

# 2. 运行 LLM 模式流水线（需 API Key）
python "LLM 输出/nlp_pipeline_llm.py"

# 3. 对比两个输出目录中的结果和报告
# 输出/nlp_pipeline_report.md       → 传统方法
# LLM 输出/nlp_pipeline_report.md   → LLM 方法
```
