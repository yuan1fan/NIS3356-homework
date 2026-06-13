# 网络媒体信息分析 — NLP 模块

面向微博/新闻等网络媒体文本的 NLP 分析系统，使用**传统方法 + BERT**混合架构。

## 项目结构

```
项目根目录/
├── NLP/                     # NLP 模块源码
│   ├── preprocessing.py          # 文本预处理
│   ├── keyword_extraction.py     # 关键词提取
│   ├── named_entity_recognition.py  # 命名实体识别
│   ├── sentiment_analysis.py     # 情感分析
│   ├── text_classification.py    # 文本分类
│   ├── event_extraction.py       # 事件抽取
│   ├── topic_discovery.py        # 主题发现
│   ├── text_summarization.py     # 自动摘要
│   ├── stopwords.txt             # 停用词表
│   ├── README.md                 # 子模块文档
│   └── 文本*.md (8份)           # 各模块说明
├── 数据/                   # 原始数据
│   └── 20260612_141245/
│       ├── representative_posts.jsonl  # 50 条代表性微博
│       ├── hot_topics_snapshot.jsonl   # 热搜榜截图
│       └── media/                      # 图片/视频媒体文件
├── 输出/                   # 流水线输出
│   ├── nlp_pipeline.py              # 全流水线脚本
│   ├── nlp_pipeline_results.jsonl   # 50 文档分析结果
│   ├── nlp_topics.json              # LDA 主题
│   └── nlp_pipeline_report.md       # 分析报告
├── _model_cache/            # BERT 模型缓存
│   └── models--ckiplab--bert-base-chinese-ner/  # BERT-NER (407MB)
├── gpt说明.md               # 项目方案说明书
└── README.md                # 本文档
```

## 运行环境

### 环境 A：沙箱 Python（默认）

用于基本 NLP 流水线（jieba 分词 + 规则方法）。

| 项目 | 值 |
|------|-----|
| Python | `python3` (Windows Store 3.11.9) |
| jieba | 0.42.1 |
| opencc | opencc-python-reimplemented |

```bash
pip install jieba opencc-python-reimplemented
python3 输出/nlp_pipeline.py
```

### 环境 B：Conda Python（BERT-NER ）

用于 BERT-NER + GPU 加速。

| 项目 | 值 |
|------|-----|
| Python | Conda 3.12 |
| torch | 2.8.0 + CUDA |
| transformers | 5.10.2 |
| huggingface_hub | 内置 |

### 模型缓存

BERT-NER 模型 `ckiplab/bert-base-chinese-ner` (~407MB) 已缓存在：

```
_model_cache/models--ckiplab--bert-base-chinese-ner/
└── snapshots/{hash}/
    ├── config.json
    ├── pytorch_model.bin
    ├── vocab.txt
    └── tokenizer_config.json
```

NER 模块启动时自动检测缓存；若不存在则降级到 jieba。

## 模块依赖

| 模块 | 依赖 | 外部包 |
|------|------|--------|
| preprocessing | jieba, opencc | 标准库 + jieba + opencc |
| keyword_extraction | preprocessing | TF-IDF / TextRank 从零实现 |
| named_entity_recognition | preprocessing, jieba.posseg | 可选: torch + transformers |
| sentiment_analysis | preprocessing | 词典 + 规则 |
| text_classification | preprocessing | 朴素贝叶斯 / 逻辑回归 从零实现 |
| event_extraction | named_entity_recognition | 触发词 + 规则 |
| topic_discovery | preprocessing | LDA Gibbs Sampling / K-means 从零实现 |
| text_summarization | preprocessing | TextRank / TF-IDF 从零实现 |

全部 8 个模块的核心算法均从零实现（不依赖 sklearn / gensim / spaCy 等外部 ML 库）。

## NLP 流水线

### 工作流

```
原始文本
  ↓ 预处理 (去噪 → 分词 → 去停用词)
  ├──→ 关键词提取 (TF-IDF / TextRank)
  ├──→ 命名实体识别 (BERT / jieba.posseg)
  ├──→ 情感分析 (词典 + 规则)
  ├──→ 事件抽取 (36 类触发词 + NER 槽位)
  ├──→ 文本分类 (朴素贝叶斯)
  ├──→ 主题发现 (LDA)
  └──→ 自动摘要 (TextRank)
  ↓
结构化 JSON 输出
```

### 执行

```bash
# 环境 A：jieba 模式
python3 输出/nlp_pipeline.py

# 环境 B：BERT-NER + GPU
E:\conda\conda\python.exe 输出/nlp_pipeline.py
```

### 输出文件

| 文件 | 格式 | 说明 |
|------|------|------|
| nlp_pipeline_results.jsonl | JSONL | 50 条 × 8 模块分析结果 |
| nlp_topics.json | JSON | LDA 8 主题 × Top-10 词 |
| nlp_pipeline_report.md | Markdown | 语料级统计报告 |

## 8 个子模块

| # | 模块 | 方法 |
|---|------|------|
| 1 | 文本预处理 | 14 步清洗流水线: HTML → URL → @ → # → emoji → 符号 → 繁转简 → 分词 → 停用词 |
| 2 | 关键词提取 | TF-IDF（语料级）+ TextRank（单文档） |
| 3 | 命名实体识别 | **BERT** (ckiplab/bert-base-chinese-ner, GPU) / **jieba.posseg** (HMM, CPU) 自动降级 |
| 4 | 情感分析 | 词典(330词) + 规则(否定/程度/转折) + 表情符号 |
| 5 | 文本分类 | 朴素贝叶斯 Multinomial NB / 逻辑回归 LR (一对其余) |
| 6 | 事件抽取 | 36 类事件, 286 触发词, NER 槽位填充 |
| 7 | 主题发现 | LDA (Collapsed Gibbs Sampling) / K-means (Cosine 距离) |
| 8 | 自动摘要 | TextRank 句子图排序 / TF-IDF 句子评分 + 多样性惩罚 |


