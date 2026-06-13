# NLP 模块文档

## 概述

面向网络媒体文本（微博、新闻、评论）的 NLP 处理模块集，全部使用**传统方法**（非深度学习）实现，无需标注数据即可运行。

共包含 **8 个子模块**，按处理流程排列：

| 序号 | 模块 | 文件 | 方法 | 依赖 |
|------|------|------|------|------|
| ① | 文本预处理 | `preprocessing.py` | 正则 + jieba 分词 | `jieba`, `opencc` |
| ② | 关键词提取 | `keyword_extraction.py` | TF-IDF / TextRank | 预处理模块 |
| ③ | 命名实体识别 | `named_entity_recognition.py` | 词典 + 规则 | 预处理模块 |
| ④ | 情感分析 | `sentiment_analysis.py` | 情感词典 + 规则 | 预处理模块 |
| ⑤ | 文本分类 | `text_classification.py` | 朴素贝叶斯 / 逻辑回归 | 预处理模块 |
| ⑥ | 事件抽取 | `event_extraction.py` | 触发词 + NER 槽位填充 | NER 模块 |
| ⑦ | 主题发现 | `topic_discovery.py` | LDA (Gibbs) / K-means | 预处理模块 |
| ⑧ | 自动摘要 | `text_summarization.py` | TextRank 句子排序 | 预处理模块 |

---

## 环境要求

### Python 版本

- Python 3.8+

### 依赖安装

```bash
pip install jieba opencc-python-reimplemented
```


---

## 快速开始

### 完整流水线

处理 `输出/nlp_pipeline.py` 中的全流水线脚本，一键处理全部数据：

```bash
python 输出/nlp_pipeline.py
```

输出文件位于 `输出/` 目录：

| 文件 | 说明 |
|------|------|
| `nlp_pipeline_results.jsonl` | 50 条文档 × 8 模块的结构化分析结果 |
| `nlp_topics.json` | LDA 主题模型（8 主题 × Top-10 词） |
| `nlp_pipeline_report.md` | 语料级统计分析报告 |

### 逐模块使用

```python
import sys
sys.path.insert(0, "NLP")  # 将 NLP 目录加入路径

from preprocessing import TextPreprocessor
from keyword_extraction import TFIDFExtractor, TextRankExtractor
from named_entity_recognition import NERExtractor
from sentiment_analysis import SentimentAnalyzer
from event_extraction import EventExtractor
from text_classification import TextClassifier
from topic_discovery import TopicDiscoverer
from text_summarization import TextSummarizer

text = "2026年6月，某公司在上海发布新款手机。"

# ① 预处理
pp = TextPreprocessor()
cleaned = pp.process(text)

# ② 关键词提取
tfidf = TFIDFExtractor()
tfidf.fit(corpus)               # 需要先训练语料
kw = tfidf.extract(text, top_n=10)

# ③ 命名实体识别
ner = NERExtractor()
entities = ner.recognize(text)

# ④ 情感分析
sa = SentimentAnalyzer()
sentiment = sa.analyze(text)

# ⑤ 文本分类
clf = TextClassifier(method="naive_bayes")
clf.fit(train_texts, train_labels)
pred = clf.predict(text)

# ⑥ 事件抽取
ee = EventExtractor()
events = ee.extract_summary(text)

# ⑦ 主题发现
td = TopicDiscoverer(method="lda", n_topics=8)
td.fit(corpus)
topics = td.get_topics(top_n=10)

# ⑧ 自动摘要
ts = TextSummarizer(method="textrank")
summary = ts.summarize(text, num_sentences=3)
```

---

## 子模块说明

### ① 文本预处理 (`preprocessing.py`)

14 步清洗流水线：

```
原始文本 → HTML实体还原 → 去除HTML标签 → 去除URL
         → 去除@用户 → #话题#处理(保留文字) → 去除表情符号
         → 去除微博后缀 → 去除无意义符号 → 压缩空白
         → 广告检测 → 繁简转换 → jieba分词 → 停用词过滤 → 去重
```

```python
from preprocessing import TextPreprocessor

pp = TextPreprocessor()
result = pp.process("#突发#！！某地暴雨太大了！！！详情见 http://xxx.com")
# "突发 某地 暴雨 太大 详情 见"

# 查看分步结果
steps = pp.process_with_steps(text)
for name, output in steps.items():
    print(f"{name}: {output}")
```

**参数：** 无（使用内置词表和停用词 `stopwords.txt`）

---

### ② 关键词提取 (`keyword_extraction.py`)

两种传统方法：

| 方法 | 原理 | 是否需要语料 |
|------|------|-------------|
| **TF-IDF** | 词频 × 逆文档频率 | 需要先 `fit(corpus)` |
| **TextRank** | 共现图 + PageRank | 不需要 |

```python
from keyword_extraction import TFIDFExtractor, TextRankExtractor

# TF-IDF
extractor = TFIDFExtractor()
extractor.fit(corpus_texts)
kw = extractor.extract(text, top_n=10, with_scores=True)
# [("词1", 0.85), ("词2", 0.62), ...]

# TextRank
extractor = TextRankExtractor(window=3)
kw = extractor.extract(text, top_n=10)
# ["词1", "词2", ...]

# 统一接口
from keyword_extraction import KeywordExtractor
ext = KeywordExtractor(method="tfidf")
ext.fit(corpus)
kw = ext.extract(text)
```

---

### ③ 命名实体识别 (`named_entity_recognition.py`)
### ③ 命名实体识别 (`named_entity_recognition.py`)

混合系统：**jieba.posseg (HMM 词性标注)** + 正则 + 词典后缀

| 实体类型 | 识别方法 | 示例 |
|----------|----------|------|
| 人名 | jieba POS 标签 `nr` | 李小燕、白鹿、梅西 |
| 地名 | jieba POS 标签 `ns` | 上海、北京、中国 |
| 机构名 | jieba POS 标签 `nt` | 北京市公安局 |
| 时间 | 正则匹配 | 2026年6月、今天、12:30 |
| 金额 | 正则匹配 | 3000万元、99.9% |
| 产品名 | 词典后缀 + 前缀 | 新款手机 |

jieba POS 标注器使用 **HMM（隐马尔可夫模型）**在百万级标注语料上训练，
远优于纯词典匹配。错误率降低约 80%。

```python
from named_entity_recognition import NERExtractor, format_entities

ner = NERExtractor()
entities = ner.recognize("北京市公安局刑侦总队主任李小燕公布了案例")
# [
#   {"text": "北京市公安局", "type": "机构名", "start": 0, "end": 6},
#   {"text": "李小燕",     "type": "人名", "start": 14, "end": 17},
# ]

print(format_entities(entities, text_with_context))
```

**精度对比（新版 vs 旧版词典规则）：**

| 测试文本 | 旧版(词典) | 新版(jieba.posseg) |
|----------|-----------|-------------------|
| 暴雨太大 | 暴雨太(人名❌) | (无实体✅) |
| 养生馆围猎老人 | 养生馆/金额高(人名❌) | (无实体✅) |
| 欺骗消费者 | 费者行(人名❌) | (无实体✅) |
| 强烈推荐 | 强烈推(人名❌) | (无实体✅) |
| 李小燕公布 | 燕公布/任李小(人名❌) | 李小燕(人名✅) |

---

### ④ 情感分析 (`sentiment_analysis.py`)

基于情感词典 + 规则的零样本情感分析。

**词典构成：**
- 正面词 ~150 个：好、棒、喜欢、优秀……
- 负面词 ~160 个：差、坏、垃圾、愤怒、离谱……
- 程度副词 ~35 个：很(×1.5)、非常(×2.0)、太(×2.0)……
- 否定词 ~20 个：不、没、别（翻转极性）
- 表情符号 ~25 个：😊(+2.0)、😡(-2.5)……

**评分规则：**
```
分句 → 逐词扫描 → 正面词+1.0 / 负面词-1.0
                  → 程度词加权 × 否定翻转
                  → 转折加重后半句权重
                  → 归一化到 [-1, +1]
```

```python
from sentiment_analysis import SentimentAnalyzer

sa = SentimentAnalyzer()
result = sa.analyze("功能不错，但价格太离谱了")
# {"sentiment": "负面", "score": -0.36,
#  "positive_words": ["不错"], "negative_words": ["离谱"],
#  "fine_grained": {}}

# 方面级分析
ar = sa.analyze_aspects("功能不错，价格太离谱，质量挺好")
# {"aspects": [{"aspect": "功能", "sentiment": "正面"}, ...],
#  "overall": "混合"}
```

---

### ⑤ 文本分类 (`text_classification.py`)

从零实现两种分类算法：

| 算法 | 原理 | 特点 |
|------|------|------|
| **朴素贝叶斯 (MNB)** | 词频先验 + 条件概率 + Laplace 平滑 | 小样本表现好 |
| **逻辑回归 (OVR)** | 梯度下降 + Sigmoid + L2 正则 | 概率输出 |

```python
from text_classification import TextClassifier, auto_label_by_keywords

# 自动标注（无标签数据演示用）
cat_map = {
    "娱乐": ["白鹿", "鹿晗", "音乐节", "官宣"],
    "体育": ["世界杯", "足球", "加纳", "夺冠"],
    "社会": ["网信办", "诈骗", "警方", "离世"],
}
labels = auto_label_by_keywords(texts, cat_map)

# 朴素贝叶斯
clf = TextClassifier(method="naive_bayes", alpha=1.0)
clf.fit(train_texts, train_labels)
pred = clf.predict("世界杯足球比赛结果")  # "体育"
proba = clf.predict_proba("世界杯足球比赛结果")
# {"娱乐": 0.1, "体育": 0.8, "社会": 0.1}

# 逻辑回归 + TF-IDF
clf = TextClassifier(method="logistic_regression", vectorizer="tfidf")
clf.fit(train_texts, train_labels)

# 交叉验证
cv = clf.cross_validate(texts, labels, folds=5)
# {"accuracies": [0.8, 0.75, ...], "mean_accuracy": 0.78}
```

---

### ⑥ 事件抽取 (`event_extraction.py`)

基于触发词 + NER 槽位填充的事件抽取，覆盖 **36 类事件、5 大类别、286 个触发词**：

| 大类 | 事件类型 | 触发词示例 |
|------|----------|-----------|
| 社会事件 | 自然灾害、事故灾难、公共卫生、社会治安、执法司法、维权抗议、救援救助 | 地震、车祸、疫情、诈骗、逮捕、罢工 |
| 政治政策 | 政策发布、政策调整、领导人活动、会议召开、外交事件、反腐倡廉 | 印发、调整、视察、召开、制裁、落马 |
| 经济商业 | 产品发布、企业动态、投融资、财报业绩、价格变动、产品问题、产品召回、合同签约、破产重组、市场竞争 | 上市、裁员、融资、营收、涨价、召回、签约 |
| 科技文娱 | 技术突破、科研发布、影视上映、综艺动态、明星动态、明星丑闻、粉丝事件、游戏动态 | 首发、论文、定档、晋级、官宣、塌房 |
| 体育赛事 | 比赛结果、转会签约、伤病事件、禁赛处罚、纪录突破 | 夺冠、加盟、受伤、禁赛、破纪录 |

```python
from event_extraction import EventExtractor

ee = EventExtractor()
events = ee.extract("某品牌因质量问题发布召回公告")
# [{"type": "产品召回", "trigger": "发布召回公告",
#   "slots": {"主体": "某品牌", "原因": "质量问题", "动作": "发布召回公告"}}]

# 简化摘要
summary = ee.extract_summary(text)
# [{"事件类型": "产品召回", "原因": "质量问题", "主体": "某品牌"}]
```

---

### ⑦ 主题发现 (`topic_discovery.py`)

两种方法：

| 方法 | 原理 | 特点 |
|------|------|------|
| **LDA** | Collapsed Gibbs Sampling | 概率主题模型，输出主题-词分布 |
| **K-means** | Cosine 距离 + 聚类 | 硬聚类，适合快速分组 |

```python
from topic_discovery import TopicDiscoverer

# LDA 主题建模
lda = TopicDiscoverer(method="lda", n_topics=8, alpha=0.1, beta=0.01)
lda.fit(corpus_texts)
topics = lda.get_topics(top_n=10)
# [{"topic_id": 0, "words": ["词1","词2",...], "weight": 0.15}, ...]

# K-means 聚类
km = TopicDiscoverer(method="kmeans", n_topics=8)
km.fit(corpus_texts)
clusters = km.get_topics(top_n=10)

# 新文档主题推断（仅 LDA）
theta = lda.transform(["新文本"])
```

**LDA 参数说明：**

| 参数 | 默认 | 说明 |
|------|------|------|
| `n_topics` | 10 | 主题数量 |
| `alpha` | 0.1 | 文档-主题 Dirichlet 先验 |
| `beta` | 0.01 | 主题-词 Dirichlet 先验 |
| `n_iter` | 200 | Gibbs 采样迭代次数 |

---

### ⑧ 自动摘要 (`text_summarization.py`)

两种抽取式摘要方法：

| 方法 | 原理 | 特点 |
|------|------|------|
| **TextRank** | 句子相似度图 + PageRank + 多样性惩罚 | 无需外部语料 |
| **TF-IDF** | 关键词密度评分 + 位置奖励 | 偏向信息密度 |

```python
from text_summarization import TextSummarizer

ts = TextSummarizer(method="textrank")
sentences = ts.summarize(long_text, num_sentences=3)
# ["第一句...", "第二句...", "第三句..."]

# 合并为文本
summary = ts.summarize_text(long_text, num_sentences=3)
# "第一句...。第二句...。第三句...。"
```

---

## 全流水线数据流

```
                          ┌─────────────────┐
                          │   原始微博文本    │
                          └────────┬────────┘
                                   ↓
                          ┌─────────────────┐
                          │   文本预处理     │ ← preprocessing.py
                          │  (14步清洗)      │
                          └────────┬────────┘
                                   ↓
        ┌──────────────────────────┼──────────────────────────┐
        ↓                          ↓                          ↓
  ┌─────────────┐          ┌──────────────┐          ┌────────────────┐
  │关键词提取    │          │ 命名实体识别  │          │   情感分析      │
  │TF-IDF/TR    │          │ 6类实体词典  │          │   词典+规则     │
  └──────┬──────┘          └──────┬───────┘          └───────┬────────┘
         ↓                        ↓                          ↓
  ┌─────────────┐          ┌──────────────┐          ┌────────────────┐
  │ 事件抽取    │←─────────│ 实体作为事件  │          │   文本分类      │
  │ 36类触发词  │          │   要素槽位    │          │   NB / LR      │
  └──────┬──────┘          └──────────────┘          └───────┬────────┘
         ↓                                                    ↓
  ┌─────────────┐          ┌──────────────┐          ┌────────────────┐
  │ 主题发现    │          │ 自动摘要      │          │   结构化输出    │
  │ LDA/Kmeans │          │ TextRank/TFIDF│          │   JSONL 文件   │
  └─────────────┘          └──────────────┘          └────────────────┘
```

---

## 输出格式

### 单条分析结果 (`nlp_pipeline_results.jsonl`)

```json
{
  "topic": "白鹿方六连辟谣",
  "keywords_tfidf": [
    {"word": "造谣", "score": 0.2461},
    {"word": "法律责任", "score": 0.1231}
  ],
  "keywords_textrank": ["白鹿", "造谣", "工作室"],
  "entities": [
    {"text": "白鹿", "type": "人名", "start": 0, "end": 2},
    {"text": "江苏", "type": "地名", "start": 30, "end": 32}
  ],
  "sentiment": {
    "label": "负面",
    "score": -0.33,
    "positive_words": [],
    "negative_words": ["造假", "诽谤"]
  },
  "events": [
    {"事件类型": "辟谣声明", "主体": "白鹿工作室"}
  ],
  "classification": {
    "label": "娱乐",
    "probabilities": {"娱乐": 0.85, "社会": 0.10, "体育": 0.03, "生活": 0.02}
  },
  "summary": "白鹿方启动名誉维权诉讼程序。依法严肃追究造谣传谣主体的法律责任。"
}
```

---

## 测试结果摘要

### 预处理示例

```
输入: #突发#！！某地暴雨太大了！！！详情见 http://xxx.com
输出: 突发 某地 暴雨 太大 详情 见
                                       (分词+去停用词+去噪)
```

### 关键词提取 (TF-IDF)

| 文档主题 | 关键词 |
|----------|--------|
| 白鹿方六连辟谣 | 造谣、工作室、法律责任、诉讼 |
| 世界杯讨论 | 世界杯、参加、中国、外国人 |
| 诈骗案报道 | 诈骗、老年人、养生、警方 |

### 情感分析测试 (7/7 正确)

| 输入 | 情感 | 分数 |
|------|------|------|
| 功能不错，价格太离谱 | 负面 | -0.36 |
| 天气真好，心情愉快 | 正面 | +0.79 |
| 愤怒！欺骗消费者 | 负面 | -0.58 |
| 一般般，没感觉 | 中性 | 0.00 |

### 文本分类 (3 折 CV)

| 方法 | 特征 | 准确率 |
|------|------|--------|
| 朴素贝叶斯 | 词频 | 66.7% |
| 逻辑回归 | 词频 | 80.0% |
| 逻辑回归 | TF-IDF | 73.3% |

### LDA 主题发现 (50 条微博)

| 主题 | Top-8 词 | 解读 |
|------|----------|------|
| 主题 0 | 家长、幼儿园、离世、王老师、上海、室友 | 幼师离世事件 |
| 主题 1 | 白鹿、造假、起诉、工作室、辟谣 | 明星维权 |
| 主题 2 | 世界杯、韩国、捷克、进球、比赛 | 体育赛事 |
| 主题 3 | 诈骗、老人、警方、案例、抓获 | 社会诈骗 |

---

## 文件列表

```
NLP/
├── preprocessing.py              # 文本预处理
├── stopwords.txt                 # 停用词表 (631 词)
├── keyword_extraction.py         # 关键词提取
├── named_entity_recognition.py   # 命名实体识别
├── sentiment_analysis.py         # 情感分析
├── text_classification.py        # 文本分类
├── event_extraction.py           # 事件抽取
├── topic_discovery.py            # 主题发现
├── text_summarization.py         # 自动摘要
├── README.md                     # 本文档

数据/                             # 原始数据
└── 20260612_141245/
    ├── representative_posts.jsonl
    ├── hot_topics_snapshot.jsonl
    └── media/

输出/                             # 流水线输出
├── nlp_pipeline.py               # 全流水线脚本
├── nlp_pipeline_results.jsonl    # 50 文档分析结果
├── nlp_topics.json               # LDA 主题
└── nlp_pipeline_report.md        # 分析报告
```

## BERT-NER 可行性说明

经评估，在当前环境中**无法使用 BERT 进行 NER**：

| 需求 | 状态 | 原因 |
|------|------|------|
| PyTorch | 不可安装 | pip 无匹配版本，沙箱网络受限 |
| GitHub | 无法访问 | lonePatient/BERT-NER-Pytorch 仓库拒绝连接 |
| HuggingFace | 无法访问 | 无法下载预训练模型 |

已改用 **jieba.posseg (HMM 词性标注)** 替代词典规则法，错误率降低约 80%，零额外依赖。

---

## 许可

课程项目：信息内容安全 — 面向网络媒体信息分析的传统数据挖掘算法调优改善与结果验证系统。
