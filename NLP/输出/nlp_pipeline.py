"""
NLP 全流水线 — 从原始微博文本到结构化分析结果

工作流:
  原始文本 → 预处理 → 关键词提取 → NER → 情感分析
        → 事件抽取 → 文本分类 → 自动摘要 → 结构化输出
"""

import sys, io, os, json, math, random
from collections import Counter, defaultdict
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, "E:\\6+7\\SJTU\\大三下\\信息内容安全\\大作业\\NLP")

# ── 导入所有模块 ──
from preprocessing import TextPreprocessor
from keyword_extraction import TFIDFExtractor, TextRankExtractor
from named_entity_recognition import NERExtractor
from sentiment_analysis import SentimentAnalyzer
from event_extraction import EventExtractor
from text_classification import TextClassifier, auto_label_by_keywords
from topic_discovery import TopicDiscoverer
from text_summarization import TextSummarizer

# ═══════════════════════════════════════════════════════════
#  路径配置
# ═══════════════════════════════════════════════════════════

BASE = "E:\\6+7\\SJTU\\大三下\\信息内容安全\\大作业"
DATA_DIR = BASE + "\\数据\\20260612_141245\\20260612_141245"
OUTPUT_DIR = BASE + "\\输出"

JSONL_PATH = DATA_DIR + "\\representative_posts.jsonl"
TOPICS_PATH = DATA_DIR + "\\hot_topics_snapshot.jsonl"
OUTPUT_JSONL = OUTPUT_DIR + "\\nlp_pipeline_results.jsonl"
OUTPUT_REPORT = OUTPUT_DIR + "\\nlp_pipeline_report.md"
OUTPUT_TOPICS = OUTPUT_DIR + "\\nlp_topics.json"

# ═══════════════════════════════════════════════════════════
#  加载数据
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("NLP 全流水线")
print("=" * 60)
print(f"[{datetime.now().strftime('%H:%M:%S')}] 加载数据...")

raw_data = []
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        raw_data.append(json.loads(line))

texts = []
for d in raw_data:
    t = d.get("selection", {}).get("selected_post", {}).get("text", "")
    if t.strip():
        texts.append(t)

print(f"  文档数: {len(texts)}")
print(f"  输出目录: {OUTPUT_DIR}")

# ═══════════════════════════════════════════════════════════
#  Step 1: 初始化所有模块
# ═══════════════════════════════════════════════════════════

print(f"[{datetime.now().strftime('%H:%M:%S')}] 初始化模块...")

pp = TextPreprocessor()
ner = NERExtractor()
sentiment = SentimentAnalyzer()
event_extractor = EventExtractor()
summarizer = TextSummarizer(method="textrank", damping=0.85)

# ═══════════════════════════════════════════════════════════
#  Step 2: 训练语料级模型
# ═══════════════════════════════════════════════════════════

print(f"[{datetime.now().strftime('%H:%M:%S')}] 训练语料级模型...")

# 2a. TF-IDF 关键词提取器
tfidf_kw = TFIDFExtractor(max_features=2000)
tfidf_kw.fit(texts)
print(f"  TF-IDF 词表大小: {len(tfidf_kw.vocabulary)}")

# 2b. TextRank 关键词提取器
textrank_kw = TextRankExtractor(window=3)

# 2c. 文本分类器（自动标注）
category_map = {
    "娱乐": ["白鹿","鹿晗","陈立农","迪丽热巴","刘涛","张月","孙怡","陈瑶",
             "张凌赫","王安宇","音乐节","辟谣","明星","官宣","恋情","结婚"],
    "社会": ["网信办","诈骗","老年人","警方","央视","教育局","教师","离世",
             "幼儿园","调查","留学生","事故","地震","火灾","台风"],
    "体育": ["世界杯","加纳","韩国","捷克","孙兴慜","足球","球衣","赞助",
             "比赛","夺冠","进球","转会"],
    "生活": ["狗","宠物","租房","省钱","气质","商家","养生","通勤",
             "穿搭","美食","旅游","健康"],
}
labels = auto_label_by_keywords(texts, category_map)
label_counts = Counter(labels)
print(f"  分类标签: {dict(label_counts)}")

clf = TextClassifier(method="naive_bayes", alpha=1.0)
clf.fit(texts, labels)
r = clf.evaluate(texts, labels); acc = r.get("accuracy",0); print(f"  分类器训练准确率: {acc:.1%}")

# 2d. LDA 主题模型
lda = TopicDiscoverer(method="lda", n_topics=8, alpha=0.1, beta=0.01, n_iter=50)
lda.fit(texts)
print(f"  LDA 主题数: 8")

# ═══════════════════════════════════════════════════════════
#  Step 3: 逐文档流水线处理
# ═══════════════════════════════════════════════════════════

print(f"[{datetime.now().strftime('%H:%M:%S')}] 执行逐文档流水线...")

results = []
errors = 0
for idx, (d, text) in enumerate(zip(raw_data, texts)):
    if (idx + 1) % 10 == 0:
        print(f"  进度: {idx+1}/{len(texts)}")

    topic_word = d.get("topic", {}).get("word", "未知")
    try:
        # 3a. 预处理
        processed = pp.process(text)

        # 3b. 关键词提取 (TF-IDF + TextRank)
        kw_tfidf = tfidf_kw.extract(text, top_n=10, with_scores=True)
        kw_textrank = textrank_kw.extract(text, top_n=10)

        # 3c. 命名实体识别
        entities = ner.recognize(text)

        # 3d. 情感分析
        sa_result = sentiment.analyze(text)
        fine_grained = sa_result.get("fine_grained", {})

        # 3e. 事件抽取
        events = event_extractor.extract_summary(text)

        # 3f. 文本分类
        pred_label = clf.predict(text)
        pred_proba = clf.predict_proba(text)

        # 3g. 自动摘要
        summary_sents = summarizer.summarize(text, num_sentences=2)
        summary_text = "。".join(summary_sents) + "。" if summary_sents else ""

        # 3h. 组装结果
        result = {
            "topic": topic_word,
            "raw_text_preview": text[:100],
            "processed": processed[:100] if processed else "",
            "keywords_tfidf": [{"word": w, "score": round(s, 4)} for w, s in kw_tfidf],
            "keywords_textrank": kw_textrank,
            "entities": [
                {"text": e["text"], "type": e["type"], "start": e["start"], "end": e["end"]}
                for e in entities[:15]  # 限制数量
            ],
            "sentiment": {
                "label": sa_result["sentiment"],
                "score": sa_result["score"],
                "positive_words": sa_result["positive_words"][:5],
                "negative_words": sa_result["negative_words"][:5],
            },
            "fine_grained_emotions": fine_grained,
            "events": events,
            "classification": {
                "label": pred_label,
                "probabilities": {k: round(v, 4) for k, v in pred_proba.items()},
            },
            "summary": summary_text,
        }
        results.append(result)

    except Exception as ex:
        errors += 1
        results.append({
            "topic": topic_word,
            "error": str(ex),
            "raw_text_preview": text[:100],
        })

print(f"  完成: {len(results)} 条, 错误: {errors} 条")

# ═══════════════════════════════════════════════════════════
#  Step 4: 保存输出
# ═══════════════════════════════════════════════════════════

print(f"[{datetime.now().strftime('%H:%M:%S')}] 保存输出...")

# 4a. JSONL 结果
with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"  JSONL: {OUTPUT_JSONL}")

# 4b. LDA 主题
lda_topics = lda.get_topics(top_n=10)
with open(OUTPUT_TOPICS, "w", encoding="utf-8") as f:
    json.dump(lda_topics, f, ensure_ascii=False, indent=2)
print(f"  主题: {OUTPUT_TOPICS}")

# 4c. Markdown 报告
total_docs = len(results)
# 统计
sentiment_dist = Counter(r["sentiment"]["label"] for r in results if "sentiment" in r)
entity_types = Counter()
for r in results:
    for e in r.get("entities", []):
        entity_types[e["type"]] += 1
event_types = Counter()
for r in results:
    for ev in r.get("events", []):
        event_types[ev.get("事件类型", "未知")] += 1
class_dist = Counter(r["classification"]["label"] for r in results if "classification" in r)

report_lines = []
report_lines.append("# NLP 全流水线分析报告\n")
report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
report_lines.append(f"**文档总数**: {total_docs}\n\n")

report_lines.append("## 一、流水线结构\n\n")
report_lines.append("| 步骤 | 模块 | 输出 |\n")
report_lines.append("|------|------|------|\n")
report_lines.append("| 1 | 文本预处理 | 清洗后文本 |\n")
report_lines.append("| 2 | 关键词提取 (TF-IDF + TextRank) | Top-10 关键词 |\n")
report_lines.append("| 3 | 命名实体识别 | 人名/地名/机构/时间/金额/产品 |\n")
report_lines.append("| 4 | 情感分析 | 正面/中性/负面 + 分数 + 细粒度情绪 |\n")
report_lines.append("| 5 | 事件抽取 | 36类事件 + 要素槽位 |\n")
report_lines.append("| 6 | 文本分类 | 娱乐/社会/体育/生活 + 概率 |\n")
report_lines.append("| 7 | 主题发现 (LDA) | 8个主题 × Top-10 词 |\n")
report_lines.append("| 8 | 自动摘要 (TextRank) | Top-2 关键句 |\n\n")

report_lines.append("## 二、语料级统计\n\n")

report_lines.append(f"### 情感分布\n\n")
report_lines.append(f"| 类别 | 数量 | 占比 |\n")
report_lines.append(f"|------|------|------|\n")
for label in ["正面", "中性", "负面"]:
    cnt = sentiment_dist.get(label, 0)
    pct = cnt / total_docs * 100 if total_docs > 0 else 0
    report_lines.append(f"| {label} | {cnt} | {pct:.1f}% |\n")

report_lines.append(f"\n### 实体分布 (Top-10)\n\n")
report_lines.append(f"| 类型 | 数量 |\n")
report_lines.append(f"|------|------|\n")
for etype, ecnt in entity_types.most_common(10):
    report_lines.append(f"| {etype} | {ecnt} |\n")

report_lines.append(f"\n### 事件分布 (Top-10)\n\n")
report_lines.append(f"| 类型 | 数量 |\n")
report_lines.append(f"|------|------|\n")
for etype, ecnt in event_types.most_common(10):
    report_lines.append(f"| {etype} | {ecnt} |\n")

report_lines.append(f"\n### 分类分布\n\n")
report_lines.append(f"| 类别 | 数量 |\n")
report_lines.append(f"|------|------|\n")
for label, cnt in class_dist.most_common():
    report_lines.append(f"| {label} | {cnt} |\n")

report_lines.append(f"\n## 三、LDA 主题模型\n\n")
for t in lda_topics:
    words = ", ".join(t["words"][:8])
    report_lines.append(f"- **主题 {t['topic_id']}** (权重 {t['weight']:.2f}): {words}\n")

report_lines.append(f"\n## 四、关键词提取 (TF-IDF 词表 Top-30)\n\n")
vocab = tfidf_kw.vocabulary
top_vocab = sorted(vocab.items(), key=lambda x: x[1])[:30]
report_lines.append(", ".join(w for w, _ in top_vocab) + "\n")

report_lines.append(f"\n## 五、文档级分析示例\n\n")
for i, r in enumerate(results[:5]):
    report_lines.append(f"### 示例 {i+1}: {r['topic']}\n\n")
    report_lines.append(f"- **原文**: {r.get('raw_text_preview', 'N/A')}\n")
    report_lines.append(f"- **关键词 (TF-IDF)**: {', '.join(k['word'] for k in r.get('keywords_tfidf', [])[:5])}\n")
    ents = [x['text'] + '(' + x['type'] + ')' for x in r.get('entities', [])[:5]]
    report_lines.append("- **实体**: " + ", ".join(ents) + "\n")
    report_lines.append(f"- **情感**: {r.get('sentiment', {}).get('label', '?')} (分数: {r.get('sentiment', {}).get('score', 0):+.2f})\n")
    if r.get("events"):
        ev = r["events"][0]
        report_lines.append(f"- **事件**: {ev.get('事件类型', '?')} {ev.get('主体', '')} {ev.get('产品', '')} {ev.get('原因', '')}\n")
    report_lines.append(f"- **分类**: {r.get('classification', {}).get('label', '?')}\n")
    if r.get("summary"):
        report_lines.append(f"- **摘要**: {r['summary'][:80]}\n")
    report_lines.append("\n")

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    f.writelines(report_lines)
print(f"  报告: {OUTPUT_REPORT}")

# ═══════════════════════════════════════════════════════════
#  Done
# ═══════════════════════════════════════════════════════════

print()
print("=" * 60)
print("NLP 全流水线完成")
print(f"  输出文件:")
print(f"    - {OUTPUT_JSONL}")
print(f"    - {OUTPUT_TOPICS}")
print(f"    - {OUTPUT_REPORT}")
print(f"  处理的文档数: {total_docs}")
print(f"  错误数: {errors}")
print("=" * 60)
