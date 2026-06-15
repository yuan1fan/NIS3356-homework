# -*- coding: utf-8 -*-
"""
NLP 文本智能分析平台 - Gradio UI

提供：

  - 传统模式：使用 NLP/ 目录下的 8 个规则/统计 NLP 模块
  - LLM 模式：使用 DeepSeek API 进行智能 NLP 分析
  - 单文本分析 + 8 个子模块可视化结果

Usage:
    python UI/NLP/app.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import gradio as gr

# ---- 路径设置 --------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
NLP_DIR = BASE_DIR / "NLP"
NLP_LLM_DIR = BASE_DIR / "NLP LLM"

sys.path.insert(0, str(NLP_DIR))
sys.path.insert(0, str(NLP_LLM_DIR))

logger = logging.getLogger(__name__)

_traditional_modules = None
_deepseek_nlp = None


# ============================================
# 1. 传统模式：加载 NLP 模块
# ============================================

def load_traditional_modules():
    """Initialize traditional NLP modules (lazy load)."""
    global _traditional_modules
    if _traditional_modules is not None:
        return _traditional_modules

    logger.info("加载传统 NLP 模块...")

    from preprocessing import TextPreprocessor
    from keyword_extraction import TextRankExtractor
    from named_entity_recognition import NERExtractor
    from sentiment_analysis import SentimentAnalyzer
    from event_extraction import EventExtractor
    from text_summarization import TextSummarizer

    mods = {
        "pp": TextPreprocessor(),
        "textrank_kw": TextRankExtractor(window=3),
        "ner": NERExtractor(),
        "sentiment": SentimentAnalyzer(),
        "event_extractor": EventExtractor(),
        "summarizer": TextSummarizer(method="textrank", damping=0.85),
        "_classifier": None,
        "_lda": None,
    }

    # 尝试加载训练语料
    data_dirs = [
        BASE_DIR / "数据" / "20260612_141245" / "20260612_141245",
        BASE_DIR / "数据",
    ]
    jsonl_path = None
    for dd in data_dirs:
        p = dd / "representative_posts.jsonl"
        if p.exists():
            jsonl_path = p
            break

    if jsonl_path and jsonl_path.exists():
        try:
            texts = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    t = d.get("selection", {}).get("selected_post", {}).get("text", "")
                    if t.strip():
                        texts.append(t)
            if texts:
                logger.info(f"加载语料: {len(texts)} 篇文档")

                from keyword_extraction import TFIDFExtractor
                tfidf = TFIDFExtractor(max_features=2000)
                tfidf.fit(texts)
                mods["tfidf_kw"] = tfidf

                from text_classification import TextClassifier, auto_label_by_keywords
                category_map = {
                    "娱乐": ["白鹿", "鹿晗", "陈立农", "迪丽热巴", "刘涛", "张月",
                             "孙怡", "陈瑶", "张凌赫", "王安宇", "音乐节", "辟谣",
                             "明星", "官宣", "恋情", "结婚", "演唱会"],
                    "社会": ["网信办", "诈骗", "老年人", "警方", "央视", "教育局",
                             "教师", "离世", "幼儿园", "调查", "留学生", "事故",
                             "地震", "火灾", "台风", "暴雨"],
                    "体育": ["世界杯", "加纳", "韩国", "捷克", "孙兴慜", "足球",
                             "球衣", "赞助", "比赛", "夺冠", "进球", "转会"],
                    "生活": ["狗", "宠物", "租房", "省钱", "气质", "商家",
                             "养生", "通勤", "穿搭", "美食", "旅游", "健康"],
                }
                labels = auto_label_by_keywords(texts, category_map)
                clf = TextClassifier(method="naive_bayes", alpha=1.0)
                clf.fit(texts, labels)
                mods["_classifier"] = clf
                mods["_category_map"] = category_map

                from topic_discovery import TopicDiscoverer
                lda = TopicDiscoverer(method="lda", n_topics=8,
                                      alpha=0.1, beta=0.01, n_iter=50)
                lda.fit(texts)
                mods["_lda"] = lda
                mods["_lda_topics"] = lda.get_topics(top_n=8)

                logger.info("语料级模型训练完成")
        except Exception as e:
            logger.warning(f"加载语料失败: {e}")
    else:
        logger.info("未找到语料文件，部分模块将使用单文档模式")

    _traditional_modules = mods
    return mods


# ============================================
# 2. LLM 模式：加载 DeepSeek NLP
# ============================================

def get_deepseek_nlp(api_key: str = "") -> Any:
    """获取 DeepSeek NLP 实例（懒加载）。"""
    global _deepseek_nlp
    from deepseek_nlp import DeepSeekNLP

    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if _deepseek_nlp is None:
        _deepseek_nlp = DeepSeekNLP(api_key=key)
    elif key and key != _deepseek_nlp.api_key:
        _deepseek_nlp = DeepSeekNLP(api_key=key)
    return _deepseek_nlp


# ============================================
# 3. 核心分析函数：传统模式
# ============================================

def analyze_text_traditional(
    text: str,
    run_preprocess: bool = True,
    run_keyword: bool = True,
    run_ner: bool = True,
    run_sentiment: bool = True,
    run_classify: bool = True,
    run_event: bool = True,
    run_topic: bool = True,
    run_summary: bool = True,
    keyword_method: str = "textrank",
    keyword_top_n: int = 10,
    summary_sentences: int = 3,
) -> Dict[str, Any]:
    """使用传统 NLP 模块分析单篇文本。"""
    mods = load_traditional_modules()
    pp = mods["pp"]
    result: Dict[str, Any] = {}

    if run_preprocess:
        try:
            processed = pp.process(text)
            steps = pp.process_with_steps(text)
            result["preprocess"] = {
                "original": text,
                "processed": processed,
                "steps": steps,
            }
        except Exception as e:
            result["preprocess"] = {"error": str(e)}
    else:
        processed = pp.process(text)

    if run_keyword:
        try:
            if keyword_method == "tfidf" and "tfidf_kw" in mods:
                kw = mods["tfidf_kw"].extract(text, top_n=keyword_top_n, with_scores=True)
                result["keyword"] = {
                    "method": "TF-IDF",
                    "keywords": [{"word": w, "score": round(s, 4)} for w, s in kw],
                }
            else:
                kw = mods["textrank_kw"].extract(text, top_n=keyword_top_n, with_scores=True)
                result["keyword"] = {
                    "method": "TextRank",
                    "keywords": [{"word": w, "score": round(s, 4)} for w, s in kw],
                }
        except Exception as e:
            result["keyword"] = {"error": str(e)}

    if run_ner:
        try:
            entities = mods["ner"].recognize(text)
            result["ner"] = {
                "entities": [
                    {"text": e["text"], "type": e["type"],
                     "start": e["start"], "end": e["end"]}
                    for e in entities
                ],
                "count": len(entities),
            }
        except Exception as e:
            result["ner"] = {"error": str(e)}

    if run_sentiment:
        try:
            sa = mods["sentiment"].analyze(text)
            aspects = mods["sentiment"].analyze_aspects(text)
            result["sentiment"] = {
                "sentiment": sa["sentiment"],
                "score": sa["score"],
                "positive_words": sa["positive_words"],
                "negative_words": sa["negative_words"],
                "fine_grained": sa["fine_grained"],
                "aspects": aspects.get("aspects", []),
            }
        except Exception as e:
            result["sentiment"] = {"error": str(e)}

    if run_classify:
        try:
            if mods["_classifier"] is not None:
                label = mods["_classifier"].predict(text)
                proba = mods["_classifier"].predict_proba(text)
                result["classification"] = {
                    "label": label,
                    "probabilities": {k: round(v, 4) for k, v in proba.items()},
                }
            else:
                category_map = mods.get("_category_map", {})
                pp_text = pp.process(text)
                matched = []
                for cat, kws in category_map.items():
                    for kw in kws:
                        if kw in pp_text:
                            matched.append(cat)
                            break
                from collections import Counter
                label = Counter(matched).most_common(1)[0][0] if matched else "其他"
                result["classification"] = {
                    "label": label,
                    "probabilities": {},
                    "note": "基于关键词匹配（未加载训练模型）",
                }
        except Exception as e:
            result["classification"] = {"error": str(e)}

    if run_event:
        try:
            events = mods["event_extractor"].extract_summary(text)
            result["event"] = {"events": events, "count": len(events)}
        except Exception as e:
            result["event"] = {"error": str(e)}

    if run_topic:
        try:
            if mods["_lda"] is not None:
                theta = mods["_lda"].transform([text])[0]
                topics = mods["_lda_topics"]
                topic_dist = []
                for k, prob in enumerate(theta):
                    if prob > 0.05 and k < len(topics):
                        topic_dist.append({
                            "topic_id": k,
                            "words": topics[k]["words"][:5],
                            "probability": round(float(prob), 4),
                        })
                topic_dist.sort(key=lambda x: -x["probability"])
                result["topic"] = {
                    "model": "LDA",
                    "topic_distribution": topic_dist,
                    "all_topics": [{
                        "topic_id": t["topic_id"],
                        "words": t["words"],
                        "weight": t["weight"],
                    } for t in topics],
                }
            else:
                result["topic"] = {"model": "N/A", "note": "未加载语料，无法运行主题模型"}
        except Exception as e:
            result["topic"] = {"error": str(e)}

    if run_summary:
        try:
            summary_text = mods["summarizer"].summarize_text(text, num_sentences=summary_sentences)
            summary_sents = mods["summarizer"].summarize(text, num_sentences=summary_sentences)
            result["summary"] = {
                "text": summary_text,
                "sentences": summary_sents,
                "sentence_count": len(summary_sents),
            }
        except Exception as e:
            result["summary"] = {"error": str(e)}

    return result


# ============================================
# 4. 核心分析函数：LLM 模式
# ============================================

def analyze_text_llm(
    text: str,
    api_key: str = "",
    run_ner: bool = True,
    run_sentiment: bool = True,
    run_keyword: bool = True,
    run_classify: bool = True,
    run_event: bool = True,
    run_summary: bool = True,
) -> Dict[str, Any]:
    """使用 DeepSeek API 分析文本。"""
    nlp = get_deepseek_nlp(api_key)
    result: Dict[str, Any] = {}

    if not nlp.api_key:
        return {
            "error": "未配置 DeepSeek API Key",
            "hint": ("请通过以下方式之一配置：\n"
                     "1. 在界面中输入 API Key\n"
                     "2. 设置环境变量: set DEEPSEEK_API_KEY=sk-xxx\n"
                     "3. 访问 https://platform.deepseek.com 获取 Key"),
        }

    if run_ner and run_sentiment and run_keyword and run_classify and run_event and run_summary:
        try:
            analysis = nlp.analyze(text)
            result["ner"] = {"entities": analysis.get("entities", []),
                             "count": len(analysis.get("entities", []))}
            s = analysis.get("sentiment", {})
            result["sentiment"] = {
                "sentiment": s.get("sentiment", "中性"),
                "score": s.get("score", 0.0),
            }
            result["keyword"] = {
                "method": "DeepSeek LLM",
                "keywords": analysis.get("keywords", []),
            }
            result["classification"] = analysis.get("classification", {"label": "其他"})
            result["event"] = {
                "events": analysis.get("events", []),
                "count": len(analysis.get("events", [])),
            }
            result["summary"] = {
                "text": analysis.get("summary", ""),
                "sentence_count": len(analysis.get("summary", "").split("。")) - 1,
            }
        except Exception as e:
            result["error"] = str(e)
    else:
        if run_ner:
            try:
                entities = nlp.ner(text)
                result["ner"] = {"entities": entities, "count": len(entities)}
            except Exception as e:
                result["ner"] = {"error": str(e)}
        if run_sentiment:
            try:
                s = nlp.sentiment(text)
                result["sentiment"] = s
            except Exception as e:
                result["sentiment"] = {"error": str(e)}
        if run_keyword:
            try:
                kw = nlp.keyword(text)
                result["keyword"] = {"method": "DeepSeek LLM", "keywords": kw}
            except Exception as e:
                result["keyword"] = {"error": str(e)}
        if run_classify:
            try:
                c = nlp.classify(text)
                result["classification"] = c
            except Exception as e:
                result["classification"] = {"error": str(e)}
        if run_event:
            try:
                events = nlp.event_extract(text)
                result["event"] = {"events": events, "count": len(events)}
            except Exception as e:
                result["event"] = {"error": str(e)}
        if run_summary:
            try:
                summary = nlp.summarize(text)
                result["summary"] = {"text": summary, "sentence_count": len(summary.split("。")) - 1}
            except Exception as e:
                result["summary"] = {"error": str(e)}

    return result


# ============================================
# 5. 结果格式化
# ============================================

def format_analysis_result(result: Dict[str, Any], mode: str) -> Dict[str, str]:
    """将分析结果格式化为 HTML 片段用于 Gradio 显示。"""
    html = {}
    from collections import Counter

    if "error" in result:
        err = result["error"]
        hint = result.get("hint", "")
        return {"_error": f"<div class='error-box'><strong>{err}</strong>{'<br>' + hint if hint else ''}</div>"}

    if "preprocess" in result:
        p = result["preprocess"]
        if "error" in p:
            html["preprocess"] = f"<div class='error-box'>{p['error']}</div>"
        else:
            steps_html = ""
            for k, v in p.get("steps", {}).items():
                val = str(v)[:60] + ("..." if len(str(v)) > 60 else "")
                steps_html += f"<div class='step-row'><span class='step-label'>{k}</span><span class='step-value'>{val}</span></div>"
            html["preprocess"] = (
                f"<div class='result-section'><h4>预处理结果</h4>"
                f"<div class='result-box'>{p.get('processed', '')}</div>"
                f"<details><summary>查看处理步骤</summary>{steps_html}</details></div>"
            )

    if "keyword" in result:
        k = result["keyword"]
        if "error" in k:
            html["keyword"] = f"<div class='error-box'>{k['error']}</div>"
        else:
            kw_list = k.get("keywords", [])
            method = k.get("method", "")
            chips = "".join(
                f"<span class='keyword-chip'>{kw['word']} <span class='kw-score'>{kw.get('score', 0):.2f}</span></span>"
                for kw in kw_list[:15]
            )
            html["keyword"] = (
                f"<div class='result-section'><h4>关键词提取 <span class='method-tag'>{method}</span></h4>"
                f"<div class='keyword-cloud'>{chips}</div></div>"
            )

    if "ner" in result:
        n = result["ner"]
        if "error" in n:
            html["ner"] = f"<div class='error-box'>{n['error']}</div>"
        else:
            ents = n.get("entities", [])
            type_colors = {
                "人名": "#3b82f6", "地名": "#10b981", "机构名": "#8b5cf6",
                "时间": "#f59e0b", "金额": "#ef4444", "产品名": "#ec4899",
            }
            table_rows = "".join(
                "<tr><td><span class='entity-tag' style='background:" + type_colors.get(e["type"], "#6b7280") + f"'>" + e['type'] + "</span></td>"
                f"<td>{e['text']}</td></tr>"
                for e in ents[:20]
            )
            tc = Counter(e["type"] for e in ents)
            type_summary = "".join(
                f"<span class='entity-count'><span class='etype-dot' style='background:{type_colors.get(t, chr(35)+chr(54)+chr(98)+chr(55)+chr(50)+chr(56)+chr(48))}'></span>{t}: {c}</span>"
                for t, c in tc.most_common()
            )
            html["ner"] = (
                f"<div class='result-section'><h4>命名实体识别 <span class='count-badge'>{n['count']} 个实体</span></h4>"
                f"<div class='entity-type-summary'>{type_summary}</div>"
                f"<table class='entity-table'><thead><tr><th>类型</th><th>实体</th></tr></thead><tbody>{table_rows}</tbody></table></div>"
            )

    if "sentiment" in result:
        s = result["sentiment"]
        if "error" in s:
            html["sentiment"] = f"<div class='error-box'>{s['error']}</div>"
        else:
            sent_label = s.get("sentiment", "中性")
            score = s.get("score", 0.0)
            emoji_map = {"正面": "🟢", "中性": "🟡", "负面": "🔴"}
            emoji = emoji_map.get(sent_label, "⚪")
            bar_pct = max(5, min(95, (score + 1) * 50))
            bar_color = "#10b981" if score > 0.1 else "#6b7280" if score >= -0.1 else "#ef4444"
            pos_words = "、".join(s.get("positive_words", [])[:8])
            neg_words = "、".join(s.get("negative_words", [])[:8])
            fine = s.get("fine_grained", {})
            fine_html = "".join(f"<span class='fine-emotion'>{e}: {c}</span>" for e, c in fine.items())
            aspects = s.get("aspects", [])
            aspects_html = ""
            if aspects:
                aspects_html = "<h5>方面级分析</h5><div class='aspect-grid'>"
                for a in aspects:
                    a_emoji = emoji_map.get(a["sentiment"], "⚪")
                    aspects_html += f"<div class='aspect-item'><strong>{a['aspect']}</strong> {a_emoji} {a['sentiment']} ({a['score']:+.2f})</div>"
                aspects_html += "</div>"
            html["sentiment"] = (
                f"<div class='result-section'><h4>情感分析</h4>"
                f"<div class='sentiment-header'>{emoji} <strong>{sent_label}</strong> <span class='sentiment-score'>({score:+.2f})</span></div>"
                f"<div class='sentiment-bar'><div class='sentiment-fill' style='width:{bar_pct}%;background:{bar_color}'></div></div>"
                + (f"<div class='word-list'><span class='pos'>正面词: {pos_words}</span></div>" if pos_words else "")
                + (f"<div class='word-list'><span class='neg'>负面词: {neg_words}</span></div>" if neg_words else "")
                + (f"<div class='fine-grid'>{fine_html}</div>" if fine_html else "") + aspects_html + "</div>"
            )

    if "classification" in result:
        c = result["classification"]
        if "error" in c:
            html["classification"] = f"<div class='error-box'>{c['error']}</div>"
        else:
            label = c.get("label", "其他")
            proba = c.get("probabilities", {})
            note = c.get("note", "")
            proba_bars = "".join(
                f"<div class='proba-row'><span class='proba-label'>{cat}</span>"
                f"<div class='proba-track'><div class='proba-fill' style='width:{max(3, p * 100)}%'></div></div>"
                f"<span class='proba-value'>{p:.1%}</span></div>"
                for cat, p in sorted(proba.items(), key=lambda x: -x[1])
            )
            html["classification"] = (
                f"<div class='result-section'><h4>文本分类</h4>"
                f"<div class='class-label'>{label}</div>"
                + (f"<p class='note'>{note}</p>" if note else "")
                + (f"<div class='proba-container'>{proba_bars}</div>" if proba_bars else "") + "</div>"
            )

    if "event" in result:
        e = result["event"]
        if "error" in e:
            html["event"] = f"<div class='error-box'>{e['error']}</div>"
        else:
            events = e.get("events", [])
            if not events:
                html["event"] = "<div class='result-section'><h4>事件抽取</h4><p class='empty-hint'>(未识别到事件)</p></div>"
            else:
                cards = ""
                for i, ev in enumerate(events[:5]):
                    slots = "".join(
                        f"<div class='slot-row'><span class='slot-name'>{k}</span><span class='slot-value'>{v}</span></div>"
                        for k, v in ev.items() if v
                    )
                    cards += f"<div class='event-card'><div class='event-type'>{ev.get('事件类型', '未知')}</div>{slots}</div>"
                html["event"] = (
                    f"<div class='result-section'><h4>事件抽取 <span class='count-badge'>{e['count']} 个事件</span></h4>"
                    f"<div class='event-grid'>{cards}</div></div>"
                )

    if "topic" in result:
        t = result["topic"]
        if "error" in t:
            html["topic"] = f"<div class='error-box'>{t['error']}</div>"
        else:
            note = t.get("note", "")
            all_topics = "".join(
                f"<div class='topic-item'>主题 {tp['topic_id']}: {'、'.join(tp['words'])} <span class='topic-weight'>w={tp['weight']}</span></div>"
                for tp in t.get("all_topics", [])
            )
            dist = t.get("topic_distribution", [])
            dist_html = "".join(
                f"<div class='proba-row'><span class='proba-label'>主题 {d['topic_id']}</span>"
                f"<div class='proba-track'><div class='proba-fill' style='width:{d['probability']*100}%'></div></div>"
                f"<span class='proba-value'>{d['probability']:.1%}</span></div>"
                for d in dist
            )
            html["topic"] = (
                f"<div class='result-section'><h4>主题发现 <span class='method-tag'>{t.get('model', '')}</span></h4>"
                + (f"<p class='note'>{note}</p>" if note else "")
                + (f"<h5>当前文档主题分布</h5><div class='proba-container'>{dist_html}</div>" if dist_html else "")
                + f"<details><summary>全部主题词</summary><div class='topic-grid'>{all_topics}</div></details></div>"
            )

    if "summary" in result:
        s = result["summary"]
        if "error" in s:
            html["summary"] = f"<div class='error-box'>{s['error']}</div>"
        else:
            html["summary"] = (
                f"<div class='result-section'><h4>自动摘要 <span class='count-badge'>{s.get('sentence_count', 0)} 句</span></h4>"
                f"<div class='summary-box'>{s.get('text', '')}</div></div>"
            )

    return html


# ============================================
# 6. 报告生成
# ============================================

def generate_full_report(result: Dict[str, Any], mode: str) -> str:
    lines = ["=" * 60, "NLP 文本分析报告", "=" * 60,
             f'分析模式: {"传统模式" if mode == "traditional" else "LLM模式"}', ""]

    if "preprocess" in result:
        p = result["preprocess"]
        lines += ["-" * 60, "一、文本预处理", "-" * 60,
                  f"处理前: {p.get('original', '')[:100]}",
                  f"处理后: {p.get('processed', '')}"]

    if "keyword" in result:
        k = result["keyword"]
        if "keywords" in k:
            kw_str = ", ".join(kw["word"] for kw in k["keywords"][:10])
            lines += ["", "-" * 60, "二、关键词提取", "-" * 60,
                      f"方法: {k.get('method', '')}", f"关键词: {kw_str}"]

    if "ner" in result:
        n = result["ner"]
        if "entities" in n:
            lines += ["", "-" * 60, "三、命名实体识别", "-" * 60]
            for e in n["entities"][:15]:
                lines.append(f"  [{e['type']}] {e['text']}")

    if "sentiment" in result:
        s = result["sentiment"]
        if "sentiment" in s:
            lines += ["", "-" * 60, "四、情感分析", "-" * 60,
                      f"情感: {s['sentiment']} (分数: {s.get('score', 0):+.2f})",
                      f"正面词: {', '.join(s.get('positive_words', [])[:5])}",
                      f"负面词: {', '.join(s.get('negative_words', [])[:5])}"]

    if "classification" in result:
        c = result["classification"]
        if "label" in c:
            lines += ["", "-" * 60, "五、文本分类", "-" * 60, f"类别: {c['label']}"]

    if "event" in result:
        e = result["event"]
        if "events" in e and e["events"]:
            lines += ["", "-" * 60, "六、事件抽取", "-" * 60]
            for ev in e["events"][:5]:
                ev_type = ev.get("事件类型", "未知")
                info = "; ".join(f"{k}={v}" for k, v in ev.items() if k != "事件类型" and v)
                lines.append(f"  [{ev_type}] {info}")

    if "topic" in result:
        t = result["topic"]
        if "topic_distribution" in t:
            lines += ["", "-" * 60, "七、主题发现", "-" * 60]
            for d in t["topic_distribution"]:
                lines.append(f"  主题 {d['topic_id']} ({d['probability']:.1%}): {'、'.join(d['words'])}")

    if "summary" in result:
        s = result["summary"]
        if "text" in s:
            lines += ["", "-" * 60, "八、自动摘要", "-" * 60, s["text"]]

    lines += ["", "=" * 60, "报告完毕", "=" * 60]
    return "\n".join(lines)

# ============================================
# 7. Gradio 界面
# ============================================

CSS = """
:root { --bg: #f7f8fb; --panel: #ffffff; --text: #1f2937; --muted: #6b7280; --line: #e5e7eb; --accent: #2563eb; --ok: #10b981; --danger: #ef4444; }
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--text); font-family: sans-serif; }
.app-shell { max-width: 1440px; margin: 0 auto; padding: 20px; }
.app-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; flex-wrap: wrap; gap: 12px; }
.app-header h1 { font-size: 28px; margin: 0; }
.app-header .eyebrow { color: var(--accent); font-size: 13px; font-weight: 700; margin: 0 0 4px; }
.app-header .status { display: inline-flex; align-items: center; gap: 8px; border: 1px solid var(--line); border-radius: 999px; padding: 6px 14px; font-size: 13px; color: var(--muted); }
.app-header .status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); }
.control-section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; box-shadow: 0 2px 12px rgba(0,0,0,0.04); }
.result-section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; margin-bottom: 12px; }
.result-section h4 { margin: 0 0 12px; font-size: 16px; display: flex; align-items: center; gap: 8px; }
.result-section h5 { margin: 10px 0 6px; font-size: 14px; color: var(--muted); }
.method-tag, .count-badge { font-size: 12px; font-weight: 600; padding: 2px 8px; border-radius: 999px; background: #eff6ff; color: var(--accent); }
.count-badge { background: #f3f4f6; color: var(--muted); }
.result-box { background: #f9fafb; border: 1px solid var(--line); border-radius: 6px; padding: 12px; font-size: 14px; line-height: 1.6; margin-bottom: 8px; }
.step-row { display: flex; gap: 8px; padding: 4px 0; font-size: 13px; border-bottom: 1px solid #f3f4f6; }
.step-label { color: var(--accent); font-weight: 600; min-width: 120px; }
.step-value { color: var(--text); overflow: hidden; text-overflow: ellipsis; }
.keyword-cloud { display: flex; flex-wrap: wrap; gap: 8px; }
.keyword-chip { display: inline-flex; align-items: center; gap: 6px; border: 1px solid #bfdbfe; border-radius: 999px; background: #eff6ff; color: #1d4ed8; padding: 5px 12px; font-size: 14px; }
.keyword-chip .kw-score { font-size: 11px; color: #6b7280; }
.entity-type-summary { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 12px; }
.entity-count { display: inline-flex; align-items: center; gap: 5px; font-size: 13px; color: var(--muted); }
.etype-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.entity-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.entity-table th { text-align: left; color: var(--muted); font-weight: 600; padding: 6px 8px; border-bottom: 2px solid var(--line); }
.entity-table td { padding: 6px 8px; border-bottom: 1px solid #f3f4f6; }
.entity-tag { display: inline-block; padding: 2px 8px; border-radius: 999px; color: white; font-size: 12px; font-weight: 600; }
.sentiment-header { font-size: 20px; margin-bottom: 8px; }
.sentiment-score { font-size: 16px; color: var(--muted); }
.sentiment-bar { height: 8px; background: #e5e7eb; border-radius: 999px; overflow: hidden; margin: 8px 0 12px; }
.sentiment-fill { height: 100%; border-radius: 999px; transition: width 0.5s; }
.word-list { margin: 4px 0; font-size: 13px; }
.word-list .pos { color: #059669; }
.word-list .neg { color: #dc2626; }
.fine-grid { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.fine-emotion { border: 1px solid var(--line); border-radius: 6px; padding: 4px 10px; font-size: 12px; background: #f9fafb; }
.class-label { font-size: 22px; font-weight: 700; color: var(--accent); margin-bottom: 8px; }
.proba-container { margin-top: 8px; }
.proba-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 13px; }
.proba-label { min-width: 60px; font-weight: 600; }
.proba-track { flex: 1; height: 20px; background: #e5e7eb; border-radius: 4px; overflow: hidden; }
.proba-fill { height: 100%; background: var(--accent); border-radius: 4px; transition: width 0.3s; }
.proba-value { min-width: 50px; text-align: right; color: var(--muted); }
.aspect-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.aspect-item { border: 1px solid var(--line); border-radius: 6px; padding: 6px 12px; font-size: 13px; background: #f9fafb; }
.event-grid { display: grid; gap: 10px; }
.event-card { border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #f9fafb; }
.event-type { font-weight: 700; color: var(--accent); font-size: 15px; margin-bottom: 8px; }
.slot-row { display: flex; gap: 8px; padding: 3px 0; font-size: 13px; }
.slot-name { color: var(--muted); min-width: 50px; font-weight: 600; }
.slot-value { color: var(--text); }
.topic-grid { margin-top: 8px; }
.topic-item { padding: 6px 0; font-size: 13px; border-bottom: 1px solid #f3f4f6; }
.topic-weight { color: var(--muted); font-size: 12px; margin-left: 8px; }
.summary-box { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 14px; font-size: 15px; line-height: 1.7; color: #166534; }
.note { font-size: 13px; color: var(--muted); font-style: italic; }
.empty-hint { color: var(--muted); font-size: 14px; padding: 20px 0; text-align: center; }
.error-box { background: #fef2f2; border: 1px solid #fecaca; border-radius: 6px; padding: 12px; color: #dc2626; font-size: 14px; }
.module-grid {  grid-template-columns: 1fr 1fr; gap: 8px; }
details > summary { cursor: pointer; color: var(--accent); font-size: 13px; padding: 4px 0; }
"""

def build_interface() -> gr.Blocks:
    """构建 Gradio 界面。"""

    with gr.Blocks(
        title="NLP 文本智能分析平台",
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="blue",
            neutral_hue="slate",
        ),
    ) as demo:

        with gr.Column(elem_classes="app-shell"):
            gr.HTML("""
            <header class="app-header">
                <div>
                    <p class="eyebrow">NLP 文本智能分析 传统方法 + LLM</p>
                    <h1>文本智能分析平台</h1>
                </div>
            </header>
            """)

            with gr.Row():
                with gr.Column(scale=1, min_width=380):
                    with gr.Group(elem_classes="control-section"):
                        gr.Markdown("### 输入与分析设置")

                        text_input = gr.Textbox(
                            label="输入文本",
                            placeholder="粘贴或输入需要分析的中文文本... 支持微博、新闻等网络文本。",
                            lines=6, max_lines=12,
                        )

                        with gr.Row():
                            load_sample_btn = gr.Button("加载示例", size="sm", scale=1)
                            clear_text_btn = gr.Button("清空", size="sm", scale=1)

                        mode_radio = gr.Radio(
                            choices=[
                                ("传统模式 - 规则/统计方法", "traditional"),
                                ("LLM 模式 - DeepSeek API", "llm"),
                            ],
                            value="traditional",
                            label="分析模式",
                        )

                        api_key_input = gr.Textbox(
                            label="DeepSeek API Key(LLM 模式需要)",
                            placeholder="sk-... 或设置环境变量 DEEPSEEK_API_KEY",
                            type="password", visible=False,
                        )

                        gr.Markdown("**选择分析模块**")
                        with gr.Group(elem_classes="module-grid"):
                            run_preprocess = gr.Checkbox(label="文本预处理", value=True)
                            run_keyword = gr.Checkbox(label="关键词提取", value=True)
                            run_ner = gr.Checkbox(label="命名实体识别", value=True)
                            run_sentiment = gr.Checkbox(label="情感分析", value=True)
                            run_classify = gr.Checkbox(label="文本分类", value=True)
                            run_event = gr.Checkbox(label="事件抽取", value=True)
                            run_topic = gr.Checkbox(label="主题发现", value=True)
                            run_summary = gr.Checkbox(label="自动摘要", value=True)

                        with gr.Accordion("高级设置", open=False):
                            keyword_method = gr.Radio(
                                choices=[("TextRank(单文档)", "textrank"),
                                         ("TF-IDF(需语料)", "tfidf")],
                                value="textrank", label="关键词方法",
                            )
                            keyword_top_n = gr.Slider(
                                minimum=5, maximum=30, value=10, step=1, label="关键词数量",
                            )
                            summary_sentences = gr.Slider(
                                minimum=1, maximum=5, value=3, step=1, label="摘要句子数",
                            )

                        with gr.Row():
                            run_btn = gr.Button("开始分析", variant="primary", scale=2)
                            clear_btn = gr.Button("清空结果", scale=1)

                with gr.Column(scale=2):
                    with gr.Tabs() as tabs:
                        with gr.TabItem("总览"):
                            overview_html = gr.HTML(value="")
                        with gr.TabItem("预处理"):
                            preprocess_html = gr.HTML(value="")
                        with gr.TabItem("关键词"):
                            keyword_html = gr.HTML(value="")
                        with gr.TabItem("命名实体"):
                            ner_html = gr.HTML(value="")
                        with gr.TabItem("情感分析"):
                            sentiment_html = gr.HTML(value="")
                        with gr.TabItem("文本分类"):
                            classification_html = gr.HTML(value="")
                        with gr.TabItem("事件抽取"):
                            event_html = gr.HTML(value="")
                        with gr.TabItem("主题发现"):
                            topic_html = gr.HTML(value="")
                        with gr.TabItem("自动摘要"):
                            summary_html = gr.HTML(value="")

                    with gr.Row():
                        export_txt_btn = gr.Button("导出分析报告(txt)", size="sm")
                        download_file = gr.File(label="下载结果", interactive=False, visible=True)
        # ---- 事件绑定 ----

        def on_mode_change(mode: str):
            return gr.update(visible=(mode == "llm"))

        mode_radio.change(
            fn=on_mode_change,
            inputs=[mode_radio],
            outputs=[api_key_input],
        )

        sample_texts = [
            "2026年6月12日，公安部召开专题新闻发布会。会上，北京市公安局刑侦总队政治处主任李小燕公布了在北京发生的两起典型案例。其中一起是400余名老年人遭健康养生诈骗。北京警方近期打掉一个专门针对老年人的诈骗团伙，抓获31名犯罪嫌疑人，涉及朝阳、顺义、平谷、密云4个区20家门店。",
            "#突发#！！某地暴雨太大了！！！详情见 http://xxx.com",
            "真的太喜欢了，这个产品功能不错，但是价格太离谱了！超级好用，强烈推荐！",
            "白鹿方否认更换编剧团队，白鹿工作室发布六连辟谣声明。网友：这次维权速度很快。",
            "某品牌因产品质量问题发布召回公告，涉及多款热门手机型号。",
        ]
        sample_idx = [0]

        def load_sample() -> str:
            idx = sample_idx[0]
            text = sample_texts[idx % len(sample_texts)]
            sample_idx[0] = (idx + 1) % len(sample_texts)
            return text

        load_sample_btn.click(fn=load_sample, outputs=[text_input])

        def clear_text() -> str:
            return ""

        clear_text_btn.click(fn=clear_text, outputs=[text_input])

        def run_analysis(
            text, mode, api_key, run_pre, run_kw, run_ner_b, run_sent,
            run_cls, run_ev, run_tpc, run_sum, kw_method, kw_n, sum_n,
        ):
            if not text or not text.strip():
                return (
                    "<div class='error-box'>请输入需要分析的文本。</div>",
                    "", "", "", "", "", "", "", "", ""
                )
            try:
                if mode == "traditional":
                    result = analyze_text_traditional(
                        text.strip(), run_preprocess=run_pre, run_keyword=run_kw,
                        run_ner=run_ner_b, run_sentiment=run_sent,
                        run_classify=run_cls, run_event=run_ev,
                        run_topic=run_tpc, run_summary=run_sum,
                        keyword_method=kw_method, keyword_top_n=kw_n,
                        summary_sentences=sum_n,
                    )
                else:
                    result = analyze_text_llm(
                        text.strip(), api_key=api_key,
                        run_ner=run_ner_b, run_sentiment=run_sent,
                        run_keyword=run_kw, run_classify=run_cls,
                        run_event=run_ev, run_summary=run_sum,
                    )
            except Exception as e:
                return (f"<div class='error-box'>分析失败: {str(e)}</div>", "", "", "", "", "", "", "", "", "")

            run_analysis._last_result = result
            run_analysis._last_mode = mode

            html = format_analysis_result(result, mode)

            def _safe(key: str) -> str:
                return html.get(key, "")

            # If there's an error (e.g. missing API key), show it in overview
            if "error" in result:
                err_msg = result.get("error", "未知错误")
                hint = result.get("hint", "")
                err_html = f"<div class='error-box'><strong>{err_msg}</strong>"
                if hint:
                    err_html += f"<br>{hint}"
                err_html += "</div>"
                overview = err_html
            else:
                overview_parts = []
                if "ner" in result:
                    n = result.get("ner", {})
                    ents = n.get("entities", [])
                    from collections import Counter
                    tc = Counter(e["type"] for e in ents)
                    overview_parts.append(f"<div class='result-section'><h4>命名实体</h4><p>{n.get('count', 0)} 个实体</p></div>")
                if "keyword" in result:
                    k = result.get("keyword", {})
                    kw_list = k.get("keywords", [])[:8]
                    chips = "".join(f"<span class='keyword-chip'>{kw['word']}</span>" for kw in kw_list)
                    overview_parts.append(f"<div class='result-section'><h4>关键词</h4><div class='keyword-cloud'>{chips}</div></div>")
                if "sentiment" in result:
                    s = result.get("sentiment", {})
                    emoji_map = {"正面": "positive", "中性": "neutral", "负面": "negative"}
                    overview_parts.append(f"<div class='result-section'><h4>情感</h4><p>{s.get('sentiment', '')} ({s.get('score', 0):+.2f})</p></div>")
                if "classification" in result:
                    c = result.get("classification", {})
                    overview_parts.append(f"<div class='result-section'><h4>分类</h4><p>{c.get('label', '')}</p></div>")
                if "event" in result:
                    e = result.get("event", {})
                    overview_parts.append(f"<div class='result-section'><h4>事件</h4><p>{e.get('count', 0)} 个事件被识别</p></div>")
                if "summary" in result:
                    s = result.get("summary", {})
                    overview_parts.append(f"<div class='result-section'><h4>摘要</h4><p>{s.get('text', '')[:80]}...</p></div>")

                overview = "".join(overview_parts) if overview_parts else "<div class='empty-hint'>请选择至少一个模块进行分析</div>"

            preprocess_content = _safe("preprocess") if mode != "llm" else "<div class='empty-hint'>LLM 模式下不进行文本预处理</div>"
            return (
                overview,
                preprocess_content, _safe("keyword"), _safe("ner"),
                _safe("sentiment"), _safe("classification"),
                _safe("event"), _safe("topic"), _safe("summary"),
            )

        run_analysis._last_result = {}
        run_analysis._last_mode = "traditional"

        def clear_results():
            run_analysis._last_result = {}
            return (gr.update(value=""),) * 9

        run_btn.click(
            fn=run_analysis,
            inputs=[
                text_input, mode_radio, api_key_input,
                run_preprocess, run_keyword, run_ner, run_sentiment,
                run_classify, run_event, run_topic, run_summary,
                keyword_method, keyword_top_n, summary_sentences,
            ],
            outputs=[
                overview_html, preprocess_html, keyword_html, ner_html,
                sentiment_html, classification_html,
                event_html, topic_html, summary_html,
            ],
        )

        clear_btn.click(
            fn=clear_results,
            inputs=None,
            outputs=[
                overview_html, preprocess_html, keyword_html, ner_html,
                sentiment_html, classification_html,
                event_html, topic_html, summary_html,
            ],
        )

        def export_report():
            result = getattr(run_analysis, "_last_result", {})
            mode = getattr(run_analysis, "_last_mode", "traditional")
            if not result:
                return None
            report = generate_full_report(result, mode)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8",
            )
            tmp.write(report)
            tmp.close()
            return tmp.name

        export_txt_btn.click(fn=export_report, outputs=[download_file])

    return demo


# ============================================
# 8. 启动
# ============================================


def main():
    """主入口。"""
    import argparse

    print("正在初始化 NLP 模块...")
    load_traditional_modules()

    parser = argparse.ArgumentParser(description="NLP 文本智能分析平台")
    parser.add_argument("--port", type=int, default=7865, help="服务端口 (default: 7865)")
    parser.add_argument("--share", action="store_true", help="生成公网分享链接")
    args = parser.parse_args()

    demo = build_interface()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()