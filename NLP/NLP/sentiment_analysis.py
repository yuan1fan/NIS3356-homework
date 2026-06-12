"""
NLP 情感分析模块 — 基于情感词典 + 规则的传统方法

适用于无标注数据的场景，通过情感词典匹配 + 否定词/程度副词/转折规则
计算文本的情感倾向与分数。

支持：
- 三分类：正面 / 中性 / 负面
- 细粒度：愤怒、悲伤、焦虑、喜悦、惊讶、厌恶
- 方面级：对文本中不同对象（功能、价格等）分别判断情感

Usage:
    from sentiment_analysis import SentimentAnalyzer
    sa = SentimentAnalyzer()
    result = sa.analyze("这个产品功能不错，但是价格太离谱了")
    # {"sentiment": "负面", "score": -0.42, "details": {...}}
"""

import re
import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

from preprocessing import TextPreprocessor


# ═══════════════════════════════════════════════════════════
#  1. 情感词典
# ═══════════════════════════════════════════════════════════

class SentimentDict:
    """内置中文情感词典，包含正负面词、否定词、程度副词、转折词、表情符号。"""

    # ── 正面词 ──
    POSITIVE = frozenset({
        "好","棒","赞","优秀","漂亮","美丽","好看","好吃","好玩",
        "喜欢","爱","开心","高兴","快乐","幸福","感动","精彩",
        "完美","厉害","牛","强","不错","满意","期待","支持",
        "希望","恭喜","祝贺","感谢","谢谢","伟大","光荣","正确",
        "进步","成功","胜利","健康","安全","先进","丰富","繁荣",
        "美好","善良","真诚","热情","积极","乐观","开朗","温柔",
        "体贴","细心","耐心","勤奋","努力","奋斗","拼搏","勇敢",
        "坚强","镇定","冷静","聪明","智慧","幽默","风趣","可爱",
        "萌","帅","美","酷","时尚","高档","大气","实用","方便",
        "快捷","高效","一流","领先","创新","突破","奇迹","惊艳",
        "经典","传奇","称赞","赞扬","表扬","赞美","赞赏","佩服",
        "崇拜","尊敬","敬重","敬佩","信赖","信任","可靠","稳定",
        "出色","杰出","卓越","非凡","精彩纷呈","激动人心","令人振奋",
        "赏心悦目","心旷神怡","无与伦比","叹为观止","赞不绝口",
        "温柔","暖心","贴心","用心","良心","高质量","高品质",
        "福音","诚意","良心","靠谱","超值","划算","大爱","绝了",
        "宝藏","神仙","心动","种草","回购","好用","细腻","清爽","愉快","舒畅","真好","挺好",
    })

    # ── 负面词 ──
    NEGATIVE = frozenset({
        "差","坏","烂","垃圾","恶心","讨厌","烦","无聊","坑",
        "黑","假","骗","虚假","伪造","糟糕","失败","错误","愚蠢",
        "笨","傻","弱","差劲","可恶","可恨","过分","无耻","卑鄙",
        "丑陋","黑暗","恐怖","可怕","危险","紧张","焦虑","担心",
        "害怕","恐惧","悲伤","伤心","难过","痛苦","悲哀","绝望",
        "愤怒","生气","恼火","气愤","恨","厌恶","鄙视","看不起",
        "嘲笑","讽刺","挖苦","攻击","谩骂","辱骂","侮辱","诽谤",
        "造谣","谣言","欺骗","欺诈","诈骗","陷阱","漏洞","问题",
        "缺陷","毛病","故障","事故","灾难","危机","风险","损失",
        "亏损","下跌","下降","减少","恶化","落后","退步","停滞",
        "衰退","不满","失望","遗憾","可惜","痛心","心疼","担忧",
        "忧虑","惶恐","不安","尴尬","难堪","丢脸","耻辱","羞愧",
        "自责","后悔","懊悔","悔恨","埋怨","抱怨","吐槽","批评",
        "批判","指责","谴责","痛斥","曝光","被坑","被宰","被割",
        "贵","贵得离谱","天价","乱收费","霸王条款","偷工减料",
        "粗制滥造","粗心","离谱","扯淡","忽悠","坑爹","无语","醉了","坑人","不值","卡顿","太差","不太好","不太好用","太贵","太差劲","太糟糕","敷衍","冷落","冷淡","粗暴","恶劣",
        "低效","落后","陈旧","腐烂","腐败","贪污","受贿",
    })

    # ── 程度副词 ──
    DEGREE = {
        "很": 1.5, "非常": 2.0, "太": 2.0, "极": 2.5, "极其": 2.5,
        "极度": 2.5, "十分": 2.0, "特别": 2.0, "尤其": 1.8,
        "比较": 1.3, "相当": 1.8, "挺": 1.3, "有点": 0.6,
        "有些": 0.6, "稍微": 0.5, "略微": 0.5, "更加": 1.8,
        "更": 1.5, "最": 2.5, "最为": 2.5, "越": 1.5, "愈发": 1.8,
        "颇": 1.5, "甚": 2.0, "极为": 2.5, "万分": 2.5,
        "超级": 2.5, "无比": 2.5, "格外": 2.0, "分外": 2.0,
        "绝": 2.0, "巨": 2.0, "超": 2.0, "老": 1.5,
    }

    # ── 否定词 ──
    NEGATION = frozenset({
        "不","没","别","无","未","莫","勿","毋","甭",
        "不用","不要","没有","不是","不会","不能","不行",
        "不可","不必","不好","不对","不该","不许","禁止","再也","再也不",
    })

    # ── 转折词 ──
    CONJUNCTION_ADVERSATIVE = frozenset({
        "但是","但","可是","然而","不过","却","虽然","尽管",
        "虽说","但","可","然而","只是","不过","唯独",
    })

    # ── 递进词 ──
    CONJUNCTION_PROGRESSIVE = frozenset({
        "而且","并且","况且","何况","甚至","更","还",
    })

    # ── 情感表情 ──
    EMOJI_SENTIMENT = {
        # 正面
        "\U0001f600": 2.0, "\U0001f601": 2.0, "\U0001f602": 2.0,
        "\U0001f604": 2.0, "\U0001f60a": 1.5, "\U0001f60d": 2.0,
        "\U0001f618": 2.0, "\U0001f61c": 1.0, "\U0001f60e": 1.5,
        "\U0001f44d": 1.0, "\U0001f44f": 1.0, "\U0001f44c": 1.0,
        "\U0001f4aa": 1.0, "\U0001f389": 2.0, "\U0001f38a": 2.0,
        "\U00002764": 2.0, "\U0001f49c": 1.5,
        # 负面
        "\U0001f622": -2.0, "\U0001f62d": -2.0, "\U0001f621": -2.5,
        "\U0001f624": -2.0, "\U0001f620": -2.0, "\U0001f616": -1.5,
        "\U0001f61e": -1.5, "\U0001f614": -1.5, "\U0001f629": -1.5,
        "\U0001f630": -2.0, "\U0001f631": -2.0, "\U0001f4a9": -2.0,
        "\U0001f44e": -1.5, "\U0001f494": -2.0, "\U0001f4a2": -2.0,
    }

    # ── 细粒度情感词映射 ──
    FINE_GRAINED = {
        "喜悦": frozenset({"开心","高兴","快乐","幸福","喜悦","欢快",
                           "欣喜","欢欣","兴高采烈","喜笑颜开"}),
        "愤怒": frozenset({"愤怒","生气","恼火","气愤","怒","怒火",
                           "暴怒","火大","气死"}),
        "悲伤": frozenset({"悲伤","伤心","难过","痛苦","悲哀","哀伤",
                           "悲痛","心碎","忧伤","沮丧"}),
        "焦虑": frozenset({"焦虑","担心","忧虑","惶恐","不安","紧张",
                           "害怕","恐惧","担忧"}),
        "惊讶": frozenset({"惊讶","吃惊","震惊","诧异","意外","惊呆",
                           "难以置信","不可思议"}),
        "厌恶": frozenset({"厌恶","讨厌","恶心","反感","憎恶","鄙视",
                           "看不起","嫌弃","厌烦"}),
    }

    # ── 方面级分析关键词 ──
    ASPECT_KEYWORDS = {
        "功能": frozenset({"功能","性能","配置","参数","规格","速度",
                          "反应","流畅"}),
        "价格": frozenset({"价格","价钱","价位","定价","售价","花费",
                          "费用","成本","性价比","折扣","优惠"}),
        "质量": frozenset({"质量","品质","材质","做工","质感","手感",
                          "耐用","寿命"}),
        "服务": frozenset({"服务","售后","客服","态度","体验","感受",
                          "物流","配送","发货"}),
        "外观": frozenset({"外观","颜值","设计","造型","颜色","款式",
                          "样式","外表","样子"}),
        "口感": frozenset({"口感","味道","口味","美味","好吃","难吃",
                          "香甜","鲜美"}),
    }


# ═══════════════════════════════════════════════════════════
#  2. 情感分析器
# ═══════════════════════════════════════════════════════════

class SentimentAnalyzer:
    """基于情感词典 + 规则的中文情感分析器。

    使用词典匹配、否定词翻转、程度副词加权、转折/递进规则
    计算文本的整体情感倾向与分数。
    """

    def __init__(self):
        self._pp = TextPreprocessor()
        self._dict = SentimentDict()

    # ── 主入口 ────────────────────────────────────────────

    def analyze(self, text: str) -> Dict:
        """分析文本情感。

        Returns
        -------
        dict: {
            "sentiment": "正面" | "中性" | "负面",
            "score": float,           # -1 ~ 1 范围
            "positive_words": [...],
            "negative_words": [...],
            "details": {"clause_scores": [...], ...},
            "fine_grained": {...},    # 细粒度分析
        }
        """
        if not text or not text.strip():
            return {"sentiment": "中性", "score": 0.0}

        # 预处理：分词并保留原始 token 序列
        tokens = self._tokenize(text)
        if not tokens:
            return {"sentiment": "中性", "score": 0.0}

        # 分句
        clauses = self._split_clauses(text)

        # 对每个分句计算情感
        clause_results = []
        total_score = 0.0
        weight_sum = 0.0
        pos_words = []
        neg_words = []

        has_adversative = False
        for clause in clauses:
            result = self._analyze_clause(clause)
            clause_results.append(result)

            # 转折处理：但后面的分句权重更高
            weight = 1.0
            if "但是" in clause or "但" in clause:
                has_adversative = True
            if has_adversative:
                clause_results[-1]["weight"] = 2.0
                weight = 2.0

            total_score += result["score"] * weight
            weight_sum += weight
            pos_words.extend(result["positive_words"])
            neg_words.extend(result["negative_words"])

        avg_score = total_score / weight_sum if weight_sum > 0 else 0.0

        # 表情符号
        emoji_score = self._analyze_emojis(text)
        avg_score += emoji_score * 0.3  # 表情符号权重

        # 归一化到 -1 ~ 1
        avg_score = max(-1.0, min(1.0, avg_score))

        # 分类
        if avg_score > 0.1:
            sentiment = "正面"
        elif avg_score < -0.1:
            sentiment = "负面"
        else:
            sentiment = "中性"

        # 细粒度
        fine_grained = self._fine_grained_analysis(tokens)

        # 去重
        pos_words = list(dict.fromkeys(pos_words))
        neg_words = list(dict.fromkeys(neg_words))

        return {
            "sentiment": sentiment,
            "score": round(avg_score, 4),
            "positive_words": pos_words[:10],
            "negative_words": neg_words[:10],
            "fine_grained": fine_grained,
            "details": {
                "num_clauses": len(clauses),
                "clause_scores": [r["score"] for r in clause_results],
            },
        }

    def analyze_aspects(self, text: str) -> Dict:
        """方面级情感分析。

        识别文本中不同方面（功能、价格、质量等）的情感倾向。

        Returns
        -------
        dict: {
            "aspects": [{"aspect": "价格", "sentiment": "负面",
                         "score": -0.8, "evidence": "价格太离谱"}],
            "overall": "偏负面",
        }
        """
        result = self.analyze(text)
        tokens = self._tokenize(text)

        aspects = []
        for aspect_name, keywords in self._dict.ASPECT_KEYWORDS.items():
            # 找到该方面的关键词在文本中的位置
            for match in re.finditer("|".join(re.escape(k) for k in keywords), text):
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end]
                clause_r = self._analyze_clause(context)
                if abs(clause_r["score"]) > 0.05:
                    if clause_r["score"] > 0.1:
                        as_sent = "正面"
                    elif clause_r["score"] < -0.1:
                        as_sent = "负面"
                    else:
                        as_sent = "中性"
                    aspects.append({
                        "aspect": aspect_name,
                        "sentiment": as_sent,
                        "score": round(clause_r["score"], 4),
                        "evidence": context.strip()[:50],
                    })
                    break  # 只取第一个匹配

        # 去重
        seen_aspects = set()
        unique_aspects = []
        for a in aspects:
            if a["aspect"] not in seen_aspects:
                seen_aspects.add(a["aspect"])
                unique_aspects.append(a)

        overall = result["sentiment"]
        # 如果有方面分析，整体调整为"偏X"
        if unique_aspects:
            neg_count = sum(1 for a in unique_aspects if a["sentiment"] == "负面")
            pos_count = sum(1 for a in unique_aspects if a["sentiment"] == "正面")
            if neg_count > pos_count:
                overall = "偏负面"
            elif pos_count > neg_count:
                overall = "偏正面"
            else:
                overall = "混合"

        return {
            "aspects": unique_aspects,
            "overall": overall,
            "overall_score": result["score"],
        }

    # ── 内部方法 ────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        """分词（保留所有词包括停用词，供否定/程度检测）。"""
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"@[\w\-_]+", "", text)
        text = re.sub(r"#([^#]+)#", r"\1", text)
        # 去除表情符号
        text = re.sub(
            "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
            "\U0001F900-\U0001F9FF\U00002702-\U000027B0"
            "\U00002600-\U000027BF\U0000FE00-\U0000FE0F]+",
            "", text)
        import jieba
        return list(jieba.cut(text))

    def _split_clauses(self, text: str) -> List[str]:
        """按标点分句。"""
        clauses = re.split(r"[。！？！\?；;，,\n]+", text)
        return [c.strip() for c in clauses if c.strip()]

    def _analyze_clause(self, clause: str) -> Dict:
        """分析单个分句的情感。"""
        tokens = self._tokenize(clause)
        if not tokens:
            return {"score": 0.0, "positive_words": [], "negative_words": []}

        score = 0.0
        pos_words_found = []
        neg_words_found = []
        negated = False

        for i, token in enumerate(tokens):
            # 检查否定词（前一个词）
            if i > 0 and tokens[i - 1] in self._dict.NEGATION:
                negated = True
            else:
                negated = False

            # 程度副词（前一个词）
            degree = 1.0
            if i > 0 and tokens[i - 1] in self._dict.DEGREE:
                degree = self._dict.DEGREE[tokens[i - 1]]

            # 正面词
            if token in self._dict.POSITIVE:
                word_score = 1.0 * degree * (-1.0 if negated else 1.0)
                score += word_score
                if word_score > 0:
                    pos_words_found.append(token)
                else:
                    neg_words_found.append(token)

            # 负面词
            elif token in self._dict.NEGATIVE:
                word_score = -1.0 * degree * (-1.0 if negated else 1.0)
                score += word_score
                if word_score < 0:
                    neg_words_found.append(token)
                else:
                    pos_words_found.append(token)

        # 极端值截断
        score = max(-5.0, min(5.0, score))

        # 标准化
        if len(tokens) > 0:
            score = score / math.sqrt(len(tokens))

        score = max(-1.0, min(1.0, score))

        return {
            "score": score,
            "positive_words": list(set(pos_words_found)),
            "negative_words": list(set(neg_words_found)),
        }

    def _analyze_emojis(self, text: str) -> float:
        """分析文本中的表情符号情感。"""
        score = 0.0
        for char in text:
            if char in self._dict.EMOJI_SENTIMENT:
                score += self._dict.EMOJI_SENTIMENT[char]
        if len(text) > 0:
            score = max(-1.0, min(1.0, score / max(1, len(text)) * 5))
        return score

    def _fine_grained_analysis(self, tokens: List[str]) -> Dict[str, float]:
        """细粒度情感分析。"""
        results = {}
        for emotion, words in self._dict.FINE_GRAINED.items():
            count = sum(1 for t in tokens if t in words)
            if count > 0:
                results[emotion] = count
        return results


# ═══════════════════════════════════════════════════════════
#  3. 基于朴素贝叶斯的情感分类器（需标注数据）
# ═══════════════════════════════════════════════════════════

class SupervisedSentimentAnalyzer:
    """基于朴素贝叶斯的情感分类器包装。

    需要已标注的情感数据（正面/中性/负面）来训练。
    使用 TextClassifier 作为底层引擎。
    """

    def __init__(self):
        from text_classification import TextClassifier
        self._clf = TextClassifier(method="naive_bayes")

    def fit(self, texts: List[str], labels: List[str]):
        """训练情感分类模型。"""
        self._clf.fit(texts, labels)

    def analyze(self, text: str) -> Dict:
        """预测情感。"""
        pred = self._clf.predict(text)
        proba = self._clf.predict_proba(text)
        max_proba = max(proba.values()) if proba else 0.0
        return {
            "sentiment": pred,
            "confidence": round(max_proba, 4),
            "probabilities": proba,
        }


# ═══════════════════════════════════════════════════════════
#  4. Demo
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("情感分析 Demo (情感词典法)")
    print("=" * 60)

    sa = SentimentAnalyzer()

    test_cases = [
        "这个产品功能不错，但是价格太离谱了",
        "今天天气真好，心情特别愉快！",
        "太让人愤怒了，这完全是在欺骗消费者！",
        "一般般吧，没什么特别的感觉。",
        "😡😡😡服务质量太差了，再也不会来了！",
        "#突发#！！某地暴雨太大了！！！详情见 http://xxx.com",
        "真的太喜欢了😍😍😍，超级好用，强烈推荐！",
    ]

    for text in test_cases:
        result = sa.analyze(text)
        emoji_mark = {"正面": "🟢", "中性": "🟡", "负面": "🔴"}
        mark = emoji_mark.get(result["sentiment"], "⚪")
        print(f"\n{mark} | {result['sentiment']:>4s} | 分数:{result['score']:+.2f}")
        print(f"  原文: {text}")
        if result["positive_words"]:
            print(f"  正面词: {', '.join(result['positive_words'][:5])}")
        if result["negative_words"]:
            print(f"  负面词: {', '.join(result['negative_words'][:5])}")
        if result["fine_grained"]:
            print(f"  细粒度: {result['fine_grained']}")
        if result["details"]["num_clauses"] > 1:
            print(f"  分句: {result['details']['clause_scores']}")

    # 方面级分析
    print("\n" + "-" * 60)
    print("方面级情感分析")
    print("-" * 60)
    aspect_text = "这个产品功能不错，但是价格太离谱了。质量倒是挺好，就是外观一般。"
    print(f"原文: {aspect_text}")
    ar = sa.analyze_aspects(aspect_text)
    print(f"整体: {ar['overall']} (分数: {ar['overall_score']:+.2f})")
    for a in ar["aspects"]:
        print(f"  [{a['aspect']}] {a['sentiment']} (分数:{a['score']:+.2f})")
        print(f"    上下文: {a['evidence']}")

    # 微博数据
    print("\n" + "-" * 60)
    print("微博数据情感分析")
    print("-" * 60)
    jsonl_path = (
        "E:\\6+7\\SJTU\\大三下\\信息内容安全\\大作业\\数据"
        "\\20260612_141245\\20260612_141245\\representative_posts.jsonl"
    )
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            data = json.loads(line)
            text = data.get("selection", {}).get("selected_post", {}).get("text", "")
            if text:
                samples.append((data.get("topic", {}).get("word", "未知"), text))

    for topic, text in samples:
        result = sa.analyze(text)
        display = text[:60] + "..." if len(text) > 60 else text
        print(f"\n[{topic}]")
        print(f"  文本: {display}")
        print(f"  情感: {result['sentiment']:>4s} | 分数:{result['score']:+.2f}")
        if result["positive_words"]:
            print(f"  正面词: {', '.join(result['positive_words'][:5])}")
        if result["negative_words"]:
            print(f"  负面词: {', '.join(result['negative_words'][:5])}")
