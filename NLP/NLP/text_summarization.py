"""
NLP 自动摘要模块 — TextRank + TF-IDF 抽取式摘要

传统方法实现：
1. TextRank 句子排序 — 构建句子相似度图，PageRank 选重要句
2. TF-IDF 句子打分 — 基于关键词密度评分

Usage:
    from text_summarization import TextSummarizer
    ts = TextSummarizer(method="textrank")
    summary = ts.summarize("长文本...", num_sentences=3)
"""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from preprocessing import TextPreprocessor


# ═══════════════════════════════════════════════════════════
#  1. 句子分割
# ═══════════════════════════════════════════════════════════

def split_sentences(text: str) -> List[str]:
    """将中文文本分割为句子列表。

    按。！？；\n 分割，过滤过短句子。
    """
    # 先按句尾标点分割
    raw = re.split(r"[。！？\n]+", text)
    sentences = []
    for s in raw:
        s = s.strip()
        # 进一步按 ；分割
        parts = re.split(r"[；;]+", s)
        for p in parts:
            p = p.strip()
            # 过滤：至少包含 4 个中文字符
            if len(p) >= 4 and re.search(r"[\u4e00-\u9fff]", p):
                sentences.append(p)
    return sentences


# ═══════════════════════════════════════════════════════════
#  2. TextRank 摘要
# ═══════════════════════════════════════════════════════════

class TextRankSummarizer:
    """TextRank 抽取式摘要。

    构建句子相似度图，用 PageRank 给句子排序，选择 Top-N 句。

    Parameters
    ----------
    damping : float, default=0.85
    max_iter : int, default=100
    tol : float, default=1e-4
    similarity_threshold : float, default=0.0
    """

    def __init__(
        self,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-4,
        similarity_threshold: float = 0.0,
    ):
        self._pp = TextPreprocessor()
        self._damping = damping
        self._max_iter = max_iter
        self._tol = tol
        self._threshold = similarity_threshold

    def summarize(self, text: str, num_sentences: int = 3) -> List[str]:
        """从文本中提取关键句子形成摘要。

        Returns
        -------
        List[str] : 按原文顺序排列的摘要句子
        """
        sentences = split_sentences(text)
        if len(sentences) <= num_sentences:
            return sentences

        # 1. 对每个句子进行分词
        token_lists = []
        for sent in sentences:
            t = self._pp.process(sent)
            token_lists.append(t.split() if t else [])

        # 2. 构建 TF 向量
        vocab = {}
        for tokens in token_lists:
            for w in tokens:
                if w not in vocab:
                    vocab[w] = len(vocab)

        V = len(vocab)
        vectors = []
        for tokens in token_lists:
            vec = [0.0] * V
            for w in tokens:
                idx = vocab.get(w)
                if idx is not None:
                    vec[idx] += 1.0
            # L2 归一化
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            vectors.append(vec)

        # 3. 构建相似度矩阵
        n = len(sentences)
        sim_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._cosine_sim(vectors[i], vectors[j])
                if sim > self._threshold:
                    sim_matrix[i][j] = sim
                    sim_matrix[j][i] = sim

        # 4. PageRank
        scores = self._pagerank(sim_matrix)

        # 5. 选择 Top-N 句子（带多样性惩罚）
        ranked = sorted(
            [(i, scores[i]) for i in range(n)],
            key=lambda x: -x[1],
        )

        selected = []
        selected_indices = []

        for idx, score in ranked:
            if len(selected) >= num_sentences:
                break
            # 多样性检查：与已选句子的最大相似度
            if selected_indices:
                max_sim = max(
                    sim_matrix[idx][sel_idx] for sel_idx in selected_indices
                )
                if max_sim > 0.8:
                    continue  # 太相似，跳过
            selected_indices.append(idx)
            selected.append((idx, score))

        # 6. 按原文顺序返回
        selected.sort(key=lambda x: x[0])
        return [sentences[idx] for idx, _ in selected]

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        return dot  # 已 L2 归一化

    def _pagerank(self, adj: List[List[float]]) -> List[float]:
        """在句子相似度图上运行 PageRank。"""
        n = len(adj)
        if n == 0:
            return []
        if n == 1:
            return [1.0]

        scores = [1.0 / n] * n
        out_degree = [sum(row) for row in adj]
        has_dangling = any(d == 0 for d in out_degree)

        d = self._damping

        for _ in range(self._max_iter):
            prev = scores[:]

            dangling_contrib = 0.0
            if has_dangling:
                dangling_contrib = d * sum(
                    scores[k] for k, od in enumerate(out_degree) if od == 0
                ) / n

            for i in range(n):
                s = 0.0
                for j in range(n):
                    if adj[j][i] > 0 and out_degree[j] > 0:
                        s += (adj[j][i] / out_degree[j]) * prev[j]
                scores[i] = (1 - d) + d * s
                if has_dangling:
                    scores[i] += dangling_contrib

            diff = sum(abs(scores[i] - prev[i]) for i in range(n))
            if diff < self._tol:
                break

        return scores


# ═══════════════════════════════════════════════════════════
#  3. TF-IDF 摘要
# ═══════════════════════════════════════════════════════════

class TFIDFSummarizer:
    """基于 TF-IDF 关键词密度的抽取式摘要。

    计算每个句子的 TF-IDF 词权重之和，选择得分最高且多样性的句子。
    """

    def __init__(self):
        self._pp = TextPreprocessor()

    def summarize(self, text: str, num_sentences: int = 3) -> List[str]:
        sentences = split_sentences(text)
        if len(sentences) <= num_sentences:
            return sentences

        # 1. 分词
        doc_tokens = []
        sent_tokens_list = []
        for sent in sentences:
            t = self._pp.process(sent)
            tokens = t.split() if t else []
            doc_tokens.extend(tokens)
            sent_tokens_list.append(tokens)

        # 2. 计算 TF-IDF
        # 全局词频
        word_freq = Counter(doc_tokens)
        n_total = len(doc_tokens)

        # 文档频率（句子级别）
        sent_df = Counter()
        for tokens in sent_tokens_list:
            for w in set(tokens):
                sent_df[w] += 1

        n_sents = len(sentences)
        word_tfidf = {}
        for w, freq in word_freq.items():
            tf = freq / n_total if n_total > 0 else 0
            idf = math.log((n_sents + 1) / (sent_df.get(w, 0) + 1)) + 1.0
            word_tfidf[w] = tf * idf

        # 3. 句子评分
        sent_scores = []
        for i, tokens in enumerate(sent_tokens_list):
            score = sum(word_tfidf.get(w, 0.0) for w in tokens)
            # 位置奖励：前 20% 句子加 30% 权重
            if i < n_sents * 0.2:
                score *= 1.3
            # 长度惩罚：太短或太长降权
            sent_len = len(tokens)
            if sent_len < 5:
                score *= 0.5
            sent_scores.append(score)

        # 4. 排序 + 多样性选择
        ranked = sorted(
            [(i, sent_scores[i]) for i in range(n_sents)],
            key=lambda x: -x[1],
        )

        # 计算句子间 Jaccard 相似度
        def jaccard(t1, t2):
            s1, s2 = set(t1), set(t2)
            if not s1 or not s2:
                return 0.0
            return len(s1 & s2) / len(s1 | s2)

        selected = []
        for idx, score in ranked:
            if len(selected) >= num_sentences:
                break
            # 多样性
            is_dup = False
            for sel_idx in selected:
                if jaccard(sent_tokens_list[idx], sent_tokens_list[sel_idx]) > 0.6:
                    is_dup = True
                    break
            if not is_dup:
                selected.append(idx)

        selected.sort()
        return [sentences[i] for i in selected]


# ═══════════════════════════════════════════════════════════
#  4. 统一接口
# ═══════════════════════════════════════════════════════════

class TextSummarizer:
    """自动摘要统一接口。

    Parameters
    ----------
    method : str, default="textrank"
        "textrank" 或 "tfidf"
    **kwargs : 传递给具体方法
    """

    def __init__(self, method: str = "textrank", **kwargs):
        self._method = method.lower()
        if self._method == "textrank":
            self._summarizer = TextRankSummarizer(**{
                k: v for k, v in kwargs.items()
                if k in ("damping", "max_iter", "tol", "similarity_threshold")
            })
        elif self._method == "tfidf":
            self._summarizer = TFIDFSummarizer()
        else:
            raise ValueError(f"未知方法: {method}")

    def summarize(self, text: str, num_sentences: int = 3) -> List[str]:
        """生成摘要。"""
        return self._summarizer.summarize(text, num_sentences)

    def summarize_text(self, text: str, num_sentences: int = 3) -> str:
        """生成摘要文本（用句号连接）。"""
        sents = self.summarize(text, num_sentences)
        return "。".join(sents) + "。"


# ═══════════════════════════════════════════════════════════
#  Demo
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("自动摘要 Demo")
    print("=" * 60)

    ts_tr = TextSummarizer(method="textrank")
    ts_tf = TextSummarizer(method="tfidf")

    # 测试用例
    test_cases = [
        "2026年6月12日，公安部召开专题新闻发布会。会上，北京市公安局刑侦总队政治处主任李小燕公布了在北京发生的两起典型案例。其中一起是400余名老年人遭健康养生诈骗。北京警方近期打掉一个专门针对老年人的诈骗团伙，抓获31名犯罪嫌疑人，涉及朝阳、顺义、平谷、密云4个区20家门店。店员以提供免费按摩、低价足疗券等方式将老年人吸引至店内，按摩过程中通过聊天锁定一些子女不在身边、经济条件较好的老年人，以其身体状况不好为由，引荐所谓的专家做免费体检，并虚构各种病症，称如不及时治疗将危及生命，诱骗充值高额治疗费用。共计400余名老年人被骗，涉案金额3000万余元。",
        "#突发#！！某地暴雨太大了！！！详情见 http://xxx.com",
        "今天天气真好，心情特别愉快！准备出去走走。",
    ]

    for text in test_cases:
        display = text[:50] + "..." if len(text) > 50 else text
        print(f"\n原文: {display}")

        summary_tr = ts_tr.summarize(text, num_sentences=2)
        summary_tf = ts_tf.summarize(text, num_sentences=2)

        if summary_tr:
            print(f"  TextRank: {'  '.join(summary_tr)}")
        else:
            print(f"  TextRank: (文本过短)")

        if summary_tf:
            print(f"  TF-IDF:   {'  '.join(summary_tf)}")
        else:
            print(f"  TF-IDF:   (文本过短)")

    # 微博数据
    print("\n" + "-" * 60)
    print("微博数据摘要")
    print("-" * 60)
    jsonl_path = (
        "E:\\6+7\\SJTU\\大三下\\信息内容安全\\大作业\\数据"
        "\\20260612_141245\\20260612_141245\\representative_posts.jsonl"
    )
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            data = json.loads(line)
            topic = data.get("topic", {}).get("word", "未知")
            text = data.get("selection", {}).get("selected_post", {}).get("text", "")
            if text:
                summary = ts_tr.summarize_text(text, num_sentences=2)
                print(f"\n[{topic}]")
                print(f"  原文({len(text)}字): {text[:40]}...")
                print(f"  摘要: {summary[:80]}...")
