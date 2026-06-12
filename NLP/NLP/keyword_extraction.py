"""
NLP 关键词提取模块

传统方法实现：
1. TF-IDF：基于词频与逆文档频率，反映词对文档的代表性
2. TextRank：基于图排序算法，利用词共现关系计算重要性

Usage:
    from keyword_extraction import TFIDFExtractor, TextRankExtractor, KeywordExtractor

    # TF-IDF (需要语料)
    extractor = TFIDFExtractor()
    extractor.fit(corpus_texts)
    keywords = extractor.extract("你的文本")

    # TextRank (单篇即可)
    extractor = TextRankExtractor()
    keywords = extractor.extract("你的文本")
"""

import math
import re
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Union

from preprocessing import TextPreprocessor


# ─────────────────────────────────────────────────────────
#  工具函数
# ─────────────────────────────────────────────────────────

def _ensure_preprocessor(preprocessor: Optional[TextPreprocessor] = None) -> TextPreprocessor:
    """获取或创建预处理器实例。"""
    if preprocessor is None:
        return TextPreprocessor()
    return preprocessor


def _tokenize(text: str, preprocessor: Optional[TextPreprocessor] = None) -> List[str]:
    """预处理并分词，返回 token 列表。"""
    pp = _ensure_preprocessor(preprocessor)
    processed = pp.process(text)
    if not processed:
        return []
    return processed.split()


# ─────────────────────────────────────────────────────────
#  1. TF-IDF 关键词提取
# ─────────────────────────────────────────────────────────

class TFIDFExtractor:
    """TF-IDF 关键词提取器

    基于词频 (TF) 和逆文档频率 (IDF) 计算每个词对文档的代表性。
    需先在语料上 `fit()` 计算 IDF，再对单篇文档 `extract()`。

    Parameters
    ----------
    preprocessor : TextPreprocessor, optional
        文本预处理器实例
    max_features : int, default=5000
        保留的最大特征词数（按文档频率排序）
    smooth_idf : bool, default=True
        是否使用平滑 IDF (log((N+1)/(df+1)) + 1)
    sublinear_tf : bool, default=False
        是否对 TF 取 log(1 + tf) 做次线性缩放
    """

    def __init__(
        self,
        preprocessor: Optional[TextPreprocessor] = None,
        max_features: int = 5000,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
    ):
        self._pp = _ensure_preprocessor(preprocessor)
        self._max_features = max_features
        self._smooth_idf = smooth_idf
        self._sublinear_tf = sublinear_tf

        # 以下在 fit() 后填充
        self._idf: Dict[str, float] = {}
        self._vocabulary: Dict[str, int] = {}       # word → index
        self._n_docs: int = 0
        self._fitted: bool = False

    # ── 训练 ────────────────────────────────────────────

    def fit(self, documents: List[str]) -> "TFIDFExtractor":
        """在语料上训练 IDF 值。

        Parameters
        ----------
        documents : List[str]
            原始文本列表（预处理在内部完成）
        """
        # 1. 预处理 & 分词
        doc_token_lists: List[List[str]] = []
        for doc in documents:
            tokens = _tokenize(doc, self._pp)
            if tokens:
                doc_token_lists.append(tokens)

        self._n_docs = len(doc_token_lists)
        if self._n_docs == 0:
            self._fitted = True
            return self

        # 2. 文档频率 (DF)：词出现在多少篇文档中
        df_counter: Dict[str, int] = defaultdict(int)
        for tokens in doc_token_lists:
            unique_words = set(tokens)
            for w in unique_words:
                df_counter[w] += 1

        # 3. 按文档频率排序，截取 max_features
        sorted_words = sorted(df_counter.items(), key=lambda x: -x[1])
        selected = sorted_words[:self._max_features]
        self._vocabulary = {w: idx for idx, (w, _) in enumerate(selected)}
        vocab_df = {w: df_counter[w] for w in self._vocabulary}

        # 4. 计算 IDF
        n = self._n_docs
        for w, df in vocab_df.items():
            if self._smooth_idf:
                idf = math.log((n + 1) / (df + 1)) + 1.0
            else:
                idf = math.log(n / df) if df > 0 else 0.0
            self._idf[w] = idf

        self._fitted = True
        return self

    # ── 提取 ────────────────────────────────────────────

    def extract(
        self,
        text: str,
        top_n: int = 10,
        with_scores: bool = False,
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """从单篇文档中提取关键词。

        Parameters
        ----------
        text : str
            原始文本
        top_n : int, default=10
            返回前 N 个关键词
        with_scores : bool, default=False
            是否同时返回分数

        Returns
        -------
        List[str] 或 List[Tuple[str, float]]
        """
        tokens = _tokenize(text, self._pp)
        if not tokens:
            return [] if not with_scores else []

        # 计算当前文档的 TF
        n_words = len(tokens)
        tf_counter = Counter(tokens)

        # 计算每个词的 TF-IDF 分数
        scores: Dict[str, float] = {}
        for w, tf_raw in tf_counter.items():
            if w not in self._idf:
                continue
            tf = math.log(1 + tf_raw) if self._sublinear_tf else tf_raw / n_words
            scores[w] = tf * self._idf[w]

        if not scores:
            return [] if not with_scores else []

        # 排序取 top_n
        sorted_keywords = sorted(scores.items(), key=lambda x: -x[1])
        top = sorted_keywords[:top_n]

        if with_scores:
            return top
        return [w for w, _ in top]

    # ── 属性 ────────────────────────────────────────────

    @property
    def vocabulary(self) -> Dict[str, int]:
        """词表 {词: 索引}"""
        return self._vocabulary.copy()

    @property
    def idf(self) -> Dict[str, float]:
        """IDF 值 {词: idf}"""
        return self._idf.copy()

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ─────────────────────────────────────────────────────────
#  2. TextRank 关键词提取
# ─────────────────────────────────────────────────────────

class TextRankExtractor:
    """TextRank 关键词提取器

    基于图排序算法，利用词与词之间的共现关系构建图，
    通过迭代传播计算节点（词）的重要性，无需语料即可提取关键词。
    算法思路类似 PageRank。

    Parameters
    ----------
    preprocessor : TextPreprocessor, optional
    window : int, default=2
        共现窗口大小（词与前后 window 个词之间建立边）
    damping : float, default=0.85
        阻尼系数
    max_iter : int, default=100
        最大迭代次数
    tol : float, default=1e-4
        收敛判据：所有节点分数变化之和小于 tol 即停止
    """

    def __init__(
        self,
        preprocessor: Optional[TextPreprocessor] = None,
        window: int = 3,
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-4,
    ):
        self._pp = _ensure_preprocessor(preprocessor)
        self._window = window
        self._damping = damping
        self._max_iter = max_iter
        self._tol = tol

    # ── 提取 ────────────────────────────────────────────

    def extract(
        self,
        text: str,
        top_n: int = 10,
        with_scores: bool = False,
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """从单篇文本中提取关键词。

        Parameters
        ----------
        text : str
            原始文本
        top_n : int, default=10
        with_scores : bool, default=False

        Returns
        -------
        List[str] 或 List[Tuple[str, float]]
        """
        tokens = _tokenize(text, self._pp)
        if not tokens:
            return [] if not with_scores else []

        # 去重保留顺序（TextRank 用唯一词作为节点）
        seen = set()
        unique_tokens = []
        for w in tokens:
            if w not in seen:
                seen.add(w)
                unique_tokens.append(w)

        if len(unique_tokens) < 2:
            # 不足 2 个词，直接返回
            if with_scores:
                return [(w, 1.0) for w in unique_tokens[:top_n]]
            return unique_tokens[:top_n]

        # 1. 构建共现图
        word_to_idx = {w: i for i, w in enumerate(unique_tokens)}
        n = len(unique_tokens)
        # 邻接矩阵 (对称, 加权)
        adj: List[List[float]] = [[0.0] * n for _ in range(n)]

        # 遍历原始 token 序列（含重复），构建共现边
        for i, w_i in enumerate(tokens):
            idx_i = word_to_idx[w_i]
            # 窗口范围 [i+1, i+window]
            for j in range(i + 1, min(i + 1 + self._window, len(tokens))):
                w_j = tokens[j]
                idx_j = word_to_idx[w_j]
                if idx_i == idx_j:
                    continue
                adj[idx_i][idx_j] += 1.0
                adj[idx_j][idx_i] += 1.0

        # 2. 运行 PageRank
        scores = self._pagerank(adj)

        # 3. 按分数排序
        sorted_indices = sorted(scores.keys(), key=lambda i: -scores[i])
        top_indices = sorted_indices[:top_n]

        if with_scores:
            return [(unique_tokens[i], scores[i]) for i in top_indices]
        return [unique_tokens[i] for i in top_indices]

    # ── PageRank 核心 ───────────────────────────────────

    def _pagerank(self, adj: List[List[float]]) -> Dict[int, float]:
        """在加权无向图上运行 PageRank 算法。"""
        n = len(adj)
        if n == 0:
            return {}
        if n == 1:
            return {0: 1.0}

        # 初始化：均匀分布
        scores = [1.0 / n] * n

        # 出度（从某节点出发的边权和）
        out_degree = [sum(row) for row in adj]
        # 处理零出度节点：视为与所有节点相连
        has_dangling = any(d == 0 for d in out_degree)

        d = self._damping
        dangling_contrib = 0.0

        for _ in range(self._max_iter):
            prev = scores[:]

            # 如果存在悬挂节点，它们贡献给所有节点
            if has_dangling:
                dangling_contrib = d * sum(scores[k] for k, od in enumerate(out_degree) if od == 0) / n

            for i in range(n):
                s = 0.0
                for j in range(n):
                    if adj[j][i] > 0 and out_degree[j] > 0:
                        s += (adj[j][i] / out_degree[j]) * prev[j]

                scores[i] = (1 - d) + d * s
                if has_dangling:
                    scores[i] += dangling_contrib

            # 收敛判断
            diff = sum(abs(scores[i] - prev[i]) for i in range(n))
            if diff < self._tol:
                break

        return {i: s for i, s in enumerate(scores)}


# ─────────────────────────────────────────────────────────
#  3. 统一接口
# ─────────────────────────────────────────────────────────

class KeywordExtractor:
    """统一关键词提取接口

    封装 TF-IDF 和 TextRank 两种方法，通过 `method` 参数切换。

    Parameters
    ----------
    method : str, default='tfidf'
        'tfidf' 或 'textrank'
    **kwargs
        传递给具体提取器的参数
    """

    def __init__(self, method: str = "tfidf", **kwargs):
        self._method = method.lower()
        if self._method == "tfidf":
            self._extractor = TFIDFExtractor(**kwargs)
        elif self._method == "textrank":
            self._extractor = TextRankExtractor(**kwargs)
        else:
            raise ValueError(f"未知方法: {method}，可选 'tfidf' 或 'textrank'")

    def fit(self, documents: List[str]) -> "KeywordExtractor":
        """训练（仅 TF-IDF 需要）。"""
        if self._method == "tfidf":
            self._extractor.fit(documents)
        # TextRank 无需训练
        return self

    def extract(
        self,
        text: str,
        top_n: int = 10,
        with_scores: bool = False,
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """提取关键词。"""
        return self._extractor.extract(text, top_n=top_n, with_scores=with_scores)


# ─────────────────────────────────────────────────────────
#  Demo
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("关键词提取 Demo")
    print("=" * 60)

    # 加载测试数据
    jsonl_path = (
        "E:\\6+7\\SJTU\\大三下\\信息内容安全\\大作业\\数据"
        "\\20260612_141245\\20260612_141245\\representative_posts.jsonl"
    )
    corpus_texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data.get("selection", {}).get("selected_post", {}).get("text", "")
            if text:
                corpus_texts.append(text)
    # 保存完整 JSON 数据用于展示标题
    corpus_data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            corpus_data.append(json.loads(line))

    print(f"\nLoaded {len(corpus_texts)} documents.")

    # ── TF-IDF ──
    print("\n" + "-" * 60)
    print("方法一: TF-IDF")
    print("-" * 60)

    tfidf = TFIDFExtractor()
    tfidf.fit(corpus_texts)

    for i, doc in enumerate(corpus_data[:5]):
        kw = tfidf.extract(doc, top_n=8)
        title = doc.get("topic", {}).get("word", f"文档{i+1}")
        print(f"  {title}: {', '.join(kw)}")

    # ── TextRank ──
    print("\n" + "-" * 60)
    print("方法二: TextRank")
    print("-" * 60)

    textrank = TextRankExtractor()

    for i, doc in enumerate(corpus_data[:5]):
        kw = textrank.extract(doc, top_n=8)
        title = doc.get("topic", {}).get("word", f"文档{i+1}")
        print(f"  {title}: {', '.join(kw)}")

    # ── 单篇示例 ──
    print("\n" + "-" * 60)
    print("单篇示例 (TextRank)")
    print("-" * 60)

    sample = "#突发#！！某地暴雨太大了！！！详情见 http://xxx.com"
    result = textrank.extract(sample, top_n=5, with_scores=True)
    for w, s in result:
        print(f"  {w}: {s:.4f}")
