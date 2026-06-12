"""
NLP 主题发现模块 — LDA + K-means 主题建模与文档聚类

传统方法实现：
1. LDA (Latent Dirichlet Allocation) — 基于 Collapsed Gibbs Sampling 的实现
2. K-means 聚类 — 基于 TF-IDF 特征的文档聚类

Usage:
    from topic_discovery import TopicDiscoverer
    td = TopicDiscoverer(method="lda", n_topics=8)
    td.fit(texts)
    topics = td.get_topics(top_n=10)
    # [{"topic_id": 0, "words": ["价格", "诈骗", ...], "weight": 0.15}, ...]
"""

import math
import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from preprocessing import TextPreprocessor


# ═══════════════════════════════════════════════════════════
#  1. LDA — Collapsed Gibbs Sampling
# ═══════════════════════════════════════════════════════════

class LDA:
    """LDA 主题模型 (Collapsed Gibbs Sampling)。

    Parameters
    ----------
    n_topics : int, default=10
    alpha : float, default=0.1
        文档-主题 Dirichlet 先验
    beta : float, default=0.01
        主题-词 Dirichlet 先验
    n_iter : int, default=200
        Gibbs 采样迭代次数
    random_state : int, optional
    """

    def __init__(
        self,
        n_topics: int = 10,
        alpha: float = 0.1,
        beta: float = 0.01,
        n_iter: int = 200,
        random_state: Optional[int] = None,
    ):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self._pp = TextPreprocessor()
        self._rng = random.Random(random_state) if random_state else random

        # Training state
        self._vocab: Dict[str, int] = {}          # word → id
        self._id2word: Dict[int, str] = {}        # id → word
        self._V: int = 0                          # vocab size
        self._D: int = 0                          # document count
        self._phi: List[List[float]] = []          # topic-word distribution
        self._theta: List[List[float]] = []        # document-topic distribution
        self._fitted = False

    # ── 训练 ──

    def fit(self, texts: List[str]) -> "LDA":
        """在文档集合上训练 LDA 模型。"""
        # 1. 预处理
        doc_tokens = []
        all_tokens = set()
        for text in texts:
            t = self._pp.process(text)
            if t:
                tokens = t.split()
                doc_tokens.append(tokens)
                all_tokens.update(tokens)

        # 过滤低频词（只出现1次的词）
        word_counts = Counter()
        for tokens in doc_tokens:
            word_counts.update(tokens)
        min_freq = 1
        vocab_words = [w for w, c in word_counts.items() if c > min_freq]

        self._vocab = {w: i for i, w in enumerate(vocab_words)}
        self._id2word = {i: w for w, i in self._vocab.items()}
        self._V = len(self._vocab)
        self._D = len(doc_tokens)

        if self._V == 0 or self._D == 0:
            return self

        # 2. 转换为 ID 序列
        docs = []
        for tokens in doc_tokens:
            doc_ids = [self._vocab[w] for w in tokens if w in self._vocab]
            docs.append(doc_ids)

        # 3. 初始化 Gibbs 采样
        # z[d][i] = topic assignment for word i in document d
        z: List[List[int]] = []
        # nd[z] = count of words in doc d assigned to topic z
        nd: List[List[int]] = [[0] * self.n_topics for _ in range(self._D)]
        # nw[z][w] = count of word w assigned to topic z
        nw: List[List[int]] = [[0] * self._V for _ in range(self.n_topics)]
        # nzsum[z] = total words assigned to topic z
        nzsum: List[int] = [0] * self.n_topics

        K = self.n_topics
        alpha = self.alpha
        beta = self.beta
        V = self._V

        for d, doc_ids in enumerate(docs):
            doc_z = []
            for w_id in doc_ids:
                topic = self._rng.randint(0, K - 1)
                doc_z.append(topic)
                nd[d][topic] += 1
                nw[topic][w_id] += 1
                nzsum[topic] += 1
            z.append(doc_z)

        # 4. Gibbs 采样迭代
        for iteration in range(self.n_iter):
            for d, doc_ids in enumerate(docs):
                for i, w_id in enumerate(doc_ids):
                    topic = z[d][i]
                    # 减去当前赋值
                    nd[d][topic] -= 1
                    nw[topic][w_id] -= 1
                    nzsum[topic] -= 1

                    # 计算条件概率 P(z=k | z_{-i}, w)
                    probs = []
                    for k in range(K):
                        p = (nd[d][k] + alpha) * (nw[k][w_id] + beta) / (nzsum[k] + V * beta)
                        probs.append(p)

                    # 归一化并采样
                    total = sum(probs)
                    if total > 0:
                        probs = [p / total for p in probs]
                    else:
                        probs = [1.0 / K] * K

                    # 轮盘赌采样
                    r = self._rng.random()
                    cum = 0.0
                    new_topic = K - 1
                    for k in range(K):
                        cum += probs[k]
                        if r < cum:
                            new_topic = k
                            break

                    # 更新赋值
                    z[d][i] = new_topic
                    nd[d][new_topic] += 1
                    nw[new_topic][w_id] += 1
                    nzsum[new_topic] += 1

        # 5. 计算 phi (topic-word) 和 theta (document-topic)
        self._phi = []
        for k in range(K):
            phi_k = [(nw[k][w_id] + beta) / (nzsum[k] + V * beta) for w_id in range(V)]
            self._phi.append(phi_k)

        self._theta = []
        for d in range(self._D):
            n_d = sum(nd[d])
            theta_d = [(nd[d][k] + alpha) / (n_d + K * alpha) for k in range(K)]
            self._theta.append(theta_d)

        self._fitted = True
        return self

    # ── 推断 ──

    def transform(self, texts: List[str]) -> List[List[float]]:
        """推断新文档的主题分布。使用 folded-in Gibbs sampling。"""
        if not self._fitted:
            return []

        result = []
        K = self.n_topics
        alpha = self.alpha
        V = self._V

        for text in texts:
            t = self._pp.process(text)
            if not t:
                result.append([1.0 / K] * K)
                continue
            tokens = t.split()
            doc_ids = [self._vocab[w] for w in tokens if w in self._vocab]
            if not doc_ids:
                result.append([1.0 / K] * K)
                continue

            # Folded-in: 固定 nw/nzsum，只更新 nd
            nd = [0] * K
            z_doc = []
            for w_id in doc_ids:
                topic = self._rng.randint(0, K - 1)
                z_doc.append(topic)
                nd[topic] += 1

            # 少量迭代
            for _ in range(20):
                for i, w_id in enumerate(doc_ids):
                    topic = z_doc[i]
                    nd[topic] -= 1
                    probs = []
                    nz_sum_k = sum(self._nw_k_sum(k) for k in range(K))
                    for k in range(K):
                        p = (nd[k] + alpha) * (self._phi[k][w_id] if w_id < V else beta)
                        denom = sum(self._phi[k]) if k < K else 1.0
                        p = (nd[k] + alpha) * self._phi[k][w_id]
                        probs.append(p)
                    total = sum(probs)
                    if total > 0:
                        probs = [p / total for p in probs]
                    else:
                        probs = [1.0 / K] * K
                    r = self._rng.random()
                    cum = 0.0
                    new_topic = K - 1
                    for k in range(K):
                        cum += probs[k]
                        if r < cum:
                            new_topic = k
                            break
                    z_doc[i] = new_topic
                    nd[new_topic] += 1

            n_d = sum(nd)
            theta_d = [(nd[k] + alpha) / (n_d + K * alpha) for k in range(K)]
            result.append(theta_d)

        return result

    def _nw_k_sum(self, k: int) -> float:
        return sum(self._phi[k]) if k < self.n_topics else 0.0

    # ── 主题词 ──

    def get_topic_words(self, topic_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """返回指定主题下概率最高的 N 个词。"""
        if not self._fitted or topic_id >= self.n_topics:
            return []
        probs = self._phi[topic_id]
        word_probs = [(self._id2word[i], probs[i]) for i in range(self._V)]
        word_probs.sort(key=lambda x: -x[1])
        return word_probs[:top_n]

    def get_topics(self, top_n: int = 10) -> List[Dict]:
        """返回所有主题及其关键词。"""
        topics = []
        for k in range(self.n_topics):
            words = self.get_topic_words(k, top_n)
            weight = sum(self._theta[d][k] for d in range(self._D)) / self._D
            topics.append({
                "topic_id": k,
                "words": [w for w, _ in words],
                "scores": [round(s, 4) for _, s in words],
                "weight": round(weight, 4),
            })
        topics.sort(key=lambda x: -x["weight"])
        return topics

    @property
    def vocab_size(self) -> int:
        return self._V

    @property
    def n_docs(self) -> int:
        return self._D


# ═══════════════════════════════════════════════════════════
#  2. K-means 文档聚类
# ═══════════════════════════════════════════════════════════

class KMeansTopic:
    """K-means 文档聚类（基于 CountVectorizer 词频特征）。

    Parameters
    ----------
    n_topics : int, default=10
    n_iter : int, default=50
    random_state : int, optional
    """

    def __init__(self, n_topics: int = 10, n_iter: int = 50, random_state: Optional[int] = None):
        self.n_topics = n_topics
        self.n_iter = n_iter
        self._pp = TextPreprocessor()
        self._rng = random.Random(random_state) if random_state else random
        self._vocab: Dict[str, int] = {}
        self._V = 0
        self._cluster_keywords: List[List[Tuple[str, float]]] = []
        self._fitted = False

    def fit(self, texts: List[str]) -> "KMeansTopic":
        """执行 K-means 聚类并提取每个簇的关键词。

        使用词频向量 + Cosine 距离的 K-means 实现。
        """
        # 预处理 + 向量化
        doc_vectors = []
        word_counts = Counter()
        doc_token_lists = []

        for text in texts:
            t = self._pp.process(text)
            if t:
                tokens = t.split()
                doc_token_lists.append(tokens)
                word_counts.update(tokens)

        # 构建词表（过滤低频）
        vocab_words = [w for w, c in word_counts.items() if c > 1]
        if not vocab_words:
            vocab_words = [w for w, c in word_counts.most_common(1000)]
        self._vocab = {w: i for i, w in enumerate(vocab_words)}
        self._V = len(self._vocab)
        D = len(doc_token_lists)

        # 转换为 TF 向量
        X = []
        for tokens in doc_token_lists:
            vec = [0.0] * self._V
            total = len(tokens)
            for w in tokens:
                idx = self._vocab.get(w)
                if idx is not None:
                    vec[idx] += 1.0 / total if total > 0 else 0.0
            X.append(vec)

        # K-means 聚类
        K = min(self.n_topics, D)
        # 随机初始化聚类中心
        centers = self._rng.sample(range(D), K)
        centroids = [X[c][:] for c in centers]

        labels = [0] * D
        for iteration in range(self.n_iter):
            # 分配
            new_labels = []
            for i in range(D):
                best_k, best_dist = 0, float("inf")
                for k in range(K):
                    dist = self._cosine_dist(X[i], centroids[k])
                    if dist < best_dist:
                        best_dist = dist
                        best_k = k
                new_labels.append(best_k)

            # 检查收敛
            if new_labels == labels:
                break
            labels = new_labels

            # 更新质心
            for k in range(K):
                members = [i for i in range(D) if labels[i] == k]
                if not members:
                    centroids[k] = [0.0] * self._V
                else:
                    new_cent = [0.0] * self._V
                    for i in members:
                        for j in range(self._V):
                            new_cent[j] += X[i][j]
                    for j in range(self._V):
                        new_cent[j] /= len(members)
                    centroids[k] = new_cent

        # 为每个簇提取关键词
        self._cluster_keywords = []
        for k in range(K):
            members = [i for i in range(D) if labels[i] == k]
            # 计算该簇内词的平均 TF
            avg_tf = [0.0] * self._V
            for i in members:
                for j in range(self._V):
                    avg_tf[j] += X[i][j]
            if members:
                avg_tf = [v / len(members) for v in avg_tf]

            # 计算全局 TF 作为对比
            global_tf = [0.0] * self._V
            for i in range(D):
                for j in range(self._V):
                    global_tf[j] += X[i][j]
            global_tf = [v / D if v > 0 else 0.0 for v in global_tf]

            # TF-IDF-like 得分：簇内频率 / 全局频率
            scores = []
            for j in range(self._V):
                if global_tf[j] > 0:
                    score = avg_tf[j] / math.sqrt(global_tf[j] + 0.01)
                else:
                    score = 0.0
                if score > 0:
                    scores.append((self._idx2word(j), score))

            scores.sort(key=lambda x: -x[1])
            self._cluster_keywords.append(scores[:15])

        self._fitted = True
        return self

    def get_topics(self, top_n: int = 10) -> List[Dict]:
        """返回所有聚类及其关键词。"""
        topics = []
        for k, words in enumerate(self._cluster_keywords):
            top_words = words[:top_n]
            topics.append({
                "topic_id": k,
                "words": [w for w, _ in top_words],
                "scores": [round(s, 4) for _, s in top_words],
                "weight": round(1.0 / len(self._cluster_keywords), 4),
            })
        return topics

    @staticmethod
    def _cosine_dist(a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 1.0
        return 1.0 - dot / (na * nb)

    def _idx2word(self, idx: int) -> str:
        for w, i in self._vocab.items():
            if i == idx:
                return w
        return f"<{idx}>"


# ═══════════════════════════════════════════════════════════
#  3. 统一接口
# ═══════════════════════════════════════════════════════════

class TopicDiscoverer:
    """主题发现统一接口。

    Parameters
    ----------
    method : str, default="lda"
        "lda" 或 "kmeans"
    **kwargs
        传递给具体模型
    """

    def __init__(self, method: str = "lda", **kwargs):
        self._method = method.lower()
        if self._method == "lda":
            self._model = LDA(**kwargs)
        elif self._method == "kmeans":
            self._model = KMeansTopic(**kwargs)
        else:
            raise ValueError(f"未知方法: {method}")

    def fit(self, texts: List[str]) -> "TopicDiscoverer":
        self._model.fit(texts)
        return self

    def get_topics(self, top_n: int = 10) -> List[Dict]:
        return self._model.get_topics(top_n)

    def transform(self, texts: List[str]) -> List[List[float]]:
        if self._method == "lda":
            return self._model.transform(texts)
        raise NotImplementedError("K-means 不支持 transform")


# ═══════════════════════════════════════════════════════════
#  Demo
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("主题发现 Demo")
    print("=" * 60)

    # 加载数据
    jsonl_path = (
        "E:\\6+7\\SJTU\\大三下\\信息内容安全\\大作业\\数据"
        "\\20260612_141245\\20260612_141245\\representative_posts.jsonl"
    )
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data.get("selection", {}).get("selected_post", {}).get("text", "")
            if text:
                texts.append(text)

    print(f"文档数: {len(texts)}")

    # ── LDA 主题建模 ──
    print("\n" + "-" * 60)
    print("【LDA 主题模型】 (Gibbs Sampling)")
    print("-" * 60)
    lda = TopicDiscoverer(method="lda", n_topics=5, n_iter=200)
    lda.fit(texts)
    topics = lda.get_topics(top_n=8)
    for t in topics:
        words = "  ".join(t["words"])
        print(f"  主题 {t['topic_id']:2d} (权重 {t['weight']:.2f}): {words}")

    # ── K-means 聚类 ──
    print("\n" + "-" * 60)
    print("【K-means 聚类】 (词频特征)")
    print("-" * 60)
    km = TopicDiscoverer(method="kmeans", n_topics=5)
    km.fit(texts)
    topics2 = km.get_topics(top_n=8)
    for t in topics2:
        words = "  ".join(t["words"])
        print(f"  簇 {t['topic_id']:2d}: {words}")

    # ── 单文档主题分布 ──
    print("\n" + "-" * 60)
    print("单文档主题推断")
    print("-" * 60)
    sample = "2026年6月，某公司在上海发布新款手机"
    theta = lda.transform([sample])[0]
    topic_words = lda._model.get_topics(top_n=3)
    for k in range(len(theta)):
        if theta[k] > 0.05:
            words = ", ".join(topic_words[k]["words"][:3])
            print(f"  主题{k} ({theta[k]:.1%}): {words}")
