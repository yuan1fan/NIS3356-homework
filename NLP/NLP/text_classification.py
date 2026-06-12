"""
NLP 文本分类模块
从零实现传统分类算法，用于中文文本分类。
"""
import math, json, random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Union
from preprocessing import TextPreprocessor

class CountVectorizer:
    def __init__(self, max_features=5000, min_df=1):
        self._max_features, self._min_df = max_features, min_df
        self._vocab = {}
        self._pp = TextPreprocessor()
        self._fitted = False

    def fit(self, texts):
        token_lists = []
        for text in texts:
            t = self._pp.process(text)
            if t: token_lists.append(t.split())
        df = defaultdict(int)
        for tokens in token_lists:
            for w in set(tokens): df[w] += 1
        filtered = [(w,c) for w,c in df.items() if c >= self._min_df]
        filtered.sort(key=lambda x: -x[1])
        self._vocab = {w:i for i,(w,_) in enumerate(filtered[:self._max_features])}
        self._fitted = True
        return self

    def transform(self, texts):
        return [self.transform_single(t) for t in texts]

    def transform_single(self, text):
        if not self._fitted: return []
        n = len(self._vocab); vec = [0.0]*n
        t = self._pp.process(text)
        if not t: return vec
        for w in t.split():
            idx = self._vocab.get(w)
            if idx is not None: vec[idx] += 1.0
        return vec

class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self._alpha = alpha
        self._classes, self._prior = [], {}
        self._cond_prob = {}
        self._fitted = False

    def fit(self, X, y):
        n = len(X)
        if n == 0: return self
        self._n_features = len(X[0])
        self._classes = sorted(set(y))
        label_counts = Counter(y)
        for c in self._classes:
            self._prior[c] = math.log(label_counts[c]/n)
        feature_counts = defaultdict(lambda: [0.0]*self._n_features)
        total_words = defaultdict(float)
        for i in range(n):
            c = y[i]
            for j in range(self._n_features):
                if X[i][j] > 0:
                    feature_counts[c][j] += X[i][j]
                    total_words[c] += X[i][j]
        V, alpha = self._n_features, self._alpha
        for c in self._classes:
            total = total_words[c] + alpha*V
            probs = [0.0]*self._n_features
            for j in range(self._n_features):
                probs[j] = math.log((feature_counts[c][j]+alpha)/total)
            self._cond_prob[c] = probs
        self._fitted = True
        return self

    def predict(self, X):
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        if not self._fitted: return ""
        best_c, best_s = self._classes[0], -float("inf")
        for c in self._classes:
            s = self._prior[c]
            for j,v in enumerate(x):
                if v > 0 and j < len(self._cond_prob[c]): s += v*self._cond_prob[c][j]
            if s > best_s: best_s, best_c = s, c
        return best_c

    def predict_proba(self, X):
        results = []
        for x in X:
            scores = {}
            for c in self._classes:
                s = self._prior[c]
                for j,v in enumerate(x):
                    if v > 0 and j < len(self._cond_prob[c]): s += v*self._cond_prob[c][j]
                scores[c] = s
            max_s = max(scores.values())
            exp_s = {c:math.exp(s-max_s) for c,s in scores.items()}
            total = sum(exp_s.values())
            results.append({c:exp_s[c]/total for c in self._classes})
        return results

class LogisticRegressionClassifier:
    def __init__(self, lr=0.1, epochs=500, l2_lambda=0.01):
        self._lr, self._epochs, self._l2 = lr, epochs, l2_lambda
        self._classes, self._weights = [], []
        self._fitted = False

    def fit(self, X, y):
        n = len(X)
        if n == 0: return self
        m = len(X[0])
        self._classes = sorted(set(y))
        k = len(self._classes)
        Y = [[1.0 if label==c else 0.0 for c in self._classes] for label in y]
        self._weights = [[0.0]*(m+1) for _ in range(k)]
        X_aug = [row+[1.0] for row in X]
        lr, l2 = self._lr, self._l2
        for epoch in range(self._epochs):
            for ci in range(k):
                w = self._weights[ci]
                preds = [self._sigmoid(sum(x[j]*w[j] for j in range(m+1))) for x in X_aug]
                grad = [0.0]*(m+1)
                for i in range(n):
                    err = preds[i] - Y[i][ci]
                    for j in range(m+1): grad[j] += err*X_aug[i][j]
                for j in range(m): grad[j] = (grad[j]+l2*w[j])/n
                grad[m] = grad[m]/n
                for j in range(m+1): w[j] -= lr*grad[j]
        self._fitted = True
        return self

    def predict(self, X):
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        if not self._fitted: return ""
        x_aug = x+[1.0]
        best_c, best_s = self._classes[0], -float("inf")
        for ci,c in enumerate(self._classes):
            s = self._sigmoid(sum(x_aug[j]*self._weights[ci][j] for j in range(len(x_aug))))
            if s > best_s: best_s, best_c = s, c
        return best_c

    def predict_proba(self, X):
        results = []
        for x in X:
            x_aug = x+[1.0]
            scores = {}
            for ci,c in enumerate(self._classes):
                scores[c] = self._sigmoid(sum(x_aug[j]*self._weights[ci][j] for j in range(len(x_aug))))
            total = sum(scores.values())
            results.append({c:scores[c]/total for c in self._classes})
        return results

    @staticmethod
    def _sigmoid(z):
        if z > 20: return 1.0
        if z < -20: return 0.0
        return 1.0/(1.0+math.exp(-z))

def accuracy(y_true, y_pred):
    return sum(1 for a,b in zip(y_true,y_pred) if a==b)/len(y_true) if y_true else 0.0

def classification_report(y_true, y_pred):
    classes = sorted(set(y_true+y_pred))
    report = {}
    for c in classes:
        tp = sum(1 for a,b in zip(y_true,y_pred) if a==c and b==c)
        fp = sum(1 for a,b in zip(y_true,y_pred) if a!=c and b==c)
        fn = sum(1 for a,b in zip(y_true,y_pred) if a==c and b!=c)
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        report[c] = {"precision":round(prec,4),"recall":round(rec,4),"f1":round(f1,4),"support":tp+fn}
    report["accuracy"] = round(accuracy(y_true,y_pred),4)
    return report

def confusion_matrix(y_true, y_pred):
    classes = sorted(set(y_true+y_pred))
    mat = {c:{c2:0 for c2 in classes} for c in classes}
    for a,b in zip(y_true,y_pred): mat[a][b] += 1
    return mat

class TextClassifier:
    def __init__(self, method="naive_bayes", vectorizer="count", max_features=5000, **kwargs):
        self._method, self._vec_name = method, vectorizer
        if vectorizer == "count":
            self._vec = CountVectorizer(max_features=max_features, min_df=kwargs.get("min_df", 1))
        else:
            self._vec = CountVectorizer(max_features=max_features)
        if method == "naive_bayes":
            self._clf = NaiveBayesClassifier(**{k:v for k,v in kwargs.items() if k=="alpha"})
        elif method == "logistic_regression":
            self._clf = LogisticRegressionClassifier(**{k:v for k,v in kwargs.items() if k in ("lr","epochs","l2_lambda")})
        else:
            raise ValueError(f"未知方法: {method}")
        self._classes, self._fitted = [], False
        self._use_tfidf = (vectorizer == "tfidf")
        self._idf = {}

    def fit(self, texts, labels):
        self._classes = sorted(set(labels))
        self._vec.fit(texts)
        X = self._vec.transform(texts)
        if self._use_tfidf:
            self._compute_idf(X)
            X = self._apply_tfidf(X, texts)
        self._clf.fit(X, labels)
        self._fitted = True
        return self

    def predict(self, text):
        if not self._fitted: return ""
        x = self._vec.transform_single(text)
        if self._use_tfidf:
            x = self._apply_tfidf_single(x, text)
        return self._clf.predict([x])[0]

    def predict_proba(self, text):
        if not self._fitted: return {}
        x = self._vec.transform_single(text)
        if self._use_tfidf:
            x = self._apply_tfidf_single(x, text)
        return self._clf.predict_proba([x])[0]

    def evaluate(self, texts, labels):
        X = self._vec.transform(texts)
        if self._use_tfidf:
            X = self._apply_tfidf(X, texts)
        preds = self._clf.predict(X)
        report = classification_report(labels, preds)
        report["confusion_matrix"] = confusion_matrix(labels, preds)
        return report

    def _apply_tfidf(self, X, texts):
        X_out = []
        for i, row in enumerate(X):
            total = sum(row)
            if total == 0:
                X_out.append([0.0]*len(row))
                continue
            tfidf = [(c/total) * self._idf.get(j, 1.0) for j, c in enumerate(row)]
            norm = math.sqrt(sum(v*v for v in tfidf))
            X_out.append([v/norm for v in tfidf] if norm > 0 else tfidf)
        return X_out

    def _apply_tfidf_single(self, x, text):
        total = sum(x)
        if total == 0: return [0.0]*len(x)
        tfidf = [(c/total) * self._idf.get(j, 1.0) for j, c in enumerate(x)]
        norm = math.sqrt(sum(v*v for v in tfidf))
        return [v/norm for v in tfidf] if norm > 0 else tfidf

    def _compute_idf(self, X):
        n = len(X)
        if n == 0: return
        df = [0]*len(X[0])
        for row in X:
            for j, v in enumerate(row):
                if v > 0: df[j] += 1
        for j, d in enumerate(df):
            self._idf[j] = math.log((n+1)/(d+1)) + 1.0

    def cross_validate(self, texts, labels, folds=5):
        n = len(texts)
        indices = list(range(n)); random.shuffle(indices)
        st, sl = [texts[i] for i in indices], [labels[i] for i in indices]
        fs = n//folds; accs = []
        for f in range(folds):
            vs, ve = f*fs, (f+1)*fs if f<folds-1 else n
            clf = TextClassifier(method=self._method, vectorizer=self._vec_name, max_features=len(self._vec.vocabulary) if hasattr(self._vec, 'vocabulary') else 5000)
            clf.fit(st[:vs]+st[ve:], sl[:vs]+sl[ve:])
            r = clf.evaluate(st[vs:ve], sl[vs:ve])
            accs.append(r.get("accuracy",0.0))
        avg = sum(accs)/len(accs)
        std = math.sqrt(sum((a-avg)**2 for a in accs)/len(accs))
        return {"accuracies": [round(a,4) for a in accs],
                "mean_accuracy": round(avg,4),
                "std_accuracy": round(std,4)}

def auto_label_by_keywords(texts, category_keywords):
    pp = TextPreprocessor()
    labels = []
    for text in texts:
        processed = pp.process(text)
        assigned = None
        for cat, kws in category_keywords.items():
            for kw in kws:
                if kw in processed: assigned = cat; break
            if assigned: break
        labels.append(assigned if assigned else "其他")
    return labels
