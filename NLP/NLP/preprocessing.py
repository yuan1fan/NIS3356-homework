import re
import html
import hashlib
from pathlib import Path
from typing import List, Optional

import jieba
from opencc import OpenCC


EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U0000231A-\U0000231B"
    "\U000023E9-\U000023F3"
    "\U000025AA-\U000025AB"
    "\U000025B6"
    "\U000025C0"
    "\U000025FB-\U000025FE"
    "\U00002600-\U000027BF"
    "\U00002702-\U000027B0"
    "\U00002934-\U00002935"
    "\U00002B05-\U00002B07"
    "\U00002B1B-\U00002B1C"
    "\U00002B50"
    "\U00002B55"
    "\U00003030"
    "\U0000303D"
    "\U00003297"
    "\U00003299"
    "\U0000200D"
    "\U000020E3"
    "\U000000A9"
    "\U000000AE"
    "\U00002122"
    "\U0000FE00-\U0000FE0F"
    "\U0000FEFF"
    "\U0000200B-\U0000200E"
    "]+",
    flags=re.UNICODE,
)

URL_PATTERN = re.compile(
    r"https?://[^\s,，。；；！!?？】\)）\]」』》》\"'，、…—–-]+",
    flags=re.IGNORECASE,
)

MENTION_PATTERN = re.compile(r"@[\w\-_]+")
HASHTAG_PATTERN = re.compile(r"#([^#]+)#")
REPEATED_PUNCT_PATTERN = re.compile(r"([！!？?。，、；：…\.\,\!\?\;\:\s]){2,}")
NOISE_CHAR_PATTERN = re.compile(r"[^\u4e00-\u9fff\u3400-\u4dbf\w\s\.\,\!\?\;\:\-\+\#\@\&\$\€\¥\%\°\~\^\*\(\)\[\]\{\}\/\"\'\、\。\，\！\？\；\：\…\—\·\【\】\（\）\《\》\——\-\~\~]")
WHITESPACE_PATTERN = re.compile(r"\s+")

AD_KEYWORDS = [
    "点击购买", "立即下单",
    "限时优惠", "免费领取", "加V",
    "添加微信", "扫码", "二维码", "转发抽奖", "关注送",
    "复制链接", "包邮", "清仓",
    "跳转",
]

WEIBO_TRAILING_PATTERNS = [
    re.compile(r"收起\w?"),
    re.compile(r"查看\s*(图片|视频|全文|详情)"),
    re.compile(r"[0O]\s*网页\s*链接"),
    re.compile(r"转发\s*微博"),
    re.compile(r"我\s*分享\s*了"),
    re.compile(r"\uff08\s*分享\s*自\s*[^\uff09]*\uff09"),
    re.compile(r"\(分享自[^)]*\)"),
]


class TextPreprocessor:

    def __init__(self, stopwords_path: Optional[str] = None):
        self._cc = OpenCC("t2s")
        self._stopwords = self._load_stopwords(stopwords_path)
        jieba.initialize()

    def process(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        t = text
        t = self._unescape_html(t)
        t = self._remove_html_tags(t)
        t = self._remove_urls(t)
        t = self._remove_mentions(t)
        t = self._process_hashtags(t)
        t = self._remove_emojis(t)
        t = self._remove_weibo_trailing(t)
        t = self._remove_meaningless_symbols(t)
        t = self._compress_whitespace(t)
        if not t.strip():
            return ""
        if self._is_ad_content(t):
            return ""
        t = self._to_simplified(t)
        tokens = self._tokenize(t)
        tokens = self._remove_stopwords(tokens)
        if not tokens:
            return ""
        tokens = self._remove_consecutive_duplicates(tokens)
        return " ".join(tokens)

    def process_with_steps(self, text: str) -> dict:
        steps = {}
        steps["原始文本"] = text
        t = text
        if not t or not t.strip():
            return steps
        t = self._unescape_html(t)
        steps["\u2460 \u53bb\u9664 HTML \u5b9e\u4f53"] = t
        t = self._remove_html_tags(t)
        steps["\u2461 \u53bb\u9664 HTML \u6807\u7b7e"] = t
        t = self._remove_urls(t)
        steps["\u2462 \u53bb\u9664 URL"] = t
        t = self._remove_mentions(t)
        steps["\u2463 \u53bb\u9664 @\u7528\u6237"] = t
        t = self._process_hashtags(t)
        steps["\u2464 \u5904\u7406 #\u8bdd\u9898#"] = t
        t = self._remove_emojis(t)
        steps["\u2465 \u53bb\u9664\u8868\u60c5\u7b26\u53f7"] = t
        t = self._remove_weibo_trailing(t)
        steps["\u2466 \u53bb\u9664\u5fae\u535a\u540e\u7f00"] = t
        t = self._remove_meaningless_symbols(t)
        steps["\u2467 \u53bb\u9664\u65e0\u610f\u4e49\u7b26\u53f7"] = t
        t = self._compress_whitespace(t)
        steps["\u2468 \u538b\u7f29\u7a7a\u767d"] = t
        if not t.strip():
            return steps
        ad_status = "\u662f (\u5df2\u4e22\u5f03)" if self._is_ad_content(t) else "\u5426"
        steps["\u2469 \u5e7f\u544a\u68c0\u6d4b"] = ad_status
        t = self._to_simplified(t)
        steps["\u246a \u7b80\u7e41\u8f6c\u6362"] = t
        tokens = self._tokenize(t)
        steps["\u246b \u5206\u8bcd"] = " | ".join(tokens)
        tokens = self._remove_stopwords(tokens)
        steps["\u246c \u505c\u7528\u8bcd\u8fc7\u6ee4"] = " | ".join(tokens)
        tokens = self._remove_consecutive_duplicates(tokens)
        steps["\u246d \u53bb\u91cd"] = " | ".join(tokens)
        steps["\u6700\u7ec8\u7ed3\u679c"] = " ".join(tokens)
        return steps

    @staticmethod
    def _unescape_html(text: str) -> str:
        return html.unescape(text)

    @staticmethod
    def _remove_html_tags(text: str) -> str:
        return re.sub(r"<[^>]+>", "", text)

    @staticmethod
    def _remove_urls(text: str) -> str:
        return URL_PATTERN.sub("", text)

    @staticmethod
    def _remove_mentions(text: str) -> str:
        return MENTION_PATTERN.sub("", text)

    @staticmethod
    def _process_hashtags(text: str) -> str:
        def _keep_content(m):
            return m.group(1)
        return HASHTAG_PATTERN.sub(_keep_content, text)

    @staticmethod
    def _remove_emojis(text: str) -> str:
        return EMOJI_PATTERN.sub("", text)

    @staticmethod
    def _remove_weibo_trailing(text: str) -> str:
        for pat in WEIBO_TRAILING_PATTERNS:
            text = pat.sub("", text)
        return text

    @staticmethod
    def _remove_meaningless_symbols(text: str) -> str:
        text = REPEATED_PUNCT_PATTERN.sub(r"\1", text)
        text = NOISE_CHAR_PATTERN.sub("", text)
        return text

    @staticmethod
    def _compress_whitespace(text: str) -> str:
        return WHITESPACE_PATTERN.sub(" ", text).strip()

    def _is_ad_content(self, text: str) -> bool:
        """检测是否为广告内容（要求至少匹配 2 个关键词以减少误判）。"""
        lowered = text.lower()
        hits = 0
        for kw in AD_KEYWORDS:
            if kw in lowered:
                hits += 1
                if hits >= 2:
                    return True
        return False

    def _to_simplified(self, text: str) -> str:
        return self._cc.convert(text)

    def _tokenize(self, text: str) -> List[str]:
        return list(jieba.cut(text, cut_all=False))

    def _remove_stopwords(self, tokens: List[str]) -> List[str]:
        result = []
        for w in tokens:
            w = w.strip()
            if not w:
                continue
            if w in self._stopwords:
                continue
            if re.match(r"^[\d\s\.\,\!\?\;\:\-\+\#\@\&\$\%\*\(\)\[\]\{\}\/\'\"]+$", w):
                continue
            if len(w) < 2 and not re.match(r"^[\u4e00-\u9fff]$", w):
                continue
            result.append(w)
        return result

    @staticmethod
    def _remove_consecutive_duplicates(tokens: List[str]) -> List[str]:
        if not tokens:
            return []
        result = [tokens[0]]
        for w in tokens[1:]:
            if w != result[-1]:
                result.append(w)
        return result

    @staticmethod
    def is_near_duplicate(text_a: str, text_b: str, threshold: float = 0.85) -> bool:
        set_a = set(text_a)
        set_b = set(text_b)
        if not set_a or not set_b:
            return False
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) >= threshold

    @staticmethod
    def _load_stopwords(path: Optional[str] = None) -> set:
        if path is None:
            path = str(Path(__file__).parent / "stopwords.txt")
        stopwords = set()
        p = Path(path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                w = line.strip()
                if w and not w.startswith("#"):
                    stopwords.add(w)
        else:
            stopwords = {
                "\u7684", "\u4e86", "\u5728", "\u662f", "\u6211", "\u6709", "\u548c", "\u5c31", "\u4e0d",
                "\u4eba", "\u90fd", "\u4e00", "\u4e00\u4e2a", "\u4e0a", "\u4e5f", "\u5f88", "\u5230", "\u8bf4",
                "\u8981", "\u53bb", "\u4f60", "\u4f1a", "\u7740", "\u6ca1\u6709", "\u770b", "\u597d", "\u81ea\u5df1",
                "\u8fd9", "\u4ed6", "\u5979", "\u5b83", "\u4eec", "\u90a3", "\u4ec0\u4e48", "\u600e\u4e48", "\u4e3a\u4ec0\u4e48",
                "\u5417", "\u5427", "\u554a", "\u5462", "\u54e6", "\u54c8", "\u5440", "\u5566",
            }
        return stopwords

    @staticmethod
    def text_fingerprint(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()


def batch_process(texts: List[str], preprocessor: Optional[TextPreprocessor] = None) -> List[str]:
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    return [preprocessor.process(t) for t in texts]


def deduplicate_corpus(texts: List[str], threshold: float = 0.85) -> List[str]:
    pp = TextPreprocessor()
    seen = {}
    result = []
    for text in texts:
        processed = pp.process(text)
        if not processed:
            continue
        is_dup = False
        for existing in seen:
            if pp.is_near_duplicate(processed, existing, threshold):
                is_dup = True
                break
        if not is_dup:
            seen[processed] = True
            result.append(text)
    return result


if __name__ == "__main__":
    pp = TextPreprocessor()
    test_cases = [
        "#突发#！！某地暴雨太大了！！！详情见 http://xxx.com",
        "转发微博【#某明星辟谣#】@娱乐小编 某明星工作室发声明啦！https://t.cn/A123 查看更多",
        "\U0001f60a\U0001f60a今天天气真好呀！！！【图片】查看图片",
        "限时优惠！添加微信xxxxx免费领取大礼包，点击链接购买",
        "真的太生气了\U0001f621\U0001f621！！！！！#社会新闻# @人民日报 报道了这件事",
        "这是一段繁体中文測試，看看轉換效果如何呢？",
    ]
    print("=" * 60)
    print("Text Preprocessing Demo")
    print("=" * 60)
    for case in test_cases:
        print(f"\n原始文本: {case}")
        result = pp.process(case)
        print(f"处理结果: {result}")
        print("-" * 60)
    print("\n--- 分步展示 ---")
    steps = pp.process_with_steps("#突发#！！某地暴雨太大了！！！详情见 http://xxx.com")
    for k, v in steps.items():
        print(f"  {k}: {v}")

