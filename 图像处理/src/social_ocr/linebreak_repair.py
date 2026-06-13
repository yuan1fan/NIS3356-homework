from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Iterable


KEEP_SEPARATOR = 0
JOIN_LINES = 1

LEFT_CLOSING_PUNCT = "。！？?!；;：:"
LEFT_CONTINUING_PUNCT = "，,、"
RIGHT_PUNCT = "，,。！？?!；;：:、）)]】》”’"
LEFT_OPENING_PUNCT = "（([【《“‘"

RIGHT_CONTINUATION_PREFIXES = (
    "的",
    "地",
    "得",
    "了",
    "着",
    "过",
    "中",
    "内",
    "上",
    "下",
    "前",
    "后",
    "间",
    "时",
    "年",
    "月",
    "日",
    "人",
    "者",
    "员",
    "疑",
    "疑人",
    "种",
    "命",
    "费用",
    "金额",
    "方式",
    "项目",
    "门店",
    "万元",
    "亿元",
    "万余元",
)
LEFT_CONTINUATION_SUFFIXES = (
    "的",
    "地",
    "得",
    "和",
    "与",
    "及",
    "或",
    "把",
    "将",
    "被",
    "为",
    "对",
    "向",
    "在",
    "从",
    "至",
    "到",
    "以",
    "因",
    "如",
    "若",
    "并",
    "但",
    "而",
    "且",
    "各",
    "所",
    "专门针对老",
    "犯罪嫌",
    "嫌",
    "涉案金额",
    "金额",
    "过程",
    "生",
    "密",
    "等",
)
RIGHT_NEW_SEGMENT_PREFIXES = (
    "北京警方",
    "警方",
    "记者",
    "店员",
    "以其",
    "每个",
    "共计",
    "据悉",
    "报道称",
    "网友",
    "专家表示",
)
PARALLEL_MEDIA_SUFFIXES = ("文本", "图片", "图像", "音频", "视频", "音视频")
PARALLEL_MEDIA_PREFIXES = ("文本", "图片", "图像", "音频", "视频", "音视频")

_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
_DIGIT_RE = re.compile(r"\d")
_LATIN_RE = re.compile(r"[A-Za-z]")


class _BoundaryModel:
    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = weights

    @classmethod
    def train(cls, samples: list[tuple[str, str, int]]) -> "_BoundaryModel":
        weights: dict[str, float] = {}
        learning_rate = 0.10
        l2 = 0.0008
        for _ in range(220):
            for left, right, label in samples:
                features = _extract_features(left, right)
                score = sum(weights.get(name, 0.0) * value for name, value in features.items())
                pred = _sigmoid(score)
                error = label - pred
                for name, value in features.items():
                    weights[name] = weights.get(name, 0.0) + learning_rate * (
                        error * value - l2 * weights.get(name, 0.0)
                    )
        return cls(weights)

    def join_probability(self, left: str, right: str) -> float:
        features = _extract_features(left, right)
        score = sum(self.weights.get(name, 0.0) * value for name, value in features.items())
        return _sigmoid(score)


def repair_ocr_linebreaks(lines: Iterable[str], separator: str = "；") -> str:
    """Merge OCR hard-wrap lines while keeping true semantic boundaries."""
    normalized = [str(line or "").strip() for line in lines if str(line or "").strip()]
    if not normalized:
        return ""

    segments = [normalized[0]]
    for line in normalized[1:]:
        if should_join_ocr_lines(segments[-1], line):
            segments[-1] += line
        else:
            segments.append(line)
    return separator.join(segments)


def should_join_ocr_lines(left: str, right: str) -> bool:
    left = str(left or "").strip()
    right = str(right or "").strip()
    if not left or not right:
        return False

    rule = _rule_decision(left, right)
    if rule is not None:
        return rule

    probability = _get_model().join_probability(left, right)
    return probability >= 0.45


def _rule_decision(left: str, right: str) -> bool | None:
    left_last = left[-1]
    right_first = right[0]
    if right_first in RIGHT_PUNCT:
        return True
    if left_last in LEFT_OPENING_PUNCT:
        return True
    if left_last in LEFT_CLOSING_PUNCT:
        return False
    if left_last in LEFT_CONTINUING_PUNCT:
        return True
    if _is_number_unit_boundary(left, right):
        return True
    if _has_prefix(right, RIGHT_NEW_SEGMENT_PREFIXES):
        return False
    if _has_suffix(left, LEFT_CONTINUATION_SUFFIXES) or _has_prefix(right, RIGHT_CONTINUATION_PREFIXES):
        return True
    if _has_suffix(left, PARALLEL_MEDIA_SUFFIXES) and _has_prefix(right, PARALLEL_MEDIA_PREFIXES):
        return True
    if _looks_like_two_short_independent_segments(left, right):
        return False
    return None


def _extract_features(left: str, right: str) -> dict[str, float]:
    left_last = left[-1] if left else ""
    right_first = right[0] if right else ""
    left_len = len(left)
    right_len = len(right)
    return {
        "bias": 1.0,
        "left_len": min(left_len, 32) / 32,
        "right_len": min(right_len, 32) / 32,
        "left_short": float(left_len <= 10),
        "right_short": float(right_len <= 10),
        "both_short": float(left_len <= 14 and right_len <= 14),
        "left_very_long": float(left_len >= 18),
        "right_very_long": float(right_len >= 18),
        "left_ends_closing_punct": float(left_last in LEFT_CLOSING_PUNCT),
        "left_ends_continuing_punct": float(left_last in LEFT_CONTINUING_PUNCT),
        "right_starts_punct": float(right_first in RIGHT_PUNCT),
        "left_open_right": float(left_last in LEFT_OPENING_PUNCT),
        "last_is_chinese": float(bool(_CHINESE_RE.fullmatch(left_last))),
        "first_is_chinese": float(bool(_CHINESE_RE.fullmatch(right_first))),
        "chinese_to_chinese": float(bool(_CHINESE_RE.fullmatch(left_last) and _CHINESE_RE.fullmatch(right_first))),
        "digit_to_digit": float(bool(_DIGIT_RE.fullmatch(left_last) and _DIGIT_RE.fullmatch(right_first))),
        "digit_to_chinese": float(bool(_DIGIT_RE.fullmatch(left_last) and _CHINESE_RE.fullmatch(right_first))),
        "latin_to_latin": float(bool(_LATIN_RE.fullmatch(left_last) and _LATIN_RE.fullmatch(right_first))),
        "number_unit_boundary": float(_is_number_unit_boundary(left, right)),
        "left_continuation_suffix": float(_has_suffix(left, LEFT_CONTINUATION_SUFFIXES)),
        "right_continuation_prefix": float(_has_prefix(right, RIGHT_CONTINUATION_PREFIXES)),
        "right_new_segment_prefix": float(_has_prefix(right, RIGHT_NEW_SEGMENT_PREFIXES)),
        "short_independent": float(_looks_like_two_short_independent_segments(left, right)),
    }


def _make_training_samples() -> list[tuple[str, str, int]]:
    samples: list[tuple[str, str, int]] = []
    for text in _TRAINING_PARAGRAPHS:
        samples.extend(_make_hard_wrap_samples(text))
        samples.extend(_make_sentence_boundary_samples(text))
    samples.extend((left, right, JOIN_LINES) for left, right in _MANUAL_JOIN_PAIRS)
    samples.extend((left, right, KEEP_SEPARATOR) for left, right in _MANUAL_KEEP_PAIRS)
    return samples


def _make_hard_wrap_samples(text: str) -> list[tuple[str, str, int]]:
    compact = re.sub(r"\s+", "", text)
    samples: list[tuple[str, str, int]] = []
    for sentence in re.split(r"[。！？?!；;]", compact):
        if len(sentence) < 16:
            continue
        for split_at in (8, 11, 14, 17, 20):
            if split_at >= len(sentence) - 3:
                continue
            left = sentence[:split_at]
            right = sentence[split_at:]
            if left[-1] in LEFT_CLOSING_PUNCT + LEFT_CONTINUING_PUNCT or right[0] in RIGHT_PUNCT:
                continue
            samples.append((left, right, JOIN_LINES))
    return samples


def _make_sentence_boundary_samples(text: str) -> list[tuple[str, str, int]]:
    pieces = [piece for piece in re.split(r"[。！？?!；;]", re.sub(r"\s+", "", text)) if len(piece) >= 4]
    return [
        (left, right, KEEP_SEPARATOR)
        for left, right in zip(pieces, pieces[1:])
        if left and right
    ]


@lru_cache(maxsize=1)
def _get_model() -> _BoundaryModel:
    return _BoundaryModel.train(_make_training_samples())


def _sigmoid(value: float) -> float:
    if value >= 30:
        return 1.0
    if value <= -30:
        return 0.0
    return 1 / (1 + math.exp(-value))


def _is_number_unit_boundary(left: str, right: str) -> bool:
    return bool(left and right and left[-1].isdigit() and right[0] in "万亿千百年月日元%％")


def _has_prefix(text: str, prefixes: tuple[str, ...]) -> bool:
    return any(text.startswith(prefix) for prefix in prefixes)


def _has_suffix(text: str, suffixes: tuple[str, ...]) -> bool:
    return any(text.endswith(suffix) for suffix in suffixes)


def _looks_like_two_short_independent_segments(left: str, right: str) -> bool:
    if len(left) > 16 or len(right) > 16:
        return False
    if _has_suffix(left, LEFT_CONTINUATION_SUFFIXES) or _has_prefix(right, RIGHT_CONTINUATION_PREFIXES):
        return False
    if left[-1] in LEFT_CONTINUING_PUNCT or right[0] in RIGHT_PUNCT:
        return False
    return True


_TRAINING_PARAGRAPHS = (
    "北京警方近期打掉一个专门针对老年人的诈骗团伙，抓获多名犯罪嫌疑人，涉及多个区的门店。",
    "店员以提供免费按摩和低价足疗券等方式将老年人吸引至店内，随后通过聊天锁定经济条件较好的目标。",
    "嫌疑人虚构各种病症，称如不及时治疗将危及生命，诱骗老人充值高额治疗费用。",
    "多个项目单次收费上万元不等，共计数百名老年人被骗，涉案金额超过三千万元。",
    "网信部门发布专项整治公告，将集中处理涉企侵权信息和虚假不实内容。",
    "平台需要对热点话题中的文本、图片和视频进行采集，并输出可供大模型分析的结构化结果。",
    "网络平台需要对热点话题中的文本图片和视频进行采集，随后由多模态模型完成综合分析。",
    "课程设计需要把图像处理模块的识别结果整理成精简文本，便于后续主题分类和趋势预测。",
    "数据爬取模块会保存微博热搜帖子的文本图片音视频，图像处理模块负责提取其中的可见文字。",
    "受强降雨影响，多地交通出现短时拥堵，相关部门已启动应急响应并加强现场排查。",
    "专家表示，识别网络谣言需要结合文本内容、发布主体、传播路径和多模态证据进行综合判断。",
)

_MANUAL_JOIN_PAIRS = (
    ("北京警方近期打掉一个专门针对老", "年人的诈骗团伙，抓获31名犯罪嫌"),
    ("年人的诈骗团伙，抓获31名犯罪嫌", "疑人，涉及朝阳、顺义、平谷、密"),
    ("疑人，涉及朝阳、顺义、平谷、密", "云4个区20家门店"),
    ("店员以提供免费按摩、低价足疗券等", "方式将老年人吸引至店内，按摩过程"),
    ("方式将老年人吸引至店内，按摩过程", "中通过聊天锁定一些子女不在身边、"),
    ("中通过聊天锁定一些子女不在身边、", "经济条件较好的老年人"),
    ("以其身体状况不好为由，引荐所谓", "的“专家”做免费体检。并虚构各"),
    ("的“专家”做免费体检。并虚构各", "种病症，称如不及时治疗将危及生"),
    ("种病症，称如不及时治疗将危及生", "命，诱骗充值高额治疗费用，涉及"),
    ("命，诱骗充值高额治疗费用，涉及", "肠道清洗、祛湿排毒等多个项目"),
    ("共计400余老年人被骗，涉案金额", "3000万余元"),
)

_MANUAL_KEEP_PAIRS = (
    ("400多名老人遭养生诈骗", "涉案超3000万"),
    ("涉案超3000万", "北京警方近期打掉一个专门针对老年人的诈骗团伙"),
    ("云4个区20家门店", "店员以提供免费按摩、低价足疗券等"),
    ("经济条件较好的老年人", "以其身体状况不好为由"),
    ("肠道清洗、祛湿排毒等多个项目", "每个项目单次收费1万至2万元不等"),
    ("每个项目单次收费1万至2万元不等", "共计400余老年人被骗"),
    ("中国队为什么没进世界杯？", "足球"),
    ("热搜第一", "网友热议"),
    ("涉案超3000万", "自报"),
)
