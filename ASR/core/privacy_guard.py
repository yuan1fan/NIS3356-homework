"""PII detection and redaction engine."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Pattern, Tuple

from utils.types import RedactionInfo, Segment


# ---------------------------------------------------------------------------
# Regex patterns for each PII category
# ---------------------------------------------------------------------------
# Mobile: 11-digit Chinese mobile, require non-digit boundaries (avoids matching
#   inside long digit strings).  e.g. "手机13812345678" -> "13812345678"
# ID card: 18-digit, region[1-9] + 16 digits + checksum X/d
#   e.g. "110101199001011234"
# Bank card: 16-19 digits, require non-digit boundaries (no \b — fails with CJK)
#   e.g. "6222021234567890123"
# Email: standard RFC-compliant pattern
_PII_PATTERNS: Dict[str, str] = {
    # Mobile: 11 digits split 3-1-4-4, optional separator after 4th and 8th position.
    # e.g. 13812345678, 173-4580-1230, 173 4580 1230, 173.4580.1230
    "mobile_phone": r"(?<!\d)1[3-9]\d{1}?[-\s.]?\d{4}[-\s.]?\d{4}(?=\D|$)",
    # Mobile in Chinese numerals: "手机号/电话" prefix + exactly 11 Chinese digits
    # (?:...) non-capturing prefix; (...) captures the 11 digits for correct span
    "mobile_phone_zh": r"(?:手机(?:号|机)|电话|号码)[是为：:：\s]*([幺两二三四五六七八九零一]{11})",
    # ID card: exactly 18 digits optionally followed by X/x
    # Consumes all 18 chars so m.span() is correct; bank_card catches 19+ digit strings
    "id_card": r"(?<!\d)[1-9]\d{17}(?!\d)",
    # Also match 18 digits + X (common real-world format)
    "id_card_x": r"(?<!\d)[1-9]\d{17}[Xx](?!\d)",
    # ID card in Chinese numerals: "身份证号是" prefix + 17 Chinese digits + X/x
    "id_card_zh": r"(?:身份证号是)([幺两二三四五六七八九零一X]{18})",
    # Bank card: 16-19 Arabic digits, must NOT end with X/x (reserved for id_card)
    "bank_card": r"(?<!\d)[1-9]\d{11,18}(?![0-9Xx])",
    # Email: standard RFC-compliant pattern
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
}


def _normalize_chinese_digits(text: str) -> str:
    """Convert Chinese numerals to Arabic digits for display."""
    _CN_MAP = str.maketrans(
        "幺两零一二三四五六七八九",
        "110123456789",
    )
    return text.translate(_CN_MAP)


@dataclass
class PrivacyGuard:
    """Detect and redact personally identifiable information (PII).

    Supported categories (extensible):
    - mobile_phone
    - id_card
    - bank_card
    - email

    Responsibilities:
    - Rule-based detection (regex + lexicon)
    - Text redaction with [PII-REDACTED]
    - Track redaction spans for downstream alignment
    """

    config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        guard_cfg = self.config.get("privacy_guard", {})
        self._enabled = guard_cfg.get("enabled", True)
        self._categories = set(guard_cfg.get("categories", list(_PII_PATTERNS.keys())))
        self._patterns: Dict[str, Pattern[str]] = {}
        for cat in self._categories:
            if cat in _PII_PATTERNS:
                self._patterns[cat] = re.compile(_PII_PATTERNS[cat])

    def _find_all(self, text: str) -> List[Tuple[int, int, str, str]]:
        """Scan all enabled patterns and return sorted spans.

        Returns:
            List of (start, end, category, matched_text) sorted by start.
        """
        spans: List[Tuple[int, int, str, str]] = []

        priority_patterns = [
            (cat, pat)
            for cat, pat in self._patterns.items()
            if cat in ("id_card", "id_card_x", "id_card_zh")
        ]
        other_patterns = [
            (cat, pat)
            for cat, pat in self._patterns.items()
            if cat not in ("id_card", "id_card_x", "id_card_zh")
        ]

        for cat, pat in priority_patterns + other_patterns:
            for m in pat.finditer(text):
                start, end = m.start(), m.end()
                if cat in ("mobile_phone_zh", "id_card_zh"):
                    # Capture group (1) holds the digit portion only; extract it
                    captured = m.group(1)
                    digit_start = start + len(m.group()) - len(captured)
                    spans.append((digit_start, digit_start + len(captured), cat, captured))
                elif cat == "id_card":
                    # Return the full 18-char span from the text
                    spans.append((start, end, cat, text[start:end]))
                else:
                    spans.append((start, end, cat, text[start:end]))
        spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        return spans

    def _merge_overlapping(
        self, spans: List[Tuple[int, int, str, str]]
    ) -> List[Tuple[int, int, str, str]]:
        """Remove nested / overlapping spans, keeping longest at each position."""
        if not spans:
            return []
        merged: List[Tuple[int, int, str, str]] = []
        last_end = -1
        for start, end, cat, text in spans:
            if start >= last_end:
                merged.append((start, end, cat, text))
                last_end = end
        return merged

    def redact(self, text: str) -> str:
        """Redact PII from text and return masked string.

        Args:
            text: Input transcript text.

        Returns:
            Redacted text with [PII-REDACTED] replacing sensitive spans.
        """
        if not self._enabled:
            return text
        merged = self._merge_overlapping(self._find_all(text))
        if not merged:
            return text
        # Build result by splicing masked string
        result = []
        cursor = 0
        for start, end, cat, text_matched in merged:
            result.append(text[cursor:start])
            result.append("[PII-REDACTED]")
            cursor = end
        result.append(text[cursor:])
        return "".join(result)

    def analyze(self, text: str) -> List[RedactionInfo]:
        """Return detailed redaction records for audit trail.

        Args:
            text: Input transcript text.

        Returns:
            List of RedactionInfo with start, end, category, original, redacted.
        """
        if not self._enabled:
            return []
        merged = self._merge_overlapping(self._find_all(text))
        return [
            RedactionInfo(
                start=start,
                end=end,
                category=cat.replace("_zh", ""),
                original=_normalize_chinese_digits(orig),
                redacted="[PII-REDACTED]",
            )
            for start, end, cat, orig in merged
        ]

    def redact_segments(self, segments: List[Segment]) -> List[Segment]:
        """Apply redaction to a list of ASR segments.

        Args:
            segments: Timestamped transcript segments.

        Returns:
            New list of segments with text redacted.
        """
        redacted_segs: List[Segment] = []
        for seg in segments:
            redacted_text = self.redact(seg.text)
            if redacted_text == seg.text:
                redacted_segs.append(seg)
            else:
                redacted_segs.append(
                    Segment(
                        start=seg.start,
                        end=seg.end,
                        text=redacted_text,
                        confidence=seg.confidence,
                    )
                )
        return redacted_segs
