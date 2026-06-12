"""Local LLM-based semantic post-correction for ASR transcripts."""

from __future__ import annotations

import logging
import re
import threading
from typing import List, Optional

from utils.types import CorrectedSegment, Segment

logger = logging.getLogger(__name__)

_HOMOPHONE_MAP = {
    "材": "才", "在": "再", "的": "得", "地": "得",
    "做": "作", "象": "像", "又": "有", "没": "每",
    "因": "应", "和": "或", "以": "已", "这": "这么",
}

_CORRECTION_RULES: List[tuple] = [
    # Extra spaces around punctuation (Chinese + English)
    (re.compile(r"[，。！？；：、''\"\u201c\u201d()（）【】]+"), r" "),
    (re.compile(r" +"), " "),
    (re.compile(r"^ +| +$"), ""),
    # Fragment completion — if line ends with a word character, add a full stop
    (re.compile(r"(\w+)$"), r"\1。"),
    # Repeated punctuation collapse
    (re.compile(r"[。！？]{3,}"), "。"),
    (re.compile(r"[,，]{3,}"), "，"),
]


class _TimeoutResult:
    """Holds result from a thread, supporting timeout sentinel."""
    def __init__(self) -> None:
        self.value: object = _TIMEOUT_SENTINEL
        self.exc: Optional[Exception] = None


_TIMEOUT_SENTINEL = object()


def _run_with_timeout(target, args, timeout_sec: float, result_holder: _TimeoutResult) -> None:
    """Execute target(*args) in a thread, store result or exception in holder."""
    try:
        result_holder.value = target(*args)
    except Exception as exc:  # noqa: BLE001
        result_holder.exc = exc


class LLMCorrector:
    """Lightweight LLM post-correction for ASR outputs.

    Modes (in priority order):
      1. transformers — local causal-LM via HuggingFace pipeline (quantized, GPU-friendly)
      2. rule         — fast heuristic corrections (homophone, punctuation, segmentation)
      3. mock         — echo input unchanged (for CI / disabled mode)

    Timeout behaviour: if inference exceeds `timeout` seconds, falls back to
    rule-based correction without raising an exception.
    """

    def __init__(self, config: dict) -> None:
        llm_cfg = config.get("llm_corrector", {})
        self._enabled = bool(llm_cfg.get("enabled", False))
        self._mode: str = "disabled"
        self._timeout = float(llm_cfg.get("timeout", 3.0))
        self._confidence_threshold = float(llm_cfg.get("confidence_threshold", 0.7))
        self._model_name = llm_cfg.get("model_name", "gpt2")
        self._device = llm_cfg.get("device", "cpu")
        self._pipeline = None
        self._llm_available = False

        if not self._enabled:
            self._mode = "mock"
            logger.info("LLM corrector disabled (llm_corrector.enabled=false)")
            return

        # Try transformers pipeline first (GPU if available, else CPU)
        try:
            self._pipeline = self._load_pipeline()
            self._mode = "transformers"
            self._llm_available = True
            logger.info("LLM corrector ready: mode=transformers model=%s", self._model_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("transformers pipeline unavailable (%s); using rule-based fallback", exc)
            self._mode = "rule"
            self._llm_available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fix(self, segments: List[Segment]) -> List[CorrectedSegment]:
        """Correct all segments unconditionally.

        Args:
            segments: Raw ASR segments.

        Returns:
            CorrectedSegment list; every segment is returned (may be unchanged).
        """
        return [self._correct_segment(seg) for seg in segments]

    def fix_low_confidence(
        self,
        segments: List[Segment],
        confidence_threshold: float | None = None,
    ) -> List[CorrectedSegment]:
        """Only correct segments whose confidence falls below the threshold.

        High-confidence segments are passed through without LLM invocation.

        Args:
            segments: Raw ASR segments.
            confidence_threshold: Minimum confidence to skip correction.
                                 Defaults to self._confidence_threshold.

        Returns:
            CorrectedSegment list; corrections may be partial.
        """
        threshold = confidence_threshold if confidence_threshold is not None else self._confidence_threshold
        results: List[CorrectedSegment] = []
        for seg in segments:
            if seg.confidence >= threshold:
                results.append(
                    CorrectedSegment(
                        start=seg.start, end=seg.end,
                        text=seg.text, confidence=seg.confidence,
                        original_text=seg.text, corrections=[],
                    )
                )
            else:
                results.append(self._correct_segment(seg))
        return results

    def generate_security_summary(self, transcript_text: str) -> str:
        """Generate a short security-focused summary from transcript text.

        Falls back to a keyword-based heuristic if LLM is unavailable.

        Args:
            transcript_text: Full concatenated transcript (may contain PII).

        Returns:
            One-sentence security summary in Chinese.
        """
        if self._llm_available and self._mode == "transformers":
            return self._llm_summary(transcript_text)
        return self._rule_summary(transcript_text)

    # ------------------------------------------------------------------
    # Internal — LLM pipeline
    # ------------------------------------------------------------------
    def _load_pipeline(self):
        """Load HuggingFace pipeline for text generation."""
        from transformers import pipeline as hf_pipeline
        return hf_pipeline(
            "text-generation",
            model=self._model_name,
            device=-1 if self._device == "cpu" else 0,
            torch_dtype=None if self._device == "cpu" else "float16",
            max_new_tokens=64,
            temperature=0.3,
            top_p=0.9,
            do_sample=True,
        )

    def _correct_with_llm(self, text: str) -> str:
        """Run LLM inference with self._timeout-second timeout."""
        prompt = (
            "你是一个中文语音转文字后处理器。请修正以下文本中的同音错字、补全断句、添加标点。"
            "只输出修正后的文本，不要解释。\n"
            f"原文：{text}\n修正："
        )
        result_holder = _TimeoutResult()
        thread = threading.Thread(
            target=_run_with_timeout,
            args=(self._pipeline, (prompt,), self._timeout, result_holder),
            daemon=True,
        )
        thread.start()
        thread.join(timeout=self._timeout + 1.0)

        if thread.is_alive() or result_holder.value is _TIMEOUT_SENTINEL:
            logger.warning("LLM inference timed out after %.1fs — falling back to rule-based", self._timeout)
            raise TimeoutError(f"LLM inference exceeded {self._timeout}s")

        if result_holder.exc is not None:
            raise result_holder.exc

        raw = result_holder.value
        if isinstance(raw, list) and len(raw) > 0:
            raw = raw[0].get("generated_text", "")
        elif isinstance(raw, dict):
            raw = raw.get("generated_text", "")

        # Extract correction after the separator
        if "修正：" in raw:
            raw = raw.split("修正：", 1)[1]
        elif "修正：" in text:
            raw = raw.split("修正：", 1)[-1]

        return raw.strip()

    # ------------------------------------------------------------------
    # Internal — rule-based fallback
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_rules(text: str) -> str:
        """Apply heuristic correction rules (homophone + punctuation)."""
        result = text
        for wrong, correct in _HOMOPHONE_MAP.items():
            result = result.replace(wrong, correct)

        # Step 2: punctuation and segmentation fixes
        for pattern, replacement in _CORRECTION_RULES:
            result = pattern.sub(replacement, result)

        return result.strip()

    @staticmethod
    def _rule_summary(text: str) -> str:
        """Keyword-based heuristic summary."""
        pii_keywords = ["手机", "身份证", "银行卡", "密码", "账户", "地址", "姓名", "号码"]
        anomaly_keywords = ["噪声", "噪音", "合成", "模糊"]

        has_pii = any(kw in text for kw in pii_keywords)
        has_anomaly = any(kw in text for kw in anomaly_keywords)
        char_count = len(text)

        parts = []
        if has_pii:
            parts.append("检测到敏感信息暴露风险")
        if has_anomaly:
            parts.append("音频存在异常特征")
        if not parts:
            if char_count > 200:
                parts.append("转写内容正常，未发现明显安全问题")
            else:
                parts.append("音频内容正常，未检测到明显异常")
        return "；".join(parts)

    # ------------------------------------------------------------------
    # Internal — LLM summary
    # ------------------------------------------------------------------
    def _llm_summary(self, text: str) -> str:
        """Generate summary via LLM with timeout."""
        snippet = text[:500] if len(text) > 500 else text
        prompt = (
            "你是一个安全分析师。请根据以下转写文本，生成一句话的中文安全摘要，"
            "指出是否包含敏感信息泄露、音频异常等风险。只输出一句话。\n"
            f"转写内容：{snippet}\n摘要："
        )
        result_holder = _TimeoutResult()
        thread = threading.Thread(
            target=_run_with_timeout,
            args=(self._pipeline, (prompt,), self._timeout, result_holder),
            daemon=True,
        )
        thread.start()
        thread.join(timeout=self._timeout + 1.0)

        if thread.is_alive() or result_holder.value is _TIMEOUT_SENTINEL:
            return self._rule_summary(text)

        raw = result_holder.value
        if isinstance(raw, list) and len(raw) > 0:
            raw = raw[0].get("generated_text", "")
        elif isinstance(raw, dict):
            raw = raw.get("generated_text", "")

        if "摘要：" in raw:
            raw = raw.split("摘要：", 1)[1]
        return raw.strip()[:200]

    # ------------------------------------------------------------------
    # Core correction dispatch
    # ------------------------------------------------------------------
    def _correct_segment(self, seg: Segment) -> CorrectedSegment:
        """Correct a single segment; applies rule-based correction always,
        then LLM correction if available (with timeout fallback)."""
        original_text = seg.text

        # Always apply rules first
        corrected_text = self._apply_rules(original_text)
        corrections: List[str] = []

        if original_text != corrected_text:
            corrections.append(f"规则修正: '{original_text}' -> '{corrected_text}'")

        # LLM correction if available (timeout-safe)
        if self._llm_available and self._mode == "transformers":
            try:
                llm_text = self._correct_with_llm(corrected_text)
                if llm_text and llm_text != corrected_text:
                    corrections.append(f"LLM修正: '{corrected_text}' -> '{llm_text}'")
                    corrected_text = llm_text
            except TimeoutError:
                logger.debug("LLM timeout for segment starting at %.2fs", seg.start)
            except Exception as exc:  # noqa: BLE001
                logger.debug("LLM correction failed: %s", exc)

        return CorrectedSegment(
            start=seg.start,
            end=seg.end,
            text=corrected_text,
            confidence=seg.confidence,
            original_text=original_text,
            corrections=corrections,
        )
