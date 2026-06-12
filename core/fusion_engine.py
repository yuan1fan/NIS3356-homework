"""Multimodal fusion engine for timeline alignment and security tagging."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.types import (
    AnomalyEvent,
    CorrectedSegment,
    FinalReport,
    RedactionInfo,
    Segment,
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label precedence (higher number = higher priority)
# ---------------------------------------------------------------------------
_LABEL_PRECEDENCE: Dict[str, int] = {
    "[PII-REDACTED]": 10,
    "[疑似合成]": 8,
    "[强噪声]": 6,
    "[低置信]": 4,
    "[LLM已纠正]": 2,
}


@dataclass
class FusionEngine:
    """Fuse ASR, audio features, privacy tags, and anomaly results.

    Responsibilities:
    - Align ASR segments with audio event timeline
    - Merge privacy redactions and anomaly labels into final report
    - Generate human-readable security audit summary
    - Support threshold filtering and conflict resolution
    """

    config: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        fusion_cfg = self.config.get("fusion", {})
        self._tolerance = float(fusion_cfg.get("timeline_tolerance", 0.1))
        self._anomaly_conf_threshold = float(
            fusion_cfg.get("anomaly_conf_threshold", 0.3)
        )
        self._pii_enabled = bool(
            self.config.get("privacy_guard", {}).get("enabled", True)
        )
        self._anomaly_enabled = bool(
            self.config.get("enhancer", {}).get("enabled", True)
        )

    def assemble(
        self,
        segments: List[Segment],
        redactions: List[RedactionInfo],
        anomaly_events: List[AnomalyEvent],
        corrected_segments: Optional[List[CorrectedSegment]] = None,
    ) -> FinalReport:
        """Build final multimodal report from all module outputs.

        Args:
            segments: Base ASR segments.
            redactions: PII redaction records.
            anomaly_events: Audio anomaly events from enhancer.
            corrected_segments: Optional LLM-corrected segments.

        Returns:
            FinalReport with unified segments, tags, and summary.
        """
        # ── 1. Build aligned text by merging security tags into segments ──
        # Build a set of start-times that have LLM corrections for fast lookup
        corrected_starts: set = set()
        if corrected_segments:
            corrected_starts = {cs.start for cs in corrected_segments if cs.corrections}

        aligned_segments = self._align_and_merge(
            segments,
            redactions,
            anomaly_events,
            corrected_segments,
        )

        # ── 2. Build plain text for display ──
        display_lines: List[str] = []
        for seg in aligned_segments:
            # Build tag list from segment's text (PII sentinel inserted by PrivacyGuard)
            tags: List[str] = []
            if "[PII-REDACTED]" in seg.text or self._segment_has_redaction(seg, redactions):
                tags.append("[PII-REDACTED]")
            if seg.start in corrected_starts:
                tags.append("[LLM已纠正]")

            # Anomaly labels by timeline overlap
            for evt in anomaly_events:
                if evt.confidence < self._anomaly_conf_threshold:
                    continue
                if self._time_overlap(seg.start, seg.end, evt.start, evt.end):
                    if evt.label not in tags:
                        tags.append(evt.label)

            # Format: [t1 → t2] [TAG1] [TAG2] actual_text_body
            line = f"[{seg.start:.2f}s → {seg.end:.2f}s]"
            if tags:
                line += " " + " ".join(tags)
            line += f" {seg.text}"
            display_lines.append(line)

        display_text = "\n".join(display_lines) if display_lines else "（无转写结果）"

        # ── 3. Count labels for summary ──
        pii_count = len(redactions)
        anomaly_counts: Dict[str, int] = {}
        for evt in anomaly_events:
            if evt.confidence >= self._anomaly_conf_threshold:
                anomaly_counts[evt.label] = anomaly_counts.get(evt.label, 0) + 1

        # ── 4. Summary string ──
        summary = self.generate_summary_str(
            pii_count, anomaly_counts, len(segments)
        )

        return FinalReport(
            segments=aligned_segments,
            corrected_segments=corrected_segments,
            redactions=redactions,
            anomalies=anomaly_events,
            summary=summary,
            metadata={
                "pii_redactions": pii_count,
                "anomaly_events": len(anomaly_events),
                "anomaly_counts": anomaly_counts,
                "total_segments": len(segments),
                "display_text": display_text,
            },
        )

    def _align_and_merge(
        self,
        segments: List[Segment],
        redactions: List[RedactionInfo],
        anomaly_events: List[AnomalyEvent],
        corrected_segments: Optional[List[CorrectedSegment]] = None,
    ) -> List[Segment]:
        """Build aligned segments with security metadata for display.

        When corrected_segments is provided, prefer its text over base segments,
        preserving [LLM已纠正] tags for transparency.
        """
        if not corrected_segments:
            return [
                Segment(start=seg.start, end=seg.end, text=seg.text, confidence=seg.confidence)
                for seg in segments
            ]

        # Build lookup by start time (unique enough for our use)
        corrected_map: dict[float, CorrectedSegment] = {
            cs.start: cs for cs in corrected_segments
        }

        result: List[Segment] = []
        for seg in segments:
            cs = corrected_map.get(seg.start)
            if cs is not None:
                text = cs.text
                if cs.corrections:
                    text = f"[LLM已纠正] {text}"
            else:
                text = seg.text

            result.append(
                Segment(start=seg.start, end=seg.end, text=text, confidence=seg.confidence)
            )
        return result

    def _overlaps(self, seg: Segment, red: RedactionInfo) -> bool:
        """Check if a redaction overlaps with a text segment.

        Since RedactionInfo.start/end are character offsets, we treat
        any redaction as applying to its parent segment (by construction).
        """
        return red.start < red.end

    def _segment_has_redaction(self, seg: Segment, redactions: List[RedactionInfo]) -> bool:
        """Return True if any redaction's original text appears inside seg.text.

        This handles the case where PrivacyGuard has already replaced the PII
        with [PII-REDACTED] and the original substring is gone.
        """
        for red in redactions:
            if red.original and red.original in seg.text:
                return True
        return False

    def _time_overlap(
        self, s1: float, e1: float, s2: float, e2: float, tol: float | None = None
    ) -> bool:
        """Return True if [s1,e1] and [s2,e2] overlap within tolerance."""
        tol = tol if tol is not None else self._tolerance
        return e1 + tol >= s2 and e2 + tol >= s1

    def _collect_tags(self, seg: Segment) -> List[str]:
        """Extract inline security tags from segment text suffix."""
        import re
        tag_pattern = r"\[(?:PII-REDACTED|强噪声|疑似合成|低置信|LLM已纠正)\]"
        tags = re.findall(tag_pattern, seg.text)
        return tags

    def generate_summary(
        self, report: FinalReport
    ) -> str:
        """Produce a concise security audit summary string.

        Args:
            report: Assembled final report.

        Returns:
            Human-readable summary for UI / report.
        """
        return self.generate_summary_str(
            report.metadata.get("pii_redactions", 0),
            report.metadata.get("anomaly_counts", {}),
            report.metadata.get("total_segments", 0),
        )

    @staticmethod
    def generate_summary_str(
        pii_count: int,
        anomaly_counts: Dict[str, int],
        total_segments: int,
    ) -> str:
        """Format summary components into a human-readable string."""
        parts = []
        if pii_count > 0:
            parts.append(f"检测到 {pii_count} 处敏感信息（已脱敏）")
        else:
            parts.append("未检测到敏感信息")
        if anomaly_counts:
            labels = "、".join(
                f"{label} {count} 次" for label, count in anomaly_counts.items()
            )
            parts.append(f"音频异常：{labels}")
        else:
            parts.append("音频特征正常")
        parts.append(f"共 {total_segments} 个转写片段")
        return "；".join(parts)
