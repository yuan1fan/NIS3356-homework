"""Shared data types for ASR pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Segment:
    """Timestamped ASR transcript segment."""

    start: float
    end: float
    text: str
    confidence: float = 0.0


@dataclass
class RedactionInfo:
    """Record of a single PII redaction."""

    start: int
    end: int
    category: str
    original: str
    redacted: str = "[PII-REDACTED]"


@dataclass
class CorrectedSegment(Segment):
    """ASR segment with optional LLM corrections."""

    original_text: str = ""
    corrections: List[str] = field(default_factory=list)


@dataclass
class AnomalyEvent:
    """Detected audio anomaly or animal sound event."""

    start: float
    end: float
    label: str
    confidence: float = 0.0


@dataclass
class FinalReport:
    """Assembled multimodal analysis report."""

    segments: List[Segment]
    corrected_segments: Optional[List[CorrectedSegment]]
    redactions: List[RedactionInfo]
    anomalies: List[AnomalyEvent]
    summary: str = ""
    metadata: dict = field(default_factory=dict)
