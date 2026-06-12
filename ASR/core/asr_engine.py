"""Baseline ASR engine with timestamped transcription."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import whisper
import zhconv  # noqa: F401 — used at runtime for Traditional→Simplified conversion

from utils.audio_utils import check_duration, validate_media
from utils.media_utils import is_video, to_supported_audio
from utils.types import Segment


logger = logging.getLogger(__name__)


class ASREngine:
    """Baseline speech-to-text engine.

    Responsibilities:
    - Load ASR model (Whisper / Paraformer / etc.)
    - Accept audio/video input and produce timestamped segments
    - Provide confidence scores per segment
    """

    def __init__(self, config: dict) -> None:
        """Initialize engine with configuration.

        Args:
            config: Model path, device, language, etc.
        """
        self.config = config.get("asr", {})
        self.model_name = self.config.get("model_name", "base")
        self.device = self.config.get("device", "cpu")
        # None = Whisper auto-detects language; string like "zh"/"en" forces specific language.
        self.language = self.config.get("language", None)
        self.beam_size = int(self.config.get("beam_size", 5))
        self.simplify_chinese = bool(self.config.get("simplify_chinese", False))
        self._model = None

    def _load_model(self):
        """Lazy-load Whisper model."""
        if self._model is not None:
            return self._model
        logger.info("Loading Whisper model: %s on %s", self.model_name, self.device)
        self._model = whisper.load_model(self.model_name, device=self.device)
        return self._model

    def _prepare_input(self, media_path: str | Path) -> str:
        """Return an audio file path suitable for ASR.

        Video files are converted to temporary WAV.
        Audio files are returned as-is.

        Args:
            media_path: Path to media file.

        Returns:
            Path to WAV file ready for transcription.
        """
        path = Path(media_path)
        if is_video(path):
            audio_path, _, _ = to_supported_audio(path, target_sr=16000, mono=True)
            return audio_path
        return str(path)

    def transcribe(
        self,
        media_path: str,
        *,
        language: str | None = None,
        beam_size: int | None = None,
    ) -> List[Segment]:
        """Transcribe audio/video and return timestamped segments.

        Args:
            media_path: Path to audio or video file.
            language: ISO language code (default from config).
            beam_size: Beam search width (default from config).

        Returns:
            List of Segment with start, end, text, confidence.
        """
        language = language or self.language
        beam_size = beam_size or self.beam_size

        validate_media(media_path)
        check_duration(media_path, max_seconds=300.0)

        audio_input = self._prepare_input(media_path)

        model = self._load_model()
        logger.info(
            "Transcribing: %s (lang=%s, beam=%s)", media_path, language, beam_size
        )

        result = model.transcribe(
            str(audio_input),
            language=language,
            beam_size=beam_size,
            verbose=False,
        )

        segments: List[Segment] = []
        for seg in result.get("segments", []):
            avg_logprob = float(seg.get("avg_logprob", -1.0))
            confidence = float(np.exp(avg_logprob))
            confidence = max(0.0, min(1.0, confidence))

            text = seg.get("text", "").strip()
            if self.simplify_chinese:
                text = zhconv.convert(text, "zh-hans")

            segments.append(
                Segment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=text,
                    confidence=confidence,
                )
            )
        return segments

    def health_check(self) -> bool:
        """Return True if model is loaded and ready."""
        try:
            self._load_model()
            return True
        except Exception:
            return False
