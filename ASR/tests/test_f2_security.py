"""Tests for P0-F2: privacy redaction, audio anomaly detection, and fusion."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.enhancer import AudioEnhancer
from core.fusion_engine import FusionEngine
from core.privacy_guard import PrivacyGuard
from utils.types import AnomalyEvent, RedactionInfo, Segment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_waveform_sine(freq: float, duration: float, sr: int = 16000) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * freq * t)


def _make_waveform_noise(duration: float, sr: int = 16000) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal(int(sr * duration)).astype(np.float32) * 0.5


# ---------------------------------------------------------------------------
# PrivacyGuard tests
# ---------------------------------------------------------------------------

class TestPrivacyGuard:
    @pytest.fixture
    def guard(self) -> PrivacyGuard:
        return PrivacyGuard({})

    def test_mobile_phone(self, guard: PrivacyGuard) -> None:
        text = "我的号码是13812345678，请联系我。"
        result = guard.redact(text)
        assert "[PII-REDACTED]" in result
        assert "138" not in result

    def test_id_card(self, guard: PrivacyGuard) -> None:
        text = "身份证号 110101199001011234 用于验证。"
        result = guard.redact(text)
        assert "[PII-REDACTED]" in result
        assert "110101" not in result

    def test_bank_card(self, guard: PrivacyGuard) -> None:
        text = "银行卡号 6222021234567890123 绑定成功。"
        result = guard.redact(text)
        assert "[PII-REDACTED]" in result
        assert "622202" not in result

    def test_email(self, guard: PrivacyGuard) -> None:
        text = "邮箱 test@example.com 已验证。"
        result = guard.redact(text)
        assert "[PII-REDACTED]" in result
        assert "test@example.com" not in result

    def test_multiple_pii(self, guard: PrivacyGuard) -> None:
        text = "手机13812345678，邮箱 a@b.com，身份证 110101199001011234。"
        result = guard.redact(text)
        assert result.count("[PII-REDACTED]") == 3

    def test_no_pii(self, guard: PrivacyGuard) -> None:
        text = "今天天气很好，适合出门散步。"
        result = guard.redact(text)
        assert result == text

    def test_analyze_returns_spans(self, guard: PrivacyGuard) -> None:
        text = "手机 13912345678 联系我。"
        info = guard.analyze(text)
        assert len(info) == 1
        assert info[0].category == "mobile_phone"
        assert info[0].original == "13912345678"

    def test_redact_segments_preserves_timestamps(self, guard: PrivacyGuard) -> None:
        segments = [
            Segment(start=0.0, end=1.0, text="我的号码是13812345678。", confidence=0.95),
            Segment(start=1.0, end=2.0, text="今天天气不错。", confidence=0.90),
        ]
        redacted = guard.redact_segments(segments)
        assert redacted[0].start == 0.0
        assert redacted[0].end == 1.0
        assert redacted[1].start == 1.0
        assert redacted[1].end == 2.0
        assert "[PII-REDACTED]" in redacted[0].text
        assert redacted[1].text == "今天天气不错。"

    def test_disabled_guard_passes_through(self) -> None:
        guard = PrivacyGuard({"privacy_guard": {"enabled": False}})
        text = "手机13812345678联系我。"
        assert guard.redact(text) == text


# ---------------------------------------------------------------------------
# AudioEnhancer tests
# ---------------------------------------------------------------------------

class TestAudioEnhancer:
    @pytest.fixture
    def enhancer(self) -> AudioEnhancer:
        return AudioEnhancer({})

    def test_load_waveform_audio_file(self, enhancer: AudioEnhancer, tmp_path: Path) -> None:
        import soundfile as sf
        wav = tmp_path / "tone.wav"
        sf.write(str(wav), _make_waveform_sine(440, 1.0), 16000)
        wave, sr = enhancer.load_waveform(str(wav))
        assert sr == 16000
        assert wave.dtype == np.float32

    def test_zcr_pure_sine_low(self, enhancer: AudioEnhancer) -> None:
        wave = _make_waveform_sine(440, 1.0)
        zcr = enhancer.zero_crossing_rate(wave)
        assert zcr.size > 0
        assert np.mean(zcr) < 0.1  # sine should have low ZCR

    def test_zcr_white_noise_high(self, enhancer: AudioEnhancer) -> None:
        wave = _make_waveform_noise(1.0)
        zcr = enhancer.zero_crossing_rate(wave)
        assert zcr.size > 0
        assert np.mean(zcr) > 0.2  # noise should have high ZCR

    def test_spectral_centroid(self, enhancer: AudioEnhancer) -> None:
        wave = _make_waveform_sine(440, 1.0)
        cents = enhancer.spectral_centroid(wave, sr=16000)
        assert cents.size > 0
        assert 200 < float(np.median(cents)) < 800  # 440 Hz sine

    def test_spectral_entropy(self, enhancer: AudioEnhancer) -> None:
        wave = _make_waveform_sine(440, 1.0)
        ent = enhancer.spectral_entropy(wave, sr=16000)
        assert ent.size > 0
        assert np.all(ent > 0)

    def test_detect_anomalies_noise(self, enhancer: AudioEnhancer) -> None:
        # High-amplitude noise should trigger [强噪声]
        wave = _make_waveform_noise(2.0)
        events = enhancer.detect_anomalies(wave, sr=16000)
        labels = [e.label for e in events]
        assert "[强噪声]" in labels

    def test_detect_anomalies_sine_clean(self, enhancer: AudioEnhancer) -> None:
        # Pure sine should not trigger anomalies
        wave = _make_waveform_sine(440, 2.0)
        events = enhancer.detect_anomalies(wave, sr=16000)
        # Clean sine may still have entropy spikes at boundaries
        strong_events = [e for e in events if e.confidence > 0.5]
        assert all(e.label != "[强噪声]" for e in strong_events)


# ---------------------------------------------------------------------------
# FusionEngine tests
# ---------------------------------------------------------------------------

class TestFusionEngine:
    @pytest.fixture
    def fusion(self) -> FusionEngine:
        return FusionEngine({})

    def test_assemble_empty(self, fusion: FusionEngine) -> None:
        report = fusion.assemble([], [], [])
        assert report.metadata["total_segments"] == 0
        assert report.metadata["pii_redactions"] == 0

    def test_assemble_with_pii_redaction(self, fusion: FusionEngine) -> None:
        segments = [
            Segment(start=0.0, end=1.0, text="手机13812345678", confidence=0.9),
        ]
        redactions = [
            RedactionInfo(start=2, end=12, category="mobile_phone", original="13812345678"),
        ]
        report = fusion.assemble(segments, redactions, [])
        text = report.metadata["display_text"]
        assert "[PII-REDACTED]" in text

    def test_assemble_with_anomaly(self, fusion: FusionEngine) -> None:
        segments = [
            Segment(start=0.0, end=1.0, text="请注意风险", confidence=0.9),
        ]
        anomalies = [
            AnomalyEvent(start=0.1, end=0.5, label="[疑似合成]", confidence=0.8),
        ]
        report = fusion.assemble(segments, [], anomalies)
        text = report.metadata["display_text"]
        assert "[疑似合成]" in text

    def test_assemble_combined_tags(self, fusion: FusionEngine) -> None:
        """Verify combined output like '[PII-REDACTED] 请联系我 [疑似合成] 注意风险'."""
        segments = [
            Segment(start=0.0, end=1.0, text="手机13812345678 请联系我", confidence=0.9),
            Segment(start=1.0, end=2.0, text="注意风险", confidence=0.8),
        ]
        redactions = [
            RedactionInfo(start=2, end=12, category="mobile_phone", original="13812345678"),
        ]
        anomalies = [
            AnomalyEvent(start=1.0, end=2.0, label="[疑似合成]", confidence=0.9),
        ]
        report = fusion.assemble(segments, redactions, anomalies)
        display = report.metadata["display_text"]
        # Both tags should appear in their respective segments
        assert "[PII-REDACTED]" in display
        assert "[疑似合成]" in display

    def test_summary_counts(self, fusion: FusionEngine) -> None:
        segments = [
            Segment(start=0.0, end=1.0, text="text", confidence=0.9),
        ]
        redactions = [
            RedactionInfo(start=0, end=5, category="mobile_phone", original="13812345678"),
            RedactionInfo(start=6, end=16, category="email", original="a@b.com"),
        ]
        anomalies = [
            AnomalyEvent(start=0.1, end=0.5, label="[强噪声]", confidence=0.9),
        ]
        report = fusion.assemble(segments, redactions, anomalies)
        assert report.metadata["pii_redactions"] == 2
        assert report.metadata["anomaly_counts"].get("[强噪声]", 0) == 1
        assert "敏感信息" in report.summary

    def test_generate_summary(self, fusion: FusionEngine) -> None:
        summary = fusion.generate_summary_str(
            pii_count=1,
            anomaly_counts={"[强噪声]": 2},
            total_segments=5,
        )
        assert "敏感信息" in summary
        assert "[强噪声]" in summary
        assert "5" in summary
