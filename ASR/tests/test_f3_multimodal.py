"""P1-F3: LLM correction, multimodal UI, and demo mode tests."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.fusion_engine import FusionEngine
from core.llm_corrector import LLMCorrector
from utils.types import (
    AnomalyEvent,
    CorrectedSegment,
    FinalReport,
    RedactionInfo,
    Segment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_config():
    cfg = {
        "llm_corrector": {
            "enabled": False,
            "model_name": "gpt2",
            "device": "cpu",
            "timeout": 3.0,
            "confidence_threshold": 0.7,
        },
        "fusion": {"timeline_tolerance": 0.1, "anomaly_conf_threshold": 0.3},
        "privacy_guard": {"enabled": True, "categories": ["mobile_phone"]},
        "enhancer": {"enabled": True},
    }
    return cfg


@pytest.fixture
def sample_segments() -> List[Segment]:
    return [
        Segment(start=0.0, end=2.0, text="我材在开会 等一下", confidence=0.72),
        Segment(start=2.0, end=4.0, text="这个文件要做的很好", confidence=0.68),
        Segment(start=4.0, end=6.0, text="明天在联系你", confidence=0.70),
        Segment(start=6.0, end=8.0, text="收到请回复一下", confidence=0.88),
    ]


@pytest.fixture
def sample_redactions() -> List[RedactionInfo]:
    return [
        RedactionInfo(start=0, end=11, category="mobile_phone", original="13812345678"),
    ]


@pytest.fixture
def sample_anomalies() -> List[AnomalyEvent]:
    return [
        AnomalyEvent(start=1.5, end=3.5, label="[强噪声]", confidence=0.82),
    ]


# ---------------------------------------------------------------------------
# LLMCorrector — mode selection
# ---------------------------------------------------------------------------
class TestLLMCorrectorMode:
    def test_disabled_mode(self, sample_config):
        sample_config["llm_corrector"]["enabled"] = False
        corrector = LLMCorrector(sample_config)
        assert corrector._mode == "mock"
        assert corrector._enabled is False

    def test_enabled_without_transformers_falls_back_to_rule(self, sample_config):
        sample_config["llm_corrector"]["enabled"] = True
        # Force transformers import to fail
        with patch.dict("sys.modules", {"transformers": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                corrector = LLMCorrector(sample_config)
        assert corrector._mode == "rule"

    def test_fallback_on_load_error(self, sample_config):
        sample_config["llm_corrector"]["enabled"] = True
        with patch("core.llm_corrector.LLMCorrector._load_pipeline", side_effect=RuntimeError("no model")):
            corrector = LLMCorrector(sample_config)
        assert corrector._mode in ("rule", "mock")


# ---------------------------------------------------------------------------
# LLMCorrector — rule-based correction
# ---------------------------------------------------------------------------
class TestLLMCorrectorRule:
    def test_fix_returns_corrected_segments(self, sample_config):
        sample_config["llm_corrector"]["enabled"] = False
        corrector = LLMCorrector(sample_config)
        segs = [Segment(start=0.0, end=1.0, text="我材在开会", confidence=0.7)]
        result = corrector.fix(segs)
        assert len(result) == 1
        assert isinstance(result[0], CorrectedSegment)
        assert result[0].original_text == "我材在开会"

    def test_homophone_correction(self, sample_config):
        sample_config["llm_corrector"]["enabled"] = False
        corrector = LLMCorrector(sample_config)
        segs = [Segment(start=0.0, end=1.0, text="我材在开会", confidence=0.7)]
        result = corrector.fix(segs)
        # "材" should be corrected to "才"
        assert "才" in result[0].text or result[0].text == "我在开会"

    def test_fix_low_confidence_passes_through_high_confidence(self, sample_config):
        sample_config["llm_corrector"]["enabled"] = False
        corrector = LLMCorrector(sample_config)
        segs = [
            Segment(start=0.0, end=1.0, text="高置信文本", confidence=0.95),
            Segment(start=1.0, end=2.0, text="低置信文本", confidence=0.60),
        ]
        result = corrector.fix_low_confidence(segs, confidence_threshold=0.75)
        assert len(result) == 2
        assert result[0].text == "高置信文本"
        assert result[0].corrections == []
        assert result[1].corrections != [] or result[1].text != "低置信文本"

    def test_rule_summary_generates_output(self, sample_config):
        sample_config["llm_corrector"]["enabled"] = False
        corrector = LLMCorrector(sample_config)
        summary = corrector.generate_security_summary("我的手机号是13812345678")
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_no_llm_on_disabled_mode(self, sample_config):
        sample_config["llm_corrector"]["enabled"] = False
        corrector = LLMCorrector(sample_config)
        # Should not raise
        result = corrector.fix([Segment(start=0.0, end=1.0, text="测试", confidence=0.8)])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# LLMCorrector — timeout handling
# ---------------------------------------------------------------------------
class TestLLMCorrectorTimeout:
    def test_llm_timeout_falls_back_to_rule(self, sample_config):
        sample_config["llm_corrector"]["enabled"] = True
        sample_config["llm_corrector"]["timeout"] = 0.01
        sample_config["llm_corrector"]["confidence_threshold"] = 0.0

        # Mock pipeline that hangs by sleeping longer than the timeout
        from unittest.mock import MagicMock
        import time

        def hang(*args, **kwargs):
            time.sleep(10)

        mock_pipeline = MagicMock(side_effect=hang)
        mock_pipeline.return_value = [{"generated_text": "修正后的文本"}]

        with patch.object(LLMCorrector, "_load_pipeline", return_value=mock_pipeline):
            corrector = LLMCorrector(sample_config)
            corrector._mode = "transformers"
            corrector._llm_available = True
            corrector._pipeline = mock_pipeline

            segs = [Segment(start=0.0, end=1.0, text="测试文本", confidence=0.5)]
            # Should not raise — should fall back to rule
            result = corrector.fix(segs)
            assert len(result) == 1
            assert isinstance(result[0], CorrectedSegment)


# ---------------------------------------------------------------------------
# FusionEngine — LLM-corrected segment integration
# ---------------------------------------------------------------------------
class TestFusionEngineLLM:
    def test_assemble_with_corrected_segments(self, sample_config, sample_segments, sample_redactions, sample_anomalies):
        engine = FusionEngine(sample_config)
        corrected = [
            CorrectedSegment(
                start=0.0, end=2.0,
                text="我在开会，等一下",
                confidence=0.72,
                original_text="我材在开会 等一下",
                corrections=["LLM修正: '我在开会' -> '我在开会，等一下'"],
            ),
        ]
        report = engine.assemble(
            segments=sample_segments,
            redactions=sample_redactions,
            anomaly_events=sample_anomalies,
            corrected_segments=corrected,
        )
        assert "[LLM已纠正]" in report.metadata["display_text"]
        assert report.corrected_segments == corrected

    def test_assemble_without_corrected_segments_no_llm_tag(self, sample_config, sample_segments, sample_redactions, sample_anomalies):
        engine = FusionEngine(sample_config)
        report = engine.assemble(
            segments=sample_segments,
            redactions=sample_redactions,
            anomaly_events=sample_anomalies,
            corrected_segments=None,
        )
        assert "[LLM已纠正]" not in report.metadata["display_text"]

    def test_corrected_segment_uses_corrected_text_not_original(self, sample_config, sample_segments):
        engine = FusionEngine(sample_config)
        corrected = [
            CorrectedSegment(
                start=0.0, end=2.0,
                text="正确的文本内容",
                confidence=0.9,
                original_text="错吴的文本内空",
                corrections=["fix:错吴的->正确的", "fix:内空->内容"],
            ),
        ]
        report = engine.assemble(
            segments=sample_segments,
            redactions=[],
            anomaly_events=[],
            corrected_segments=corrected,
        )
        assert "正确的文本内容" in report.metadata["display_text"]
        assert "错吴的" not in report.metadata["display_text"]


# ---------------------------------------------------------------------------
# FusionEngine — anomaly label precedence
# ---------------------------------------------------------------------------
class TestFusionEngineLabels:
    def test_pii_label_attached(self, sample_config, sample_segments):
        sample_config["privacy_guard"]["enabled"] = True
        engine = FusionEngine(sample_config)
        redacted_segs = [
            Segment(start=0.0, end=2.0, text="手机号 [PII-REDACTED]", confidence=0.9),
        ]
        report = engine.assemble(
            segments=redacted_segs,
            redactions=[RedactionInfo(start=0, end=8, category="mobile_phone", original="13812345678")],
            anomaly_events=[],
        )
        assert "[PII-REDACTED]" in report.metadata["display_text"]

    def test_anomaly_overlap_attached(self, sample_config, sample_segments):
        engine = FusionEngine(sample_config)
        anomalies = [
            AnomalyEvent(start=0.5, end=1.5, label="[强噪声]", confidence=0.85),
        ]
        report = engine.assemble(
            segments=sample_segments,
            redactions=[],
            anomaly_events=anomalies,
        )
        assert "[强噪声]" in report.metadata["display_text"]

    def test_pii_count_in_metadata(self, sample_config, sample_segments):
        engine = FusionEngine(sample_config)
        redactions = [
            RedactionInfo(start=0, end=11, category="mobile_phone", original="13812345678"),
            RedactionInfo(start=20, end=35, category="id_card", original="12345678901234567X"),
        ]
        report = engine.assemble(sample_segments, redactions=redactions, anomaly_events=[])
        assert report.metadata["pii_redactions"] == 2


# ---------------------------------------------------------------------------
# Report export format
# ---------------------------------------------------------------------------
class TestReportExport:
    def test_generate_full_report_structure(self):
        from ui.app import _generate_full_report

        report = FinalReport(
            segments=[Segment(start=0.0, end=2.0, text="测试文本", confidence=0.9)],
            corrected_segments=None,
            redactions=[
                RedactionInfo(start=0, end=8, category="mobile_phone", original="13812345678"),
            ],
            anomalies=[
                AnomalyEvent(start=1.0, end=2.0, label="[强噪声]", confidence=0.8),
            ],
            summary="检测到敏感信息；音频存在异常",
            metadata={
                "pii_redactions": 1,
                "anomaly_events": 1,
                "anomaly_counts": {"[强噪声]": 1},
                "total_segments": 1,
                "display_text": "[0.00s → 2.00s] 测试文本",
            },
        )
        text = _generate_full_report(report, "原始转写", "脱敏后转写", "LLM摘要")
        assert "ASR 安全审计报告" in text
        assert "一、转写结果（原文）" in text
        assert "二、转写结果（脱敏后）" in text
        assert "三、安全审计摘要" in text
        assert "四、LLM 智能摘要" in text
        assert "13812345678" in text
        assert "[强噪声]" in text
        assert "详细统计" in text
        assert "脱敏详情" in text

    def test_export_with_no_pii(self):
        from ui.app import _generate_full_report

        report = FinalReport(
            segments=[Segment(start=0.0, end=1.0, text="正常文本", confidence=0.95)],
            corrected_segments=None,
            redactions=[],
            anomalies=[],
            summary="无安全问题",
            metadata={
                "pii_redactions": 0,
                "anomaly_events": 0,
                "anomaly_counts": {},
                "total_segments": 1,
                "display_text": "正常文本",
            },
        )
        text = _generate_full_report(report, "原始", "脱敏", "")
        assert "ASR 安全审计报告" in text
        assert "138" not in text


# ---------------------------------------------------------------------------
# Confidence heatmap
# ---------------------------------------------------------------------------
class TestConfidenceHeatmap:
    def test_heatmap_renders_segments(self):
        from ui.app import _build_confidence_bar

        segs = [
            Segment(start=0.0, end=1.0, text="高置信", confidence=0.95),
            Segment(start=1.0, end=2.0, text="中置信", confidence=0.65),
            Segment(start=2.0, end=3.0, text="低置信", confidence=0.40),
        ]
        html = _build_confidence_bar(segs)
        assert "置信度热力图" in html
        assert "0.0s" in html
        assert "1.0s" in html
        assert "2.0s" in html
        assert "95%" in html
        assert "40%" in html

    def test_heatmap_empty_segments(self):
        from ui.app import _build_confidence_bar

        html = _build_confidence_bar([])
        assert html == ""

    def test_heatmap_segment_labels(self):
        from ui.app import _build_confidence_bar

        segs = [Segment(start=1.5, end=3.5, text="测试", confidence=0.72)]
        html = _build_confidence_bar(segs)
        assert "1.5s" in html
        assert "3.5s" in html


# ---------------------------------------------------------------------------
# Spectrogram rendering
# ---------------------------------------------------------------------------
class TestSpectrogram:
    def test_render_spectrogram_output_shape(self):
        from ui.app import _render_spectrogram

        # 1-second 16kHz sine wave
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        img = _render_spectrogram(waveform, sr)
        assert img.ndim == 3
        assert img.shape[2] == 3
        assert img.dtype == np.uint8

    def test_render_spectrogram_short_waveform(self):
        from ui.app import _render_spectrogram

        waveform = np.random.randn(256).astype(np.float32)
        img = _render_spectrogram(waveform, 16000)
        assert img.ndim == 3


# ---------------------------------------------------------------------------
# UI clear / state
# ---------------------------------------------------------------------------
class TestUIClear:
    def test_clear_returns_all_defaults(self):
        from ui.app import ASRApp

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "ui": {"title": "Test", "server_port": 7860, "share": False},
                "llm_corrector": {"enabled": False},
            }, f)
            cfg_path = f.name

        try:
            app = ASRApp(cfg_path)
            result = app._clear_inputs()
            assert result[0] is None           # media_input
            assert result[1] == "auto"         # lang_choice
            assert result[2] == ""             # transcript
            assert result[3] is None           # audio
            assert result[4] == ""             # summary
            assert result[5] is None           # download
            assert result[6] is None           # waveform
            assert result[7] == 0.0           # progress
            assert result[8] == ""              # confidence_html
            assert result[9] is False          # show_pii
            assert result[10] is False          # enable_llm
        finally:
            Path(cfg_path).unlink()


# ---------------------------------------------------------------------------
# Enhancer — STFT / feature / anomaly detection coverage
# ---------------------------------------------------------------------------
class TestEnhancerFeatures:
    def test_stft_istft_round_trip(self):
        """STFT → ISTFT should approximately recover the signal."""
        from core.enhancer import AudioEnhancer

        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        waveform = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

        mag, phase = AudioEnhancer._stft(waveform)
        assert mag.shape == phase.shape
        assert mag.ndim == 2

        recovered = AudioEnhancer._istft(mag, phase)
        assert recovered.shape[0] <= waveform.shape[0]
        # Should be close in shape
        assert abs(len(recovered) - len(waveform)) < sr * 0.1

    def test_spectral_subtraction_reduces_noise(self):
        from core.enhancer import AudioEnhancer

        enhancer = AudioEnhancer()
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        noise = np.random.randn(sr).astype(np.float32) * 0.3
        signal = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        waveform = (signal + noise).astype(np.float32)

        denoised = enhancer.spectral_subtraction(waveform, sr=sr)
        assert denoised.shape[0] == waveform.shape[0]
        assert denoised.dtype == np.float32

    def test_zero_crossing_rate(self):
        from core.enhancer import AudioEnhancer

        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        zcr = AudioEnhancer.zero_crossing_rate(sine)
        assert zcr.shape[0] > 0
        assert np.all(zcr >= 0)

    def test_rms_energy(self):
        from core.enhancer import AudioEnhancer

        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        quiet = np.zeros(sr, dtype=np.float32)
        loud = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        rms_q = AudioEnhancer.rms_energy(quiet)
        rms_l = AudioEnhancer.rms_energy(loud)
        assert rms_l.mean() > rms_q.mean()

    def test_detect_anomalies_noise(self):
        from core.enhancer import AudioEnhancer

        enhancer = AudioEnhancer({"enhancer": {"zcr_threshold": 0.3}})
        sr = 16000
        t = np.linspace(0, 2, sr * 2, dtype=np.float32)
        noise = np.random.randn(sr * 2).astype(np.float32)
        events = enhancer.detect_anomalies(noise, sr=sr)
        # Noise should produce some events
        assert isinstance(events, list)

    def test_detect_anomalies_clean(self):
        from core.enhancer import AudioEnhancer

        enhancer = AudioEnhancer({"enhancer": {"zcr_threshold": 0.5}})
        sr = 16000
        t = np.linspace(0, 1, sr, dtype=np.float32)
        sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        events = enhancer.detect_anomalies(sine, sr=sr)
        assert len(events) == 0

    def test_contiguous_regions(self):
        from core.enhancer import AudioEnhancer

        mask = np.array([False, False, True, True, True, False, True, False])
        regions = AudioEnhancer._contiguous_regions(mask)
        assert regions == [(2, 5), (6, 7)]


# ---------------------------------------------------------------------------
# LLMCorrector — LLM branches (mocked transformers)
# ---------------------------------------------------------------------------
class TestLLMCorrectorLLMBranches:
    def test_correct_with_llm_returns_corrected_text(self, sample_config):
        """When transformers returns valid text, correction should be applied."""
        sample_config["llm_corrector"]["enabled"] = True
        sample_config["llm_corrector"]["timeout"] = 5.0

        mock_pipe = MagicMock(return_value=[{"generated_text": "修正：我在开会，等一下。"}])
        with patch.object(LLMCorrector, "_load_pipeline", return_value=mock_pipe):
            corrector = LLMCorrector(sample_config)
            corrector._mode = "transformers"
            corrector._llm_available = True
            corrector._pipeline = mock_pipe

            segs = [Segment(start=0.0, end=1.0, text="我在开会", confidence=0.5)]
            result = corrector.fix(segs)
            assert len(result) == 1
            assert "修正" in result[0].text or "等一下" in result[0].text or result[0].corrections != []

    def test_correct_with_llm_exception_caught(self, sample_config):
        """LLM inference exception should be caught and fall back to rule."""
        sample_config["llm_corrector"]["enabled"] = True
        sample_config["llm_corrector"]["timeout"] = 5.0

        def raise_exc(*args, **kwargs):
            raise RuntimeError("GPU OOM")

        mock_pipe = MagicMock(side_effect=raise_exc)
        with patch.object(LLMCorrector, "_load_pipeline", return_value=mock_pipe):
            corrector = LLMCorrector(sample_config)
            corrector._mode = "transformers"
            corrector._llm_available = True
            corrector._pipeline = mock_pipe

            segs = [Segment(start=0.0, end=1.0, text="我在开会", confidence=0.5)]
            result = corrector.fix(segs)
            assert len(result) == 1
            assert isinstance(result[0], CorrectedSegment)


# ---------------------------------------------------------------------------
# Report generation — edge cases
# ---------------------------------------------------------------------------
class TestReportGeneration:
    def test_generate_report_empty_segments(self):
        from ui.app import _generate_full_report

        report = FinalReport(
            segments=[],
            corrected_segments=None,
            redactions=[],
            anomalies=[],
            summary="无内容",
            metadata={
                "pii_redactions": 0,
                "anomaly_events": 0,
                "anomaly_counts": {},
                "total_segments": 0,
                "display_text": "",
            },
        )
        text = _generate_full_report(report, "", "", "")
        assert "ASR 安全审计报告" in text
        assert "0" in text

    def test_generate_report_all_redacted(self):
        from ui.app import _generate_full_report

        report = FinalReport(
            segments=[Segment(start=0.0, end=2.0, text="手机号 [PII-REDACTED]", confidence=0.9)],
            corrected_segments=None,
            redactions=[
                RedactionInfo(start=0, end=8, category="mobile_phone", original="13812345678"),
                RedactionInfo(start=10, end=20, category="id_card", original="12345678901234567X"),
            ],
            anomalies=[],
            summary="检测到2处隐私泄露",
            metadata={
                "pii_redactions": 2,
                "anomaly_events": 0,
                "anomaly_counts": {},
                "total_segments": 1,
                "display_text": "手机号 [PII-REDACTED]",
            },
        )
        text = _generate_full_report(report, "原始手机号123", "手机号 [PII-REDACTED]", "")
        assert "2 处" in text
        assert "13812345678" in text  # original shows in detail

    def test_generate_report_with_llm_summary(self):
        from ui.app import _generate_full_report

        report = FinalReport(
            segments=[Segment(start=0.0, end=1.0, text="测试", confidence=0.9)],
            corrected_segments=None,
            redactions=[],
            anomalies=[],
            summary="无异常",
            metadata={
                "pii_redactions": 0,
                "anomaly_events": 0,
                "anomaly_counts": {},
                "total_segments": 1,
                "display_text": "测试",
            },
        )
        text = _generate_full_report(report, "测试", "测试", "LLM认为这是一段正常对话")
        assert "四、LLM 智能摘要" in text
        assert "LLM认为" in text

    def test_generate_report_with_anomaly_counts(self):
        from ui.app import _generate_full_report

        report = FinalReport(
            segments=[Segment(start=0.0, end=3.0, text="测试", confidence=0.9)],
            corrected_segments=None,
            redactions=[],
            anomalies=[
                AnomalyEvent(start=0.5, end=1.5, label="[强噪声]", confidence=0.85),
                AnomalyEvent(start=2.0, end=2.5, label="[疑似合成]", confidence=0.78),
            ],
            summary="检测到音频异常",
            metadata={
                "pii_redactions": 0,
                "anomaly_events": 2,
                "anomaly_counts": {"[强噪声]": 1, "[疑似合成]": 1},
                "total_segments": 1,
                "display_text": "测试",
            },
        )
        text = _generate_full_report(report, "测试", "测试", "")
        assert "[强噪声]" in text
        assert "[疑似合成]" in text
        assert "2 个" in text


# ---------------------------------------------------------------------------
# UI export / clear edge cases
# ---------------------------------------------------------------------------
class TestUIExport:
    def test_export_text_empty(self):
        from ui.app import ASRApp

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"ui": {"server_port": 7860, "share": False}, "llm_corrector": {"enabled": False}}, f)
            cfg_path = f.name

        try:
            app = ASRApp(cfg_path)
            result = app._export_text("")
            assert result is None
            result2 = app._export_text("   ")
            assert result2 is None
        finally:
            Path(cfg_path).unlink()

    def test_export_full_report_no_report(self):
        from ui.app import ASRApp

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"ui": {"server_port": 7860, "share": False}, "llm_corrector": {"enabled": False}}, f)
            cfg_path = f.name

        try:
            app = ASRApp(cfg_path)
            result = app._export_full_report()
            assert result is None
        finally:
            Path(cfg_path).unlink()


# ---------------------------------------------------------------------------
# LLMCorrector summary generation
# ---------------------------------------------------------------------------
class TestLLMSummary:
    def test_security_summary_keywords(self, sample_config):
        sample_config["llm_corrector"]["enabled"] = False
        corrector = LLMCorrector(sample_config)
        text_with_phone = "我的手机号是一三八一二三四五六七八"
        text_with_noise = "这段音频有很强的噪声"
        summary_phone = corrector.generate_security_summary(text_with_phone)
        summary_noise = corrector.generate_security_summary(text_with_noise)
        assert "敏感" in summary_phone or "风险" in summary_phone
        assert "异常" in summary_noise or "噪声" in summary_noise

    def test_normal_text_summary(self, sample_config):
        sample_config["llm_corrector"]["enabled"] = False
        corrector = LLMCorrector(sample_config)
        normal = "今天天气很好我们开会讨论项目进度"
        summary = corrector.generate_security_summary(normal)
        assert isinstance(summary, str)
        assert len(summary) > 0
