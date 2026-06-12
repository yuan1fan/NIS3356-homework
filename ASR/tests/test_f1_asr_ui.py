"""Tests for P0-F1: ASR baseline engine, audio utils, and UI skeleton."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import gradio as gr

from core.asr_engine import ASREngine
from ui.app import ASRApp
from utils.audio_utils import check_duration, load_and_resample, validate_media
from utils.media_utils import is_video


@pytest.fixture
def sample_wav(tmp_path: Path) -> str:
    """Generate a 1-second 440Hz sine wave WAV file for testing."""
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t)
    file_path = tmp_path / "test_tone.wav"
    import soundfile as sf

    sf.write(str(file_path), waveform, sr)
    return str(file_path)


@pytest.fixture
def sample_mp3(tmp_path: Path) -> str:
    """Generate a dummy MP3 file by copying the WAV with .mp3 extension."""
    # soundfile may not write mp3 without extra libs; use wav and rename for extension test.
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False, dtype=np.float32)
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t)
    file_path = tmp_path / "test_tone.mp3"
    import soundfile as sf

    sf.write(str(file_path), waveform, sr)
    return str(file_path)


class TestAudioUtils:
    def test_validate_media_success(self, sample_wav: str) -> None:
        validate_media(sample_wav)

    def test_validate_media_missing(self) -> None:
        with pytest.raises(FileNotFoundError):
            validate_media("/nonexistent/path.wav")

    def test_validate_media_unsupported(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "document.txt"
        bad_file.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported media format"):
            validate_media(str(bad_file))

    def test_validate_media_empty(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.wav"
        empty_file.touch()
        with pytest.raises(ValueError, match="empty"):
            validate_media(str(empty_file))

    def test_validate_media_too_small(self, tmp_path: Path) -> None:
        tiny_file = tmp_path / "tiny.wav"
        tiny_file.write_bytes(b"NOT_AUDIO_DATA")
        with pytest.raises(ValueError, match="too small"):
            validate_media(str(tiny_file))

    def test_load_and_resample(self, sample_wav: str) -> None:
        waveform, sr = load_and_resample(sample_wav, target_sr=16000)
        assert sr == 16000
        assert isinstance(waveform, np.ndarray)
        assert waveform.dtype == np.float32
        assert waveform.ndim == 1

    def test_check_duration(self, sample_wav: str) -> None:
        dur = check_duration(sample_wav, max_seconds=5.0)
        assert 0.9 <= dur <= 1.1

    def test_check_duration_exceeds_limit(self, sample_wav: str) -> None:
        with pytest.raises(ValueError, match="exceeds limit"):
            check_duration(sample_wav, max_seconds=0.1)


class TestASREngine:
    @patch("core.asr_engine.whisper")
    def test_transcribe_returns_segments(self, mock_whisper: MagicMock, sample_wav: str) -> None:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "测试语音内容。",
                    "avg_logprob": -0.1,
                }
            ]
        }
        mock_whisper.load_model.return_value = mock_model

        engine = ASREngine({"asr": {"model_name": "base", "device": "cpu", "language": "zh", "beam_size": 5}})
        segments = engine.transcribe(sample_wav)

        assert len(segments) == 1
        assert segments[0].text == "测试语音内容。"
        assert 0.0 <= segments[0].confidence <= 1.0
        assert segments[0].start == 0.0
        assert segments[0].end == 1.0

    def test_transcribe_invalid_file(self) -> None:
        engine = ASREngine({"asr": {"model_name": "base", "device": "cpu"}})
        with pytest.raises((FileNotFoundError, ValueError)):
            engine.transcribe("/nonexistent/file.wav")


class TestMediaUtils:
    def test_video_detection(self, tmp_path: Path) -> None:
        assert is_video(tmp_path / "clip.mp4")
        assert not is_video(tmp_path / "tone.wav")

    def test_validate_media_accepts_video(self, tmp_path: Path) -> None:
        video_path = tmp_path / "clip.mp4"
        video_path.write_bytes(b"f" * 128)
        validate_media(video_path)


class TestASRApp:
    def test_build_interface_returns_blocks(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "ui:\n  title: Test\n  server_port: 7860\n", encoding="utf-8"
        )
        app = ASRApp(str(config_path))
        interface = app.build_interface()
        assert interface is not None
        assert isinstance(interface, type(gr.Blocks()))

    def test_process_media_empty_input(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "ui:\n  title: Test\n  server_port: 7860\n", encoding="utf-8"
        )
        app = ASRApp(str(config_path))
        text, playback, summary, *_ = app._process_media(None)
        assert "请先上传" in text
        assert playback is None
