"""Gradio-based multimodal UI for ASR + security analysis with LLM correction."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import numpy as np

from core.asr_engine import ASREngine
from core.enhancer import AudioEnhancer
from core.fusion_engine import FusionEngine
from core.llm_corrector import LLMCorrector
from core.privacy_guard import PrivacyGuard
from utils.config import load_config
from utils.logger import setup_logger
from utils.types import AnomalyEvent, CorrectedSegment, FinalReport, RedactionInfo, Segment

# ---------------------------------------------------------------------------
# Demo fixtures
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _render_spectrogram(waveform: np.ndarray, sr: int) -> np.ndarray:
    """Render a spectrogram as a 2D uint8 RGB array for gr.Image.

    Output is capped at MAX_W × MAX_H pixels so Gradio's WebP encoder
    never exceeds the 16383-pixel single-dimension limit.
    """
    from scipy.signal import spectrogram

    MAX_W, MAX_H = 8192, 2048

    nperseg = min(256, max(32, len(waveform) // 4))
    noverlap = nperseg // 2

    freqs, times, Sxx = spectrogram(
        waveform, fs=sr, nperseg=nperseg, noverlap=noverlap, detrend=False,
    )
    # Sxx shape: (n_freq_bins, n_time_frames)
    n_time = Sxx.shape[1]

    # Scale down time axis if it would exceed MAX_W
    if n_time > MAX_W:
        time_scale = MAX_W / n_time
        new_n_time = MAX_W
        new_nperseg = max(1, int(nperseg * time_scale))
        new_noverlap = new_nperseg // 2
        if new_nperseg >= 16:
            freqs, times, Sxx = spectrogram(
                waveform, fs=sr, nperseg=new_nperseg,
                noverlap=new_noverlap, detrend=False,
            )

    # Log-scale magnitude for better visual contrast
    mag = np.maximum(10 * np.log10(Sxx + 1e-10), -80)
    mag_norm = ((mag + 80) / 80 * 255).astype(np.uint8).T
    h, w = mag_norm.shape[:2]

    # Cap height
    if h > MAX_H:
        mag_norm = mag_norm[:MAX_H, :]

    # RGB: dark-blue (low) → cyan → yellow → red (high)
    heatmap = np.zeros((mag_norm.shape[0], mag_norm.shape[1], 3), dtype=np.uint8)
    t = mag_norm.astype(float) / 255.0
    heatmap[:, :, 0] = (np.clip(t * 2.0, 0, 1) * 200 + np.clip((t - 0.5) * 2.0, 0, 1) * 55).astype(np.uint8)
    heatmap[:, :, 1] = (np.clip(t * 2.0, 0, 1) * 180).astype(np.uint8)
    heatmap[:, :, 2] = (np.clip(1 - t * 2.0, 0, 1) * 220).astype(np.uint8)
    return heatmap


def _build_confidence_bar(segments: List[Segment]) -> str:
    """Build an HTML heatmap showing per-segment confidence."""
    if not segments:
        return ""
    rows = []
    for seg in segments:
        conf = seg.confidence
        # colour: red (low) → amber → green (high)
        if conf < 0.5:
            r, g, b = 220, 60, 60
        elif conf < 0.75:
            r, g, b = 240, 160, 40
        else:
            r, g, b = 60, 200, 100
        pct = max(0, min(100, int(conf * 100)))
        label = f"{seg.start:.1f}s–{seg.end:.1f}s"
        rows.append(
            f'<div style="margin:2px 0;display:flex;align-items:center;gap:6px">'
            f'<span style="width:80px;font-size:11px;color:#aaa">{label}</span>'
            f'<div style="flex:1;height:14px;background:#222;border-radius:3px;overflow:hidden">'
            f'<div style="width:{pct}%;height:100%;background:rgb({r},{g},{b});border-radius:3px;'
            f'transition:width 0.3s"></div>'
            f'</div>'
            f'<span style="width:36px;text-align:right;font-size:11px;color:#888">{pct}%</span>'
            f'</div>'
        )
    return (
        '<div style="font-family:sans-serif;padding:4px 0">'
        + '<div style="font-size:12px;color:#666;margin-bottom:6px">置信度热力图</div>'
        + "".join(rows)
        + "</div>"
    )


def _generate_full_report(
    report: FinalReport,
    transcript_raw: str,
    transcript_redacted: str,
    llm_summary: str,
) -> str:
    """Generate a formatted security audit report string."""
    import datetime

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 60,
        "        ASR 安全审计报告  (ASR Security Audit Report)",
        "=" * 60,
        f"生成时间：{now}",
        "",
        "─" * 60,
        "一、转写结果（原文）",
        "─" * 60,
        transcript_raw or "（无）",
        "",
        "─" * 60,
        "二、转写结果（脱敏后）",
        "─" * 60,
        transcript_redacted or "（无）",
        "",
        "─" * 60,
        "三、安全审计摘要",
        "─" * 60,
        report.summary or "无安全问题",
        "",
    ]
    if llm_summary and llm_summary != report.summary:
        lines += [
            "",
            "─" * 60,
            "四、LLM 智能摘要（参考）",
            "─" * 60,
            llm_summary,
            "",
        ]

    pii_count = report.metadata.get("pii_redactions", 0)
    anomaly_counts = report.metadata.get("anomaly_counts", {})
    total_segs = report.metadata.get("total_segments", 0)
    anomaly_events = report.metadata.get("anomaly_events", 0)

    lines += [
        "─" * 60,
        "五、详细统计",
        "─" * 60,
        f"  · 转写片段总数：{total_segs}",
        f"  · 隐私信息泄露点：{pii_count} 处",
        f"  · 音频异常事件：{anomaly_events} 个",
    ]
    if anomaly_counts:
        for label, count in anomaly_counts.items():
            lines.append(f"      - {label}：{count} 次")

    if report.redactions:
        lines += [
            "",
            "─" * 60,
            "六、脱敏详情",
            "─" * 60,
        ]
        seen: set = set()
        for red in report.redactions:
            key = (red.category, red.original)
            if key not in seen:
                seen.add(key)
                lines.append(f"  · 类型：{red.category}  |  原文：{red.original}")

    if report.anomalies:
        lines += [
            "",
            "─" * 60,
            "七、音频异常记录",
            "─" * 60,
        ]
        for evt in report.anomalies:
            lines.append(
                f"  · {evt.label}  |  时间：{evt.start:.2f}s – {evt.end:.2f}s"
                f"  |  置信度：{evt.confidence:.0%}"
            )

    lines += [
        "",
        "=" * 60,
        "报告完毕。",
        "=" * 60,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ASRApp
# ---------------------------------------------------------------------------
class ASRApp:
    """Main application entry point with multimodal UI and LLM correction."""

    def __init__(self, config_path: str) -> None:
        self.config = load_config(config_path)
        self.logger = setup_logger()
        self.asr_engine = ASREngine(self.config)
        self.privacy_guard = PrivacyGuard(self.config)
        self.audio_enhancer = AudioEnhancer(self.config)
        self.fusion_engine = FusionEngine(self.config)
        self.llm_corrector = LLMCorrector(self.config)

    def build_interface(self) -> gr.Blocks:
        """Build and return Gradio interface."""
        ui_cfg = self.config.get("ui", {})

        with gr.Blocks(title=ui_cfg.get("title", "ASR 安全审计系统")) as demo:
            gr.Markdown("## ASR 安全审计模块")
            gr.Markdown(
                "上传音频/视频文件，自动进行语音转写、隐私脱敏、LLM语义纠错、音频异常检测与安全报告生成。"
            )

            # ── 正式分析 Tab ────────────────────────────────────────────────
            with gr.TabItem("🔍 正式分析"):
                with gr.Row():
                    # Left panel: controls
                    with gr.Column(scale=1):
                        media_input = gr.File(
                            type="filepath",
                            label="上传音频/视频文件（wav/mp3/mp4/mov 等）",
                        )
                        lang_choice = gr.Dropdown(
                            choices=[
                                ("自动检测", "auto"),
                                ("中文", "zh"),
                                ("英文", "en"),
                                ("日文", "ja"),
                                ("韩文", "ko"),
                            ],
                            value="auto",
                            label="识别语言",
                        )
                        show_pii_toggle = gr.Checkbox(
                            label="显示隐私信息原文",
                            value=False,
                            info="关闭时以 [已脱敏] 标记隐藏隐私信息。",
                        )
                        enable_llm_toggle = gr.Checkbox(
                            label="启用 LLM 语义纠错",
                            value=False,
                            info="对低置信度片段进行同音字修正和断句补全（可能增加处理时间）。",
                        )
                        transcribe_btn = gr.Button("🚀 开始分析", variant="primary")
                        clear_btn = gr.Button("清空")

                    # Right panel: results
                    with gr.Column(scale=2):
                        # Audio playback + waveform
                        audio_playback = gr.Audio(
                            label="音频播放",
                            interactive=False,
                            autoplay=False,
                        )
                        waveform_image = gr.Image(
                            label="频谱图（点击跳转）",
                            interactive=True,
                            visible=False,
                        )
                        progress_bar = gr.Slider(
                            minimum=0, maximum=100, value=0,
                            label="分析进度", interactive=False,
                            visible=False,
                        )
                        confidence_html = gr.HTML(
                            label="置信度热力图",
                            value="",
                        )
                        security_summary = gr.Textbox(
                            label="安全审计摘要",
                            lines=2, interactive=False,
                        )
                        transcript = gr.Textbox(
                            label="转写结果（含安全标签）",
                            lines=14,
                            placeholder="转写文本和安全标签将显示在这里...",
                        )
                        with gr.Row():
                            export_txt_btn = gr.Button("导出文本（txt）")
                            export_report_btn = gr.Button("导出安全审查报告（txt）")
                        download_file = gr.File(label="下载", interactive=False)

            # ── Event wiring ────────────────────────────────────────────────
            def on_waveform_click(evt: gr.SelectData):
                """Return the clicked time in seconds for audio seeking."""
                return evt.index

            transcribe_btn.click(
                fn=self._process_media,
                inputs=[media_input, lang_choice, show_pii_toggle, enable_llm_toggle],
                outputs=[
                    transcript, audio_playback, security_summary,
                    waveform_image, progress_bar, confidence_html,
                ],
            )

            waveform_image.select(
                fn=on_waveform_click,
                inputs=None,
                outputs=[audio_playback],
            )

            clear_btn.click(
                fn=self._clear_inputs,
                inputs=None,
                outputs=[
                    media_input, lang_choice, transcript,
                    audio_playback, security_summary, download_file,
                    waveform_image, progress_bar, confidence_html,
                    show_pii_toggle, enable_llm_toggle,
                ],
            )

            export_txt_btn.click(
                fn=self._export_text,
                inputs=transcript,
                outputs=download_file,
            )

            export_report_btn.click(
                fn=self._export_full_report,
                inputs=None,
                outputs=download_file,
            )

        return demo

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def _process_media(
        self,
        media_path: str | None,
        lang_choice: str = "auto",
        show_pii: bool = False,
        enable_llm: bool = False,
    ) -> Tuple[
        str, str | None, str,
        Any, float, str,
    ]:
        """Full pipeline: ASR → LLM → privacy → anomaly → fusion → waveform.

        Returns:
            (transcript_text, audio_path, summary,
             waveform_image, progress, confidence_html)
        """
        if not media_path:
            return (
                "请先上传音频或视频文件。",
                None, "", None, 0.0, "",
            )

        try:
            # Step 0: Show loading
            progress = 5.0

            # Step 1: ASR
            lang = None if lang_choice == "auto" else lang_choice
            segments = self.asr_engine.transcribe(media_path, language=lang)
            progress = 35.0

            # Step 2: LLM correction (if enabled)
            corrected_segs: Optional[List[CorrectedSegment]] = None
            llm_summary_text = ""
            if enable_llm and self.llm_corrector is not None:
                corrected_segs = self.llm_corrector.fix_low_confidence(
                    segments,
                    confidence_threshold=0.75,
                )
                transcript_for_summary = " ".join(s.text for s in segments)
                llm_summary_text = self.llm_corrector.generate_security_summary(transcript_for_summary)
            progress = 55.0

            # Step 3: Privacy detection
            redactions: List[RedactionInfo] = []
            for seg in segments:
                redactions.extend(self.privacy_guard.analyze(seg.text))
            progress = 65.0

            # Step 4: Display text (redacted or raw)
            display_segs: List[Segment] = segments
            if not show_pii:
                display_segs = self.privacy_guard.redact_segments(segments)
            progress = 70.0

            # Step 5: Audio anomaly detection
            waveform, sr = self.audio_enhancer.load_waveform(media_path)
            anomaly_events = self.audio_enhancer.detect_anomalies(waveform, sr)
            progress = 80.0

            # Step 6: Spectrogram
            try:
                spec_image = _render_spectrogram(waveform, sr)
            except Exception:  # noqa: BLE001
                spec_image = None
            progress = 85.0

            # Step 7: Fusion
            report = self.fusion_engine.assemble(
                segments=display_segs,
                redactions=redactions,
                anomaly_events=anomaly_events,
                corrected_segments=corrected_segs,
            )
            progress = 95.0

            # Step 8: Confidence heatmap
            conf_html = _build_confidence_bar(segments)
            progress = 100.0

            # Store report for export
            self._last_report = report
            self._last_transcript_raw = " ".join(s.text for s in segments)
            self._last_transcript_redacted = report.metadata.get("display_text", "")
            self._last_llm_summary = llm_summary_text

        except Exception as exc:  # noqa: BLE001
            self.logger.exception("Pipeline failed")
            return (
                f"处理失败: {exc}",
                None, "", None, 0.0, "",
            )

        display_text = report.metadata.get("display_text", "")
        summary = report.summary
        if not display_text:
            display_text = "未识别到有效语音内容。"
        if not summary:
            summary = "无安全问题。"

        return display_text, media_path, summary, spec_image, progress, conf_html

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def _export_text(self, text: str) -> str | None:
        """Export transcript text to a temporary txt file."""
        if not text or not text.strip():
            return None
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8",
        )
        tmp.write(text)
        tmp.close()
        return tmp.name

    def _export_full_report(self) -> str | None:
        """Export the last generated full security audit report."""
        report: FinalReport | None = getattr(self, "_last_report", None)
        if report is None:
            return None
        raw = getattr(self, "_last_transcript_raw", "")
        redacted = getattr(self, "_last_transcript_redacted", "")
        llm_sum = getattr(self, "_last_llm_summary", "")
        text = _generate_full_report(report, raw, redacted, llm_sum)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8",
        )
        tmp.write(text)
        tmp.close()
        return tmp.name

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------
    def _clear_inputs(self):
        """Reset all UI inputs to default state."""
        self._last_report = None
        self._last_transcript_raw = ""
        self._last_transcript_redacted = ""
        self._last_llm_summary = ""
        return [
            None,          # media_input
            "auto",        # lang_choice
            "",            # transcript
            None,          # audio_playback
            "",            # security_summary
            None,          # download_file
            None,          # waveform_image
            0.0,           # progress_bar
            "",            # confidence_html
            False,         # show_pii_toggle
            False,         # enable_llm_toggle
        ]

    # ------------------------------------------------------------------
    # Launch
    # ------------------------------------------------------------------
    def launch(self) -> None:
        """Launch Gradio web server."""
        ui_cfg = self.config.get("ui", {})
        port = int(ui_cfg.get("server_port", 7860))
        share = bool(ui_cfg.get("share", False))
        demo = self.build_interface()
        demo.launch(server_port=port, share=share)


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="ASR Security Audit System")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent.parent / "config" / "default.yaml"),
        help="Path to config file",
    )
    args = parser.parse_args()
    app = ASRApp(args.config)
    app.launch()


if __name__ == "__main__":
    main()
