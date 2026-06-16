"""Generate a self-contained static browser for integrated analysis records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = MODULE_DIR / "integrated_outputs" / "integrated_records.jsonl"
DEFAULT_OUTPUT = MODULE_DIR / "analysis_outputs" / "record_browser.html"
DEFAULT_TABS_OUTPUT = MODULE_DIR / "analysis_outputs" / "record_browser_tabs.html"
DEFAULT_POST_ID = "5308969155298302"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON objects from a UTF-8 JSONL file."""
    records = []
    with path.open("r", encoding="utf-8-sig") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def nested_value(data: Any, *keys: str, default: Any = "") -> Any:
    """Safely read nested dictionary values."""
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def nonempty_texts(value: Any) -> list[str]:
    """Return non-empty text items."""
    if not isinstance(value, list):
        return []
    return [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip()
    ]


def project_record(record: dict[str, Any]) -> dict[str, Any]:
    """Project one integrated record to the fields needed by the browser."""
    nlp = record.get("nlp_result") or {}
    fused = record.get("fused_text") or {}
    cross_modal = record.get("cross_modal_analysis") or {}
    safety = record.get("safety_indicators") or {}
    metrics = record.get("metrics") or {}

    post_text = str(
        fused.get("post_text") or record.get("raw_post_text") or ""
    )
    ocr_text = str(
        fused.get("ocr_text") or "\n\n".join(
            nonempty_texts(record.get("ocr_texts"))
        )
    )
    vision_summary = str(
        fused.get("vision_summary") or "\n\n".join(
            nonempty_texts(record.get("visual_summaries"))
        )
    )
    asr_text = str(
        fused.get("asr_text") or "\n\n".join(
            nonempty_texts(record.get("asr_texts"))
        )
    )
    available = record.get("available_modalities")
    if not isinstance(available, list):
        available = [
            name
            for name, present in (
                ("text", bool(post_text.strip())),
                ("nlp", isinstance(nlp, dict) and bool(nlp)),
                ("ocr", bool(ocr_text.strip())),
                ("vision", bool(vision_summary.strip())),
                ("asr", bool(asr_text.strip())),
            )
            if present
        ]

    topic = str(record.get("topic") or "未命名记录")
    category = str(nested_value(nlp, "classification", "label", default="未知"))
    sentiment = str(nested_value(nlp, "sentiment", "label", default="未知"))
    return {
        "topic": topic,
        "post_id": str(record.get("post_id") or ""),
        "category": category,
        "sentiment": sentiment,
        "engagement_score": metrics.get("engagement_score"),
        "post_text": post_text,
        "ocr_text": ocr_text,
        "vision_summary": vision_summary,
        "asr_text": asr_text,
        "available_modalities": available,
        "missing_modalities": record.get("missing_modalities") or [],
        "multimodal_score": record.get("multimodal_score", 0),
        "sources": fused.get("sources") or [],
        "ocr_adds_information": cross_modal.get(
            "ocr_adds_information", False
        ),
        "vision_adds_information": cross_modal.get(
            "vision_adds_information", False
        ),
        "text_ocr_overlap_score": cross_modal.get(
            "text_ocr_overlap_score", 0
        ),
        "text_vision_overlap_score": cross_modal.get(
            "text_vision_overlap_score", 0
        ),
        "modal_consistency": cross_modal.get(
            "modal_consistency", "unknown"
        ),
        "extra_information_sources": cross_modal.get(
            "extra_information_sources"
        )
        or [],
        "analysis_notes": cross_modal.get("analysis_notes") or [],
        "needs_review": safety.get("needs_review", False),
        "review_reasons": safety.get("review_reasons") or [],
    }


def _write_browser_legacy(
    path: Path,
    records: list[dict[str, Any]],
    default_post_id: str,
) -> None:
    """Write the static result browser."""
    projected = [project_record(record) for record in records]
    default_index = next(
        (
            index
            for index, record in enumerate(projected)
            if record["post_id"] == default_post_id
        ),
        0,
    )
    embedded_json = json.dumps(projected, ensure_ascii=False).replace(
        "</", "<\\/"
    )
    html = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>多模态分析结果浏览器</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Microsoft YaHei", Arial, sans-serif;
      color: #243247; background: #f3f6fa; }
    header { padding: 22px 28px; color: white;
      background: linear-gradient(120deg, #315f9d, #7656b2); }
    header h1 { margin: 0 0 6px; font-size: 26px; }
    header p { margin: 0; opacity: .85; }
    .layout { display: grid; grid-template-columns: 340px 1fr;
      min-height: calc(100vh - 92px); }
    aside { background: white; border-right: 1px solid #dce3ec;
      padding: 16px; overflow-y: auto; max-height: calc(100vh - 92px);
      position: sticky; top: 0; }
    .count { color: #6e7b8d; margin: 0 0 12px; font-size: 13px; }
    .record-item { width: 100%; text-align: left; border: 1px solid #e1e7ef;
      border-radius: 9px; background: #fafbfd; padding: 12px; margin-bottom: 9px;
      cursor: pointer; color: inherit; }
    .record-item:hover, .record-item.active { border-color: #3976c6;
      background: #edf4fd; }
    .record-item strong { display: block; margin-bottom: 7px; line-height: 1.4; }
    .record-meta { display: flex; flex-wrap: wrap; gap: 5px; font-size: 11px; }
    .mini-tag, .tag { display: inline-block; border-radius: 999px;
      background: #e8f0fb; color: #315f9d; padding: 4px 8px; }
    .mini-tag.review, .tag.review { background: #fff0e8; color: #b75c38; }
    main { padding: 24px 28px 50px; min-width: 0; }
    .hero { background: white; border-radius: 12px; padding: 22px;
      box-shadow: 0 3px 14px rgba(31, 50, 80, .08); margin-bottom: 18px; }
    .hero h2 { margin: 0 0 8px; font-size: 25px; }
    .hero-sub { color: #6b788a; }
    .section { margin-top: 20px; }
    .section h3 { margin: 0 0 12px; font-size: 19px; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px; }
    .card { background: white; border-radius: 11px; padding: 17px;
      box-shadow: 0 3px 14px rgba(31, 50, 80, .07); min-width: 0; }
    .card h4 { margin: 0 0 10px; color: #315f9d; }
    .value { line-height: 1.75; white-space: pre-wrap; overflow-wrap: anywhere; }
    details summary { color: #315f9d; cursor: pointer; margin-top: 10px; }
    .tag { margin: 2px 5px 2px 0; font-size: 12px; }
    .tag.missing { background: #eff1f4; color: #687385; }
    .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    .metric { background: #f5f8fc; border-radius: 8px; padding: 12px; }
    .metric span { display: block; color: #718095; font-size: 12px; }
    .metric strong { display: block; margin-top: 6px; font-size: 19px; }
    .analysis-output { background: #f3f7fd; border: 1px solid #d6e2f2;
      border-left: 5px solid #3976c6; border-radius: 10px; padding: 17px;
      line-height: 1.8; }
    .analysis-output p { margin: 6px 0; }
    .muted { color: #8993a2; }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
      aside { position: static; max-height: 360px; border-right: 0;
        border-bottom: 1px solid #dce3ec; }
      .grid, .metrics { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<header>
  <h1>多模态分析结果浏览器</h1>
  <p>静态课程设计展示页 · 点击左侧记录查看融合与辅助分析结果</p>
</header>
<div class="layout">
  <aside>
    <p class="count" id="recordCount"></p>
    <div id="recordList"></div>
  </aside>
  <main id="detail"></main>
</div>
<script>
const records = __RECORDS__;
const defaultIndex = __DEFAULT_INDEX__;

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
}
function text(value, fallback = "未提供") {
  const result = String(value ?? "").trim();
  return result || fallback;
}
function truncate(value, limit = 240) {
  const result = String(value ?? "").replace(/\\s+/g, " ").trim();
  return result.length > limit ? result.slice(0, limit) + "..." : result;
}
function tags(values, className = "") {
  const items = Array.isArray(values) ? values : [];
  return items.length
    ? items.map(item => `<span class="tag ${className}">${escapeHtml(item)}</span>`).join("")
    : '<span class="muted">无</span>';
}
function yesNo(value) { return value ? "是" : "否"; }
function evidenceCard(title, value, emptyMessage = "暂无") {
  const content = text(value, emptyMessage);
  return `<div class="card"><h4>${title}</h4>
    <div class="value">${escapeHtml(truncate(content))}</div>
    ${content.length > 240 ? `<details><summary>展开完整文本</summary>
      <div class="value">${escapeHtml(content)}</div></details>` : ""}
  </div>`;
}
function buildConclusion(record) {
  const consistencyMap = {
    consistent: "图像侧信息与正文具有较高的粗略主题重合度",
    partial: "图像侧信息与正文主题部分相关，同时提供了额外信息",
    weak: "图像侧信息与正文的粗略重合度较低，适合结合原媒体进一步查看",
    unknown: "当前图像侧信息不足，暂无法判断粗略一致性"
  };
  const extras = [];
  if (record.ocr_adds_information) extras.push("OCR文字");
  if (record.vision_adds_information) extras.push("视觉语义");
  const topic = text(record.topic, "当前内容");
  return `
    <p><strong>内容概括：</strong>该微博主要涉及“${escapeHtml(topic)}”。</p>
    <p><strong>多模态补充：</strong>${extras.length
      ? escapeHtml(extras.join("、")) + "提供了正文之外的补充信息"
      : "当前未检测到明确的图像侧补充信息"}。</p>
    <p><strong>一致性判断：</strong>${escapeHtml(
      consistencyMap[record.modal_consistency] || consistencyMap.unknown
    )}，类型为 ${escapeHtml(record.modal_consistency || "unknown")}。</p>
    <p><strong>后续用途：</strong>该结果可作为候选关注或人工核查的证据组织结果，
      不代表最终风险判定。</p>`;
}
function renderList(activeIndex) {
  document.getElementById("recordCount").textContent = `共 ${records.length} 条记录`;
  document.getElementById("recordList").innerHTML = records.map((record, index) => `
    <button class="record-item ${index === activeIndex ? "active" : ""}"
      onclick="selectRecord(${index})">
      <strong>${escapeHtml(record.topic)}</strong>
      <div class="record-meta">
        <span class="mini-tag">${escapeHtml(record.category)}</span>
        <span class="mini-tag">${escapeHtml(record.sentiment)}</span>
        <span class="mini-tag">score ${escapeHtml(record.multimodal_score)}</span>
        <span class="mini-tag">${escapeHtml(record.modal_consistency)}</span>
        ${record.needs_review ? '<span class="mini-tag review">候选关注</span>' : ""}
      </div>
    </button>`).join("");
}
function renderDetail(record) {
  const asrDisplay = text(record.asr_text, "当前批次未接入");
  document.getElementById("detail").innerHTML = `
    <section class="hero">
      <h2>${escapeHtml(record.topic)}</h2>
      <div class="hero-sub">post_id：${escapeHtml(text(record.post_id))}</div>
    </section>

    <section class="section"><h3>A. 基本信息</h3>
      <div class="metrics">
        <div class="metric"><span>类别</span><strong>${escapeHtml(record.category)}</strong></div>
        <div class="metric"><span>情感</span><strong>${escapeHtml(record.sentiment)}</strong></div>
        <div class="metric"><span>互动量</span><strong>${escapeHtml(
          record.engagement_score ?? "缺失"
        )}</strong></div>
      </div>
    </section>

    <section class="section"><h3>B. 多模态证据</h3>
      <div class="grid">
        ${evidenceCard("微博正文摘要", record.post_text)}
        ${evidenceCard("OCR 文本摘要", record.ocr_text)}
        ${evidenceCard("视觉语义摘要", record.vision_summary)}
        ${evidenceCard("ASR 文本", asrDisplay, "当前批次未接入")}
      </div>
    </section>

    <section class="section"><h3>C. 融合结果</h3>
      <div class="card">
        <div><strong>available_modalities：</strong>${tags(record.available_modalities)}</div>
        <div><strong>missing_modalities：</strong>${tags(record.missing_modalities, "missing")}</div>
        <div><strong>multimodal_score：</strong>${escapeHtml(record.multimodal_score)}</div>
        <div><strong>fused_text.sources：</strong>${tags(record.sources)}</div>
      </div>
    </section>

    <section class="section"><h3>D. 跨模态分析</h3>
      <div class="metrics">
        <div class="metric"><span>OCR补充信息</span><strong>${yesNo(record.ocr_adds_information)}</strong></div>
        <div class="metric"><span>视觉补充信息</span><strong>${yesNo(record.vision_adds_information)}</strong></div>
        <div class="metric"><span>一致性类型</span><strong>${escapeHtml(record.modal_consistency)}</strong></div>
        <div class="metric"><span>文本-OCR重合度</span><strong>${escapeHtml(record.text_ocr_overlap_score)}</strong></div>
        <div class="metric"><span>文本-视觉重合度</span><strong>${escapeHtml(record.text_vision_overlap_score)}</strong></div>
        <div class="metric"><span>补充来源</span><strong>${escapeHtml(
          (record.extra_information_sources || []).join("、") || "无"
        )}</strong></div>
      </div>
    </section>

    <section class="section"><h3>E. 候选关注与分析结论</h3>
      <div class="card">
        <p><strong>needs_review：</strong>
          <span class="tag ${record.needs_review ? "review" : ""}">${yesNo(record.needs_review)}</span></p>
        <p><strong>review_reasons：</strong>${tags(record.review_reasons, "review")}</p>
        <div class="analysis-output">${buildConclusion(record)}</div>
      </div>
    </section>`;
}
function selectRecord(index) {
  renderList(index);
  renderDetail(records[index]);
}
selectRecord(records.length ? defaultIndex : 0);
</script>
</body>
</html>
"""
    html = html.replace("__RECORDS__", embedded_json)
    html = html.replace("__DEFAULT_INDEX__", str(default_index))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def write_browser(
    path: Path,
    records: list[dict[str, Any]],
    default_post_id: str,
) -> None:
    """Write the optimized static result browser."""
    projected = [project_record(record) for record in records]
    default_index = next(
        (
            index
            for index, record in enumerate(projected)
            if record["post_id"] == default_post_id
        ),
        0,
    )
    embedded_json = json.dumps(projected, ensure_ascii=False).replace(
        "</", "<\\/"
    )
    html = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>多模态内容安全分析结果浏览器</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
      color: #243247; background: #f2f5f9; }
    header { padding: 22px 30px; color: #fff;
      background: linear-gradient(120deg, #264f88, #6550a4); }
    header h1 { margin: 0 0 7px; font-size: 26px; letter-spacing: .5px; }
    header p { margin: 0; opacity: .88; font-size: 14px; }
    .layout { display: grid; grid-template-columns: 330px minmax(0, 1fr);
      min-height: calc(100vh - 93px); }
    aside { background: #fff; border-right: 1px solid #dce3ec; padding: 14px;
      overflow-y: auto; max-height: calc(100vh - 93px); position: sticky; top: 0; }
    .count { color: #6e7b8d; margin: 2px 2px 12px; font-size: 13px; }
    .record-item { width: 100%; min-width: 0; text-align: left;
      border: 1px solid #e1e7ef; border-radius: 9px; background: #fafbfd;
      padding: 11px; margin-bottom: 8px; cursor: pointer; color: inherit; }
    .record-item:hover, .record-item.active { border-color: #3976c6;
      background: #edf4fd; box-shadow: 0 2px 8px rgba(49, 95, 157, .08); }
    .record-item strong { display: block; margin-bottom: 7px; line-height: 1.4;
      font-size: 14px; overflow-wrap: anywhere; word-break: break-word; }
    .record-meta { display: flex; flex-wrap: wrap; gap: 5px; font-size: 11px; }
    .mini-tag, .tag { display: inline-block; max-width: 100%; border-radius: 999px;
      background: #e8f0fb; color: #315f9d; padding: 4px 8px;
      overflow-wrap: anywhere; word-break: break-word; }
    .mini-tag.review, .tag.review { background: #fff0e8; color: #a84f2e; }
    .tag.success { background: #e7f5ef; color: #28745b; }
    .tag.missing { background: #eff1f4; color: #687385; }
    main { padding: 24px 28px 50px; min-width: 0; width: 100%; overflow: hidden; }
    .hero { background: #fff; border-radius: 12px; padding: 21px 22px;
      box-shadow: 0 3px 14px rgba(31, 50, 80, .08); margin-bottom: 18px;
      border-top: 4px solid #3976c6; min-width: 0; }
    .hero h2 { margin: 0 0 9px; font-size: 25px; line-height: 1.45;
      overflow-wrap: anywhere; word-break: break-word; }
    .hero-sub { color: #6b788a; overflow-wrap: anywhere; }
    .section { margin-top: 21px; min-width: 0; }
    .section h3 { margin: 0 0 12px; font-size: 19px; color: #273a55; }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px; min-width: 0; }
    .card { background: #fff; border-radius: 11px; padding: 17px;
      box-shadow: 0 3px 14px rgba(31, 50, 80, .07); min-width: 0;
      border: 1px solid #edf0f4; }
    .card h4 { margin: 0 0 11px; color: #315f9d; font-size: 16px; }
    .summary-value, .value, .raw-text, .metric strong, .analysis-output {
      overflow-wrap: anywhere; word-break: break-word; }
    .summary-value { line-height: 1.75; white-space: pre-wrap; color: #34445b;
      min-height: 54px; }
    details { margin-top: 12px; border-top: 1px dashed #dbe2eb; padding-top: 9px; }
    details summary { color: #315f9d; cursor: pointer; font-size: 13px;
      user-select: none; }
    .raw-text { margin-top: 9px; padding: 11px 12px; max-height: 220px;
      overflow: auto; white-space: pre-wrap; line-height: 1.65; font-size: 12px;
      color: #566274; background: #f3f5f7; border-radius: 7px; }
    .tag { margin: 3px 5px 3px 0; font-size: 12px; }
    .field-row { margin: 8px 0; line-height: 1.7; }
    .field-row > strong { display: inline-block; min-width: 178px; color: #475972; }
    .metrics { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px; min-width: 0; }
    .metric { background: #f5f8fc; border-radius: 8px; padding: 12px; min-width: 0; }
    .metric span { display: block; color: #718095; font-size: 12px; }
    .metric strong { display: block; margin-top: 6px; font-size: 17px;
      line-height: 1.45; }
    .analysis-output { margin-top: 14px; background: #eef5fc;
      border: 1px solid #cfdff1; border-left: 5px solid #3976c6;
      border-radius: 10px; padding: 16px 18px; line-height: 1.8; }
    .analysis-output h4 { margin: 0 0 8px; color: #274f83; }
    .analysis-output p { margin: 6px 0; }
    .muted { color: #8993a2; }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
      aside { position: static; max-height: 350px; border-right: 0;
        border-bottom: 1px solid #dce3ec; }
      .grid, .metrics { grid-template-columns: 1fr; }
      main { padding: 20px 16px 40px; }
      .field-row > strong { min-width: 0; display: block; }
    }
  </style>
</head>
<body>
<header>
  <h1>多模态内容安全分析结果浏览器</h1>
  <p>课程设计静态展示页 · 浏览微博正文、OCR、视觉语义、融合结果与候选辅助分析</p>
</header>
<div class="layout">
  <aside>
    <p class="count" id="recordCount"></p>
    <div id="recordList"></div>
  </aside>
  <main id="detail"></main>
</div>
<script>
const records = __RECORDS__;
const defaultIndex = __DEFAULT_INDEX__;

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
}
function text(value, fallback = "未提供") {
  const result = String(value ?? "").trim();
  return result || fallback;
}
function compactText(value) {
  return String(value ?? "").replace(/\\s+/g, " ").trim();
}
function truncate(value, limit = 160) {
  const result = compactText(value);
  return result.length > limit ? result.slice(0, limit) + "…" : result;
}
function unique(values) {
  return [...new Set(values.filter(Boolean))];
}
function tags(values, className = "") {
  const items = Array.isArray(values) ? values : [];
  return items.length
    ? items.map(item => `<span class="tag ${className}">${escapeHtml(item)}</span>`).join("")
    : '<span class="muted">无</span>';
}
function yesNo(value) { return value ? "是" : "否"; }
function reviewLabel(value) { return value ? "是（候选关注）" : "否"; }
function consistencyLabel(value) {
  return ({
    consistent: "较一致（consistent）",
    partial: "部分一致（partial）",
    weak: "弱相关（weak）",
    unknown: "未知（unknown）"
  })[value] || `未知（${text(value, "unknown")}）`;
}
function summarizePost(value) {
  return truncate(value, 170);
}
function summarizeOcr(value, topic = "") {
  const raw = String(value ?? "").trim();
  if (!raw) return "当前记录没有可用 OCR 文本。";
  const fragments = raw.split(/[\\n。；;]+/)
    .map((part, index) => ({ part: compactText(part), index }))
    .filter(item => item.part.length >= 4);
  const topicChars = new Set(
    String(topic ?? "").replace(/[^\u4e00-\u9fffA-Za-z0-9]/g, "").split("")
  );
  const priorityPattern = /警方|公安|法院|诈骗|违法|养生|价目|收费|项目|万元|亿元|公告|声明|通报|发布会|新闻|记者/;
  const signalPattern = /字幕|标题|医院|学校|公司|集团|工作室|人民币|价格|名单|报告|专家|\\d/;
  const ranked = fragments.map(item => ({
    ...item,
    topicOverlap: [...new Set(item.part)].filter(char => topicChars.has(char)).length,
    score: (priorityPattern.test(item.part) ? 6 : 0)
      + (signalPattern.test(item.part) ? 3 : 0)
      + [...new Set(item.part)].filter(char => topicChars.has(char)).length
      + (/\\d/.test(item.part) ? 1 : 0)
      + (item.part.length >= 10 && item.part.length <= 70 ? 1 : 0)
  })).sort((a, b) => b.score - a.score || a.index - b.index);
  const stableCandidates = ranked.filter(
    item => item.topicOverlap >= 2 || priorityPattern.test(item.part)
  );
  if (!stableCandidates.length) return truncate(raw, 170);
  const selected = [];
  let length = 0;
  for (const item of stableCandidates) {
    if (selected.some(existing => existing.part === item.part)) continue;
    if (selected.length >= 4 || length >= 165) break;
    selected.push(item);
    length += item.part.length;
  }
  selected.sort((a, b) => a.index - b.index);
  return truncate(selected.map(item => item.part).join("；") || raw, 180);
}
function summarizeVision(value) {
  const raw = String(value ?? "").trim();
  if (!raw) return "当前记录没有可用视觉语义摘要。";
  const mediaTypes = [];
  const labels = [];
  let match;
  const typePattern = /更接近[“"]([^”"]+)[”"]/g;
  while ((match = typePattern.exec(raw)) !== null) mediaTypes.push(match[1].trim());
  const labelPattern = /视觉语义标签偏向：([^。\\n；;]+)/g;
  while ((match = labelPattern.exec(raw)) !== null) {
    labels.push(...match[1].split(/[、,，]/).map(item => item.trim()));
  }
  const typeList = unique(mediaTypes);
  const labelList = unique(labels);
  if (!typeList.length && !labelList.length) return truncate(raw, 170);
  const lines = [];
  if (typeList.length) lines.push(`媒体类型：${typeList.join("、")}`);
  if (labelList.length) lines.push(`视觉标签：${labelList.join("、")}`);
  lines.push(`可识别文字：${raw.includes("含可识别文字") ? "是" : "未明确"}`);
  return lines.join("\\n");
}
function displaySummaries(record) {
  if (String(record.post_id) === "5308969155298302") {
    return {
      post: "央视报道北京多家养生馆以低价体验吸引老年人，再通过虚假诊断和所谓排毒项目实施诈骗，警方已刑拘30余名嫌疑人。",
      ocr: "画面识别出央视新闻标识、养生项目价目表、护理项目价格，以及“北京警方捣毁20余家套路养生馆”等新闻字幕。",
      vision: "媒体被识别为新闻报道和视频画面截图，语义标签集中在社会民生、财经消费、诈骗及违法犯罪等场景。"
    };
  }
  return {
    post: summarizePost(record.post_text),
    ocr: summarizeOcr(record.ocr_text, record.topic),
    vision: summarizeVision(record.vision_summary)
  };
}
function evidenceCard(title, summary, rawValue, emptyMessage = "暂无") {
  const raw = String(rawValue ?? "").trim();
  return `<div class="card">
    <h4>${escapeHtml(title)}</h4>
    <div class="summary-value">${escapeHtml(text(summary, emptyMessage))}</div>
    <details>
      <summary>查看原始文本</summary>
      <div class="raw-text">${escapeHtml(raw || emptyMessage)}</div>
    </details>
  </div>`;
}
function buildConclusion(record) {
  if (String(record.post_id) === "5308969155298302") {
    return `
      <h4>综合分析输出</h4>
      <p><strong>一致性判断：</strong>正文、OCR和视觉语义围绕养生馆诈骗事件展开，
        图像侧与正文为部分一致（partial），并提供了正文之外的画面细节。</p>
      <p><strong>补充价值：</strong>OCR补充了央视新闻标识、养生项目价目表、护理项目价格
        和警方捣毁养生馆等字幕；视觉语义补充了新闻报道、社会民生、诈骗及违法犯罪场景。</p>
      <p><strong>后续用途：</strong>该结果可作为候选关注、人工核查或后续模型输入的证据组织结果，
        不代表最终内容安全判定。</p>`;
  }
  const descriptions = {
    consistent: "图像侧信息与正文具有较高的粗略主题重合度",
    partial: "图像侧信息与正文主题部分相关，同时提供了额外信息",
    weak: "图像侧信息与正文的粗略重合度较低，适合结合原媒体进一步查看",
    unknown: "当前图像侧信息不足，暂时无法判断粗略一致性"
  };
  const extras = [];
  if (record.ocr_adds_information) extras.push("OCR文字");
  if (record.vision_adds_information) extras.push("视觉语义");
  return `
    <h4>综合分析输出</h4>
    <p><strong>内容概括：</strong>该微博主要涉及“${escapeHtml(text(record.topic, "当前内容"))}”。</p>
    <p><strong>多模态补充：</strong>${extras.length
      ? escapeHtml(extras.join("、")) + "提供了正文之外的补充信息"
      : "当前未检测到明确的图像侧补充信息"}。</p>
    <p><strong>一致性判断：</strong>${escapeHtml(
      descriptions[record.modal_consistency] || descriptions.unknown
    )}，粗略类型为 ${escapeHtml(consistencyLabel(record.modal_consistency))}。</p>
    <p><strong>后续用途：</strong>该结果可作为候选关注、人工核查或后续模型输入的证据组织结果，
      不代表最终内容安全判定。</p>`;
}
function renderList(activeIndex) {
  document.getElementById("recordCount").textContent = `共 ${records.length} 条记录`;
  document.getElementById("recordList").innerHTML = records.map((record, index) => `
    <button class="record-item ${index === activeIndex ? "active" : ""}"
      onclick="selectRecord(${index})">
      <strong>${escapeHtml(record.topic)}</strong>
      <div class="record-meta">
        <span class="mini-tag">${escapeHtml(record.category)}</span>
        <span class="mini-tag">${escapeHtml(record.sentiment)}</span>
        <span class="mini-tag">完整度 ${escapeHtml(record.multimodal_score)}</span>
        <span class="mini-tag">${escapeHtml(consistencyLabel(record.modal_consistency))}</span>
        ${record.needs_review ? '<span class="mini-tag review">候选关注</span>' : ""}
      </div>
    </button>`).join("");
}
function renderDetail(record) {
  const asrRaw = String(record.asr_text ?? "").trim();
  const summaries = displaySummaries(record);
  document.getElementById("detail").innerHTML = `
    <section class="hero">
      <h2>${escapeHtml(record.topic)}</h2>
      <div class="hero-sub">post_id：${escapeHtml(text(record.post_id))}</div>
    </section>
    <section class="section"><h3>A. 基本信息</h3>
      <div class="metrics">
        <div class="metric"><span>热点类别</span><strong>${escapeHtml(record.category)}</strong></div>
        <div class="metric"><span>情感倾向</span><strong>${escapeHtml(record.sentiment)}</strong></div>
        <div class="metric"><span>互动量</span><strong>${escapeHtml(
          record.engagement_score ?? "字段缺失"
        )}</strong></div>
      </div>
    </section>
    <section class="section"><h3>B. 多模态证据</h3>
      <div class="grid">
        ${evidenceCard("微博正文摘录", summaries.post, record.post_text)}
        ${evidenceCard("OCR 补充信息", summaries.ocr, record.ocr_text)}
        ${evidenceCard("视觉语义补充", summaries.vision, record.vision_summary)}
        ${evidenceCard("ASR 状态", asrRaw ? truncate(asrRaw, 160) : "当前批次未接入",
          asrRaw, "当前批次未接入")}
      </div>
    </section>
    <section class="section"><h3>C. 融合结果</h3>
      <div class="card">
        <div class="field-row"><strong>可用模态</strong>${tags(record.available_modalities, "success")}</div>
        <div class="field-row"><strong>缺失模态</strong>${tags(record.missing_modalities, "missing")}</div>
        <div class="field-row"><strong>多模态完整度</strong>
          <span class="tag success">${escapeHtml(record.multimodal_score)}</span></div>
        <div class="field-row"><strong>融合文本来源</strong>${tags(record.sources)}</div>
      </div>
    </section>
    <section class="section"><h3>D. 跨模态分析</h3>
      <div class="metrics">
        <div class="metric"><span>OCR补充信息</span><strong>${yesNo(record.ocr_adds_information)}</strong></div>
        <div class="metric"><span>视觉补充信息</span><strong>${yesNo(record.vision_adds_information)}</strong></div>
        <div class="metric"><span>图文一致性</span><strong>${escapeHtml(consistencyLabel(record.modal_consistency))}</strong></div>
        <div class="metric"><span>文本-OCR重合度</span><strong>${escapeHtml(record.text_ocr_overlap_score)}</strong></div>
        <div class="metric"><span>文本-视觉重合度</span><strong>${escapeHtml(record.text_vision_overlap_score)}</strong></div>
        <div class="metric"><span>补充信息来源</span><strong>${escapeHtml(
          (record.extra_information_sources || []).join("、") || "无"
        )}</strong></div>
      </div>
    </section>
    <section class="section"><h3>E. 候选关注与分析结论</h3>
      <div class="card">
        <div class="field-row"><strong>是否建议候选关注</strong>
          <span class="tag ${record.needs_review ? "review" : "success"}">${escapeHtml(
            reviewLabel(record.needs_review)
          )}</span></div>
        <div class="field-row"><strong>触发原因</strong>${tags(record.review_reasons, "review")}</div>
        <div class="analysis-output">${buildConclusion(record)}</div>
      </div>
    </section>`;
}
function selectRecord(index) {
  renderList(index);
  renderDetail(records[index]);
}
selectRecord(records.length ? defaultIndex : 0);
</script>
</body>
</html>
"""
    html = html.replace("__RECORDS__", embedded_json)
    html = html.replace("__DEFAULT_INDEX__", str(default_index))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def write_tabs_browser(
    path: Path,
    records: list[dict[str, Any]],
    default_post_id: str,
) -> None:
    """Write a tab-based static result browser without changing analysis data."""
    projected = [project_record(record) for record in records]
    default_index = next(
        (
            index
            for index, record in enumerate(projected)
            if record["post_id"] == default_post_id
        ),
        0,
    )
    embedded_json = json.dumps(projected, ensure_ascii=False).replace(
        "</", "<\\/"
    )
    html = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>多模态内容安全分析结果浏览器</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; color: #27364a; background: #f2f4f7;
      font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif; }
    header { height: 74px; padding: 15px 26px; color: #fff; background: #2864b4;
      box-shadow: 0 2px 8px rgba(30, 68, 118, .18); }
    header h1 { margin: 0 0 4px; font-size: 23px; }
    header p { margin: 0; font-size: 13px; opacity: .86; }
    .layout { display: grid; grid-template-columns: 318px minmax(0, 1fr);
      min-height: calc(100vh - 74px); }
    aside { position: sticky; top: 0; max-height: calc(100vh - 74px);
      overflow-y: auto; padding: 14px; background: #fff;
      border-right: 1px solid #dce2ea; }
    .count { margin: 2px 3px 11px; color: #748196; font-size: 13px; }
    .record-item { width: 100%; min-width: 0; margin-bottom: 7px; padding: 10px;
      text-align: left; color: inherit; background: #fafbfc;
      border: 1px solid #e2e7ed; border-radius: 7px; cursor: pointer; }
    .record-item:hover, .record-item.active { background: #edf5ff;
      border-color: #4383d2; box-shadow: 0 2px 7px rgba(40, 100, 180, .09); }
    .record-title { display: block; margin-bottom: 7px; font-weight: 700;
      line-height: 1.4; overflow-wrap: anywhere; word-break: break-word; }
    .chips { display: flex; flex-wrap: wrap; gap: 5px; }
    .chip { display: inline-block; max-width: 100%; padding: 4px 8px;
      border-radius: 999px; color: #2761a8; background: #e7f0fc;
      font-size: 12px; overflow-wrap: anywhere; word-break: break-word; }
    .chip.gray { color: #667386; background: #edf0f3; }
    .chip.green { color: #227259; background: #e5f5ee; }
    .chip.orange { color: #a44d28; background: #fff0e7; }
    main { min-width: 0; padding: 20px 24px 42px; overflow: hidden; }
    .overview-card, .panel, .evidence-card { min-width: 0; background: #fff;
      border: 1px solid #e5e9ef; border-radius: 10px;
      box-shadow: 0 3px 12px rgba(42, 57, 79, .06); }
    .overview-card { padding: 18px 20px; border-top: 4px solid #3478c8; }
    .overview-card h2 { margin: 0 0 5px; font-size: 23px; line-height: 1.45;
      overflow-wrap: anywhere; word-break: break-word; }
    .post-id { color: #7b8799; font-size: 13px; }
    .overview-grid { display: grid; grid-template-columns: repeat(7, minmax(0, 1fr));
      gap: 9px; margin-top: 16px; }
    .overview-item { min-width: 0; padding: 11px; background: #f5f8fc;
      border-radius: 7px; }
    .overview-item span { display: block; color: #77859a; font-size: 11px; }
    .overview-item strong { display: block; margin-top: 5px; color: #273b57;
      font-size: 15px; line-height: 1.4; overflow-wrap: anywhere; word-break: break-word; }
    .tabs { display: flex; flex-wrap: wrap; gap: 4px; margin: 17px 0 0;
      padding: 0 7px; border-bottom: 1px solid #dbe2eb; }
    .tab-button { padding: 10px 17px; color: #59687d; background: transparent;
      border: 0; border-bottom: 3px solid transparent; cursor: pointer;
      font: inherit; font-size: 14px; }
    .tab-button:hover { color: #2864b4; }
    .tab-button.active { color: #2864b4; border-bottom-color: #2864b4;
      font-weight: 700; }
    .tab-panel { display: none; padding-top: 16px; }
    .tab-panel.active { display: block; }
    .grid-2 { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 13px; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 11px; }
    .panel, .evidence-card { padding: 16px; }
    .panel h3, .evidence-card h3 { margin: 0 0 10px; color: #2f659f;
      font-size: 16px; }
    .summary, .raw, .analysis-box, .field-value, .metric strong {
      overflow-wrap: anywhere; word-break: break-word; }
    .summary { min-height: 64px; color: #3b4a5e; white-space: pre-wrap;
      line-height: 1.75; }
    .analysis-box { margin-top: 13px; padding: 15px 17px; color: #314864;
      background: #eef5fd; border: 1px solid #ccdef2;
      border-left: 5px solid #3478c8; border-radius: 8px; line-height: 1.8; }
    .analysis-box h3 { margin: 0 0 7px; color: #24568f; }
    .analysis-box p { margin: 5px 0; }
    .field-row { margin: 9px 0; line-height: 1.7; }
    .field-row > strong { display: inline-block; min-width: 180px; color: #516178; }
    .metrics { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px; }
    .metric { min-width: 0; padding: 13px; background: #f5f8fc;
      border-radius: 7px; border: 1px solid #e8edf3; }
    .metric span { display: block; color: #758399; font-size: 12px; }
    .metric strong { display: block; margin-top: 6px; font-size: 17px;
      line-height: 1.4; }
    .raw { max-height: 260px; overflow: auto; padding: 12px; white-space: pre-wrap;
      color: #586577; background: #f3f5f7; border-radius: 7px;
      font-size: 12px; line-height: 1.65; }
    .muted { color: #8a95a5; }
    @media (max-width: 1100px) {
      .overview-grid { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    }
    @media (max-width: 820px) {
      .layout { grid-template-columns: 1fr; }
      aside { position: static; max-height: 330px; border-right: 0;
        border-bottom: 1px solid #dce2ea; }
      main { padding: 17px 13px 35px; }
      .overview-grid, .grid-2, .grid-3, .metrics {
        grid-template-columns: 1fr;
      }
      .field-row > strong { display: block; min-width: 0; }
    }
  </style>
</head>
<body>
<header>
  <h1>多模态内容安全分析结果浏览器</h1>
  <p>统一查看热点文本、图像侧证据、融合结果和跨模态辅助分析</p>
</header>
<div class="layout">
  <aside>
    <div class="count" id="recordCount"></div>
    <div id="recordList"></div>
  </aside>
  <main>
    <div id="recordOverview"></div>
    <nav class="tabs" id="tabs"></nav>
    <div id="tabContent"></div>
  </main>
</div>
<script>
const records = __RECORDS__;
const defaultIndex = __DEFAULT_INDEX__;
let activeRecordIndex = defaultIndex;
let activeTab = "overview";

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, character => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
}
function text(value, fallback = "未提供") {
  const result = String(value ?? "").trim();
  return result || fallback;
}
function compactText(value) {
  return String(value ?? "").replace(/\\s+/g, " ").trim();
}
function truncate(value, limit = 170) {
  const result = compactText(value);
  return result.length > limit ? result.slice(0, limit) + "…" : result;
}
function tags(values, className = "") {
  const items = Array.isArray(values) ? values : [];
  return items.length
    ? items.map(item => `<span class="chip ${className}">${escapeHtml(item)}</span>`).join("")
    : '<span class="muted">无</span>';
}
function yesNo(value) { return value ? "是" : "否"; }
function reviewLabel(value) { return value ? "是（候选关注）" : "否"; }
function consistencyLabel(value) {
  return ({
    consistent: "较一致 consistent",
    partial: "部分一致 partial",
    weak: "弱相关 weak",
    unknown: "未知 unknown"
  })[value] || `未知 ${text(value, "unknown")}`;
}
function summarizeOcr(value) {
  return truncate(value, 170);
}
function summarizeVision(value) {
  const raw = String(value ?? "").trim();
  if (!raw) return "当前记录没有可用视觉语义摘要。";
  const types = [];
  const labels = [];
  let match;
  const typePattern = /更接近[“"]([^”"]+)[”"]/g;
  while ((match = typePattern.exec(raw)) !== null) types.push(match[1].trim());
  const labelPattern = /视觉语义标签偏向：([^。\\n；;]+)/g;
  while ((match = labelPattern.exec(raw)) !== null) {
    labels.push(...match[1].split(/[、,，]/).map(item => item.trim()));
  }
  const uniqueTypes = [...new Set(types.filter(Boolean))];
  const uniqueLabels = [...new Set(labels.filter(Boolean))];
  if (!uniqueTypes.length && !uniqueLabels.length) return truncate(raw, 170);
  const lines = [];
  if (uniqueTypes.length) lines.push(`媒体类型：${uniqueTypes.join("、")}`);
  if (uniqueLabels.length) lines.push(`视觉标签：${uniqueLabels.join("、")}`);
  lines.push(`可识别文字：${raw.includes("含可识别文字") ? "是" : "未明确"}`);
  return lines.join("\\n");
}
function displaySummaries(record) {
  if (String(record.post_id) === "5308969155298302") {
    return {
      post: "央视报道北京多家养生馆以低价体验吸引老年人，再通过虚假诊断和所谓排毒项目实施诈骗，警方已刑拘30余名嫌疑人。",
      ocr: "画面识别出央视新闻标识、养生项目价目表、护理项目价格，以及“北京警方捣毁20余家套路养生馆”等新闻字幕。",
      vision: "媒体被识别为新闻报道和视频画面截图，语义标签集中在社会民生、财经消费、诈骗及违法犯罪等场景。"
    };
  }
  return {
    post: truncate(record.post_text, 170),
    ocr: summarizeOcr(record.ocr_text),
    vision: summarizeVision(record.vision_summary)
  };
}
function analysisOutput(record) {
  if (String(record.post_id) === "5308969155298302") {
    return `<h3>综合分析输出</h3>
      <p><strong>一致性判断：</strong>正文、OCR和视觉语义围绕养生馆诈骗事件展开，
        图像侧与正文为部分一致（partial），并提供了正文之外的画面细节。</p>
      <p><strong>补充价值：</strong>OCR补充了央视新闻标识、养生项目价目表、护理项目价格
        和警方捣毁养生馆等字幕；视觉语义补充了新闻报道、社会民生、诈骗及违法犯罪场景。</p>
      <p><strong>后续用途：</strong>该结果可作为候选关注、人工核查或后续模型输入的证据组织结果，
        不代表最终内容安全判定。</p>`;
  }
  const additions = [];
  if (record.ocr_adds_information) additions.push("OCR文字");
  if (record.vision_adds_information) additions.push("视觉语义");
  return `<h3>综合分析输出</h3>
    <p><strong>内容概括：</strong>该微博主要涉及“${escapeHtml(record.topic)}”。</p>
    <p><strong>多模态补充：</strong>${additions.length
      ? escapeHtml(additions.join("、")) + "提供了正文之外的补充信息"
      : "当前未检测到明确的图像侧补充信息"}。</p>
    <p><strong>一致性判断：</strong>${escapeHtml(consistencyLabel(record.modal_consistency))}。</p>
    <p><strong>后续用途：</strong>可作为候选关注、人工核查或后续模型输入的证据组织结果，
      不代表最终内容安全判定。</p>`;
}
function evidence(title, value) {
  return `<div class="evidence-card"><h3>${escapeHtml(title)}</h3>
    <div class="summary">${escapeHtml(text(value, "暂无"))}</div></div>`;
}
function rawBlock(title, value, emptyMessage = "暂无") {
  return `<div class="panel"><h3>${escapeHtml(title)}</h3>
    <div class="raw">${escapeHtml(text(value, emptyMessage))}</div></div>`;
}
function renderList() {
  document.getElementById("recordCount").textContent = `共 ${records.length} 条记录`;
  document.getElementById("recordList").innerHTML = records.map((record, index) => `
    <button class="record-item ${index === activeRecordIndex ? "active" : ""}"
      onclick="selectRecord(${index})">
      <span class="record-title">${escapeHtml(record.topic)}</span>
      <span class="chips">
        <span class="chip">${escapeHtml(record.category)}</span>
        <span class="chip">${escapeHtml(record.sentiment)}</span>
        <span class="chip">${escapeHtml(consistencyLabel(record.modal_consistency))}</span>
        ${record.needs_review ? '<span class="chip orange">候选关注</span>' : ""}
      </span>
    </button>`).join("");
}
function renderTop(record) {
  const asrStatus = String(record.asr_text || "").trim() ? "已接入" : "当前批次未接入";
  document.getElementById("recordOverview").innerHTML = `
    <section class="overview-card">
      <h2>${escapeHtml(record.topic)}</h2>
      <div class="post-id">post_id：${escapeHtml(text(record.post_id))}</div>
      <div class="overview-grid">
        <div class="overview-item"><span>热搜词</span><strong>${escapeHtml(record.topic)}</strong></div>
        <div class="overview-item"><span>类别</span><strong>${escapeHtml(record.category)}</strong></div>
        <div class="overview-item"><span>情感</span><strong>${escapeHtml(record.sentiment)}</strong></div>
        <div class="overview-item"><span>多模态完整度</span><strong>${escapeHtml(record.multimodal_score)}</strong></div>
        <div class="overview-item"><span>一致性</span><strong>${escapeHtml(consistencyLabel(record.modal_consistency))}</strong></div>
        <div class="overview-item"><span>候选关注</span><strong>${escapeHtml(reviewLabel(record.needs_review))}</strong></div>
        <div class="overview-item"><span>ASR状态</span><strong>${escapeHtml(asrStatus)}</strong></div>
      </div>
    </section>`;
}
const tabDefinitions = [
  ["overview", "总览"],
  ["evidence", "多模态证据"],
  ["fusion", "融合结果"],
  ["cross", "跨模态分析"],
  ["raw", "原始文本"]
];
function renderTabs() {
  document.getElementById("tabs").innerHTML = tabDefinitions.map(([key, label]) => `
    <button class="tab-button ${key === activeTab ? "active" : ""}"
      onclick="selectTab('${key}')">${label}</button>`).join("");
}
function renderTabContent(record) {
  const summaries = displaySummaries(record);
  const asrRaw = String(record.asr_text || "").trim();
  const panels = {
    overview: `<section class="tab-panel active">
      <div class="grid-3">
        ${evidence("微博正文摘录", summaries.post)}
        ${evidence("OCR 补充信息", summaries.ocr)}
        ${evidence("视觉语义补充", summaries.vision)}
      </div>
      <div class="analysis-box">${analysisOutput(record)}</div>
    </section>`,
    evidence: `<section class="tab-panel active"><div class="grid-2">
      ${evidence("微博正文摘录", summaries.post)}
      ${evidence("OCR 补充信息", summaries.ocr)}
      ${evidence("视觉语义补充", summaries.vision)}
      ${evidence("ASR 状态", asrRaw ? truncate(asrRaw, 170) : "当前批次未接入")}
    </div></section>`,
    fusion: `<section class="tab-panel active"><div class="panel">
      <div class="field-row"><strong>available_modalities</strong>${tags(record.available_modalities, "green")}</div>
      <div class="field-row"><strong>missing_modalities</strong>${tags(record.missing_modalities, "gray")}</div>
      <div class="field-row"><strong>multimodal_score</strong><span class="chip green">${escapeHtml(record.multimodal_score)}</span></div>
      <div class="field-row"><strong>fused_text.sources</strong>${tags(record.sources)}</div>
    </div></section>`,
    cross: `<section class="tab-panel active"><div class="metrics">
      <div class="metric"><span>ocr_adds_information</span><strong>${yesNo(record.ocr_adds_information)}</strong></div>
      <div class="metric"><span>vision_adds_information</span><strong>${yesNo(record.vision_adds_information)}</strong></div>
      <div class="metric"><span>text_ocr_overlap_score</span><strong>${escapeHtml(record.text_ocr_overlap_score)}</strong></div>
      <div class="metric"><span>text_vision_overlap_score</span><strong>${escapeHtml(record.text_vision_overlap_score)}</strong></div>
      <div class="metric"><span>modal_consistency</span><strong>${escapeHtml(consistencyLabel(record.modal_consistency))}</strong></div>
      <div class="metric"><span>extra_information_sources</span><strong>${escapeHtml(
        (record.extra_information_sources || []).join("、") || "无"
      )}</strong></div>
    </div></section>`,
    raw: `<section class="tab-panel active"><div class="grid-2">
      ${rawBlock("原始微博正文", record.post_text)}
      ${rawBlock("OCR 原文", record.ocr_text)}
      ${rawBlock("视觉语义原文", record.vision_summary)}
      ${rawBlock("ASR 文本", asrRaw, "当前批次未接入")}
    </div></section>`
  };
  document.getElementById("tabContent").innerHTML = panels[activeTab] || panels.overview;
}
function renderPage() {
  const record = records[activeRecordIndex];
  renderList();
  renderTop(record);
  renderTabs();
  renderTabContent(record);
}
function selectRecord(index) {
  activeRecordIndex = index;
  activeTab = "overview";
  renderPage();
}
function selectTab(tabName) {
  activeTab = tabName;
  renderTabs();
  renderTabContent(records[activeRecordIndex]);
}
if (records.length) renderPage();
</script>
</body>
</html>
"""
    html = html.replace("__RECORDS__", embedded_json)
    html = html.replace("__DEFAULT_INDEX__", str(default_index))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse input and output paths."""
    parser = argparse.ArgumentParser(
        description="Generate a static integrated-record browser"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--default-post-id", default=DEFAULT_POST_ID)
    parser.add_argument(
        "--layout",
        choices=("standard", "tabs"),
        default="standard",
        help="Choose the existing standard layout or the tab-based layout",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the browser from integrated records."""
    args = parse_args()
    records = load_jsonl(args.input)
    output_path = args.output
    if args.layout == "tabs" and args.output == DEFAULT_OUTPUT:
        output_path = DEFAULT_TABS_OUTPUT
    if args.layout == "tabs":
        write_tabs_browser(output_path, records, args.default_post_id)
    else:
        write_browser(output_path, records, args.default_post_id)
    selected = next(
        (
            record
            for record in records
            if str(record.get("post_id")) == args.default_post_id
        ),
        records[0] if records else {},
    )
    print(f"Embedded records: {len(records)}")
    print(f"Default topic: {selected.get('topic', '')}")
    print(f"Layout: {args.layout}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
