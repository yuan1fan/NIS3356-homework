const fileInput = document.querySelector("#fileInput");
const dropzone = document.querySelector("#dropzone");
const chooseButton = document.querySelector("#chooseButton");
const fileList = document.querySelector("#fileList");
const form = document.querySelector("#ocrForm");
const runButton = document.querySelector("#runButton");
const emptyState = document.querySelector("#emptyState");
const summaryGrid = document.querySelector("#summaryGrid");
const resultList = document.querySelector("#resultList");
const resultTitle = document.querySelector("#resultTitle");
const copyAllButton = document.querySelector("#copyAllButton");
const progressWrap = document.querySelector("#progressWrap");
const progressBar = document.querySelector("#progressBar");
const progressLabel = document.querySelector("#progressLabel");
const progressValue = document.querySelector("#progressValue");
const resultTemplate = document.querySelector("#resultTemplate");

let selectedFiles = [];
let compactTexts = [];
let progressTicker = null;

chooseButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => setFiles(Array.from(fileInput.files || [])));

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const files = Array.from(event.dataTransfer?.files || []);
  setFiles(files);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFiles.length) {
    setStatus("请先选择图片或视频文件", "0%");
    return;
  }

  setBusy(true);
  setStatus("上传并初始化 OCR", "8%");

  const payload = new FormData(form);
  payload.delete("files");
  selectedFiles.forEach((file) => payload.append("files", file));

  try {
    progressTicker = startProgressTicker(buildProgressStages(form));
    const response = await fetch("/api/ocr", {
      method: "POST",
      body: payload,
    });
    stopProgressTicker();
    setStatus("整理识别结果", "96%");

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "OCR 处理失败");
    }
    renderResults(data);
    setStatus("完成", "100%");
  } catch (error) {
    stopProgressTicker();
    renderError(error);
  } finally {
    setBusy(false);
  }
});

copyAllButton.addEventListener("click", async () => {
  if (!compactTexts.length) return;
  await navigator.clipboard.writeText(compactTexts.join("\n\n"));
  copyAllButton.textContent = "已复制";
  window.setTimeout(() => {
    copyAllButton.textContent = "复制压缩文本";
  }, 1200);
});

function setFiles(files) {
  selectedFiles = files;
  fileInput.value = "";
  fileList.innerHTML = "";
  if (!files.length) return;

  for (const file of files) {
    const item = document.createElement("div");
    item.className = "file-item";
    item.innerHTML = `<strong>${escapeHtml(file.name)}</strong><span>${formatBytes(file.size)}</span>`;
    fileList.appendChild(item);
  }
}

function setBusy(isBusy) {
  runButton.disabled = isBusy;
  progressWrap.hidden = false;
  if (isBusy) {
    runButton.innerHTML = "正在识别...";
  } else {
    runButton.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 3 14 9-14 9V3Z" /></svg>开始识别`;
  }
}

function setStatus(label, percent) {
  progressLabel.textContent = label;
  progressValue.textContent = percent;
  progressBar.style.width = percent;
}

function buildProgressStages(formElement) {
  const data = new FormData(formElement);
  const stages = [
    { label: "上传并读取媒体文件", start: 8, end: 16, seconds: 2 },
    { label: "抽帧与图像预处理", start: 16, end: 34, seconds: hasVideoFile() ? 8 : 3 },
    { label: "PaddleOCR 文本识别", start: 34, end: 58, seconds: hasVideoFile() ? 18 : 8 },
    { label: "多版本结果评分择优", start: 58, end: 68, seconds: 3 },
  ];
  if (data.get("enable_clip") === "on") {
    stages.push({ label: "Chinese-CLIP 视觉分类", start: 68, end: 82, seconds: 8 });
  }
  if (data.get("enable_caption") === "on") {
    stages.push({ label: "按分类判断是否生成一句话描述", start: 82, end: 91, seconds: 8 });
  }
  const lastEnd = stages[stages.length - 1].end;
  stages.push({ label: "压缩文本与整理 JSON", start: Math.min(92, lastEnd), end: 94, seconds: 3 });
  return stages;
}

function startProgressTicker(stages) {
  stopProgressTicker();
  const startedAt = Date.now();
  return window.setInterval(() => {
    const elapsedSeconds = (Date.now() - startedAt) / 1000;
    const stage = currentProgressStage(stages, elapsedSeconds);
    setStatus(stage.label, `${Math.round(stage.value)}%`);
  }, 500);
}

function stopProgressTicker() {
  if (progressTicker) {
    clearInterval(progressTicker);
    progressTicker = null;
  }
}

function currentProgressStage(stages, elapsedSeconds) {
  let cursor = 0;
  for (const stage of stages) {
    const duration = Math.max(0.5, stage.seconds);
    if (elapsedSeconds <= cursor + duration) {
      const ratio = Math.max(0, Math.min(1, (elapsedSeconds - cursor) / duration));
      return {
        label: stage.label,
        value: stage.start + (stage.end - stage.start) * ratio,
      };
    }
    cursor += duration;
  }
  const last = stages[stages.length - 1];
  return { label: last.label, value: Math.min(94, last.end) };
}

function hasVideoFile() {
  return selectedFiles.some((file) => {
    const name = file.name.toLowerCase();
    return file.type.startsWith("video/") || /\.(mp4|webm|avi|mov|mkv|bin)$/.test(name);
  });
}

function renderResults(data) {
  emptyState.hidden = true;
  summaryGrid.hidden = false;
  resultList.innerHTML = "";
  compactTexts = data.results.map((item) => item.ocr_text_compact || "").filter(Boolean);
  copyAllButton.disabled = compactTexts.length === 0;
  copyAllButton.textContent = "复制压缩文本";
  resultTitle.textContent = `完成 ${data.results.length} 个文件`;

  document.querySelector("#metricFiles").textContent = data.results.length;
  document.querySelector("#metricConfidence").textContent = formatConfidence(data.summary.avg_confidence);
  document.querySelector("#metricChars").textContent = data.summary.char_count;
  document.querySelector("#metricReview").textContent = data.summary.needs_review_count;

  for (const item of data.results) {
    resultList.appendChild(createResultCard(item));
  }
}

function createResultCard(item) {
  const node = resultTemplate.content.firstElementChild.cloneNode(true);
  node.querySelector("h3").textContent = item.filename;
  node.querySelector("p").textContent = `${item.media_type} · ${item.source_media}`;

  const badge = node.querySelector(".badge");
  badge.textContent = item.ocr_quality.needs_review ? "需要复核" : "质量正常";
  badge.classList.add(item.ocr_quality.needs_review ? "review" : "good");

  const quality = node.querySelector(".quality-row");
  const chips = [
    `平均置信度 ${formatConfidence(item.ocr_quality.avg_confidence)}`,
    `字符数 ${item.ocr_quality.char_count ?? 0}`,
    `文本块 ${item.ocr_quality.text_block_count ?? 0}`,
    `最佳预处理 ${item.preprocess.best_variant || "-"}`,
    `评分 ${formatScore(item.preprocess.score)}`,
  ];
  if (item.video) {
    chips.push(`视频 ${formatSeconds(item.video.duration_seconds)}`);
    chips.push(`抽帧 ${item.video.sampled_frame_count ?? 0}`);
  }
  quality.innerHTML = chips.map((chip) => `<span>${escapeHtml(chip)}</span>`).join("");

  renderVisualSemantics(node, item.visual_semantics);

  node.querySelector("textarea").value = item.ocr_text_compact || "";

  const strip = node.querySelector(".preview-strip");
  for (const preview of item.previews || []) {
    const image = document.createElement("img");
    image.src = preview.url;
    image.alt = preview.label || "OCR 可视化结果";
    strip.appendChild(image);
  }
  if (!strip.children.length) {
    strip.remove();
  }
  return node;
}

function renderVisualSemantics(node, semantics) {
  const panel = node.querySelector(".visual-panel");
  const title = panel.querySelector(".visual-main strong");
  const tags = panel.querySelector(".visual-tags");
  const caption = panel.querySelector(".visual-caption");
  const summary = panel.querySelector(".visual-summary");
  if (!semantics || !semantics.visual_type) {
    panel.classList.add("muted-panel");
    title.textContent = "未生成";
    tags.innerHTML = "";
    caption.textContent = "";
    summary.textContent = "Chinese-CLIP 视觉分类未启用，或本次处理未返回视觉语义结果。";
    return;
  }
  const type = semantics.visual_type;
  panel.classList.remove("muted-panel");
  panel.hidden = false;
  title.textContent = `${type.label || "-"} · ${formatConfidence(type.confidence)}`;
  tags.innerHTML = (semantics.semantic_tags || [])
    .slice(0, 5)
    .map((tag) => `<span>${escapeHtml(tag.label)} ${formatConfidence(tag.score)}</span>`)
    .join("");
  if (semantics.image_caption?.text) {
    caption.textContent = `一句话描述：${semantics.image_caption.text}`;
    caption.hidden = false;
  } else {
    caption.textContent = "";
    caption.hidden = true;
  }
  summary.textContent = semantics.visual_summary || "";
}

function renderError(error) {
  emptyState.hidden = false;
  summaryGrid.hidden = true;
  resultList.innerHTML = "";
  resultTitle.textContent = "处理失败";
  emptyState.querySelector("p").textContent = error.message || String(error);
  copyAllButton.disabled = true;
  setStatus("失败", "100%");
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb.toFixed(1)} KB`;
  return `${(kb / 1024).toFixed(1)} MB`;
}

function formatConfidence(value) {
  if (typeof value !== "number") return "-";
  return `${Math.round(value * 100)}%`;
}

function formatScore(value) {
  if (typeof value !== "number") return "-";
  return value.toFixed(3);
}

function formatSeconds(value) {
  if (typeof value !== "number") return "-";
  return `${value.toFixed(1)}s`;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
