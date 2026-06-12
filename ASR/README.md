# ASR 安全审计系统 — 用户使用手册

> 适用版本：v1.0 | 课程：信息内容安全 | 维护者：Kevin Chen

---

## 目录

1. [项目简介](#1-项目简介)
2. [创新点说明](#2-创新点说明)
3. [环境安装](#3-环境安装)
4. [一键启动](#4-一键启动)
5. [配置说明](#5-配置说明)
6. [模型下载](#6-模型下载)
7. [常见问题 FAQ](#7-常见问题-faq)

---

## 1. 项目简介

### 1.1 是什么

**ASR 安全审计系统**（ASR Security Audit System）是一个本地部署的多模态语音内容安全分析工具，基于 OpenAI Whisper 语音识别、F2 隐私脱敏、F3 LLM 语义纠错三大模块实现从"语音 → 转写 → 隐私检测 → 脱敏 → 语义纠错 → 安全报告"的全链路自动化处理。

### 1.2 解决什么问题

- **隐私泄露风险**：电话客服、会议录音、语音消息中的手机号、身份证、银行卡等信息被 ASR 完整转写
- **音频质量问题**：背景噪声、回声、疑似合成语音影响转写准确性
- **口语化转写错误**：ASR 固有同音错字、断句不完整问题影响语义理解
- **缺乏安全审计**：无法对大量语音内容做批量隐私合规检查

### 1.3 系统架构

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  F1 ASR    │───▶│  F2 隐私    │───▶│  F3 LLM    │
│  语音转写   │    │  脱敏+异常   │    │  语义纠错   │
└─────────────┘    └─────────────┘    └─────────────┘
                                           │
                                           ▼
                                     ┌─────────────┐
                                     │  融合报告   │
                                     │  Web UI    │
                                     └─────────────┘
```

---

## 2. 创新点说明

### 2.1 如何契合"多模态融合与信息内容安全"课程要求

| 课程要求 | 本项目实现 |
|---------|-----------|
| 多模态信息融合 | 音频时域特征（ZCR/频谱质心）+ 文本语义（PII正则+LLM）+ 时间戳对齐融合 |
| 信息内容安全检测 | 6类PII检测（手机号/身份证/银行卡/邮箱，中英文数字全覆盖）|
| 语音处理与分析 | 零交叉率、频谱质心、频谱熵三类特征异常检测 |
| AI 语义后处理 | LLM（transformers）或规则引擎双模式同音纠错 |
| 工程实现 | Gradio 可视化 Web UI，模型热插拔，无外部依赖 |

### 2.2 核心创新

1. **中文数字读音 PII 检测**：首次实现"幺二三四五六七八九零一"全字集中文数字串的隐私信息识别，覆盖 ASR 中文数字读音输出
2. **多模态标签时间轴对齐**：将 PII 标签、异常事件、LLM 纠错信息统一附加到 ASR 时间戳片段上
3. **LLM 超时优雅降级**：transformers 推理超时自动切换到规则引擎，保证系统在 CPU/GPU 不同环境下均可用
4. **频谱图 WebP 尺寸自适应**：解决长音频频谱图超出 WebP 16383px 上限导致 Gradio 崩溃的问题

---

## 3. 环境安装

### 3.1 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|---------|---------|
| Python | ≥ 3.9 | 3.10–3.12 |
| 内存 | 4 GB | 8 GB+ |
| 磁盘 | 2 GB | 5 GB（含 Whisper 模型）|
| 操作系统 | macOS / Linux / Windows | macOS/Linux |
| GPU | 可选（CPU 可运行全部功能）| NVIDIA GPU + CUDA 加速 |

### 3.2 安装步骤

#### macOS / Linux

```bash
# 1. 克隆项目（或解压）
cd /path/to/ASR

# 2. 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate    # macOS/Linux

# 3. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 4. 验证安装
python3 -c "import whisper; import gradio; print('OK')"
```

#### Windows

```powershell
# 1. 克隆项目
cd C:\path\to\ASR

# 2. 创建虚拟环境
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 4. 验证
python -c "import whisper; print('OK')"
```

### 3.3 requirements.txt 内容

```
gradio>=4.0.0
numpy>=1.24.0
pyyaml>=6.0
soundfile>=0.12.0
librosa>=0.10.0
torch>=2.0.0
transformers>=4.30.0
openai-whisper>=20231117
pytest>=7.0.0
zhconv>=1.2
scipy>=1.10.0
```

---

## 4. 一键启动

### 4.1 启动 Web UI

```bash
# 默认配置（本机访问）
python3 ui/app.py

# 自定义端口
python3 ui/app.py --port 8080

# 打开浏览器访问 http://127.0.0.1:7860
```

### 4.2 启动参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `config/default.yaml` |
| `--port` | Web 服务端口（仅 Python 3.11+） | 7860 |

### 4.3 使用流程

```
1. 打开 http://127.0.0.1:7860
2. 上传音频/视频文件（支持 wav/mp3/mp4/mov/avi 等）
3. 选择识别语言（默认自动检测）
4. 点击「开始分析」按钮
5. 等待处理完成（进度条实时显示）
6. 查看转写文本、置信度热力图、频谱图
7. 可选：导出完整安全报告（.txt）
```

---

## 5. 配置说明

配置文件位于 `config/default.yaml`：

```yaml
asr:                          # F1: 语音识别配置
  model_name: "base"           # Whisper 模型：tiny/base/small/medium/large
  device: "cpu"               # 推理设备："cpu" 或 "cuda"（需 GPU）
  language: null               # null=自动检测；"zh"=强制中文；"en"=强制英文
  beam_size: 5                # Beam size（越大越慢越准确，1-10）
  simplify_chinese: true      # 是否将繁体转换为简体

enhancer:                     # F2: 音频增强配置
  target_sr: 16000            # 重采样目标采样率（Hz）
  denoise: false             # 是否启用谱减法降噪

privacy_guard:                # F2: 隐私检测配置
  enabled: true              # 是否启用隐私检测
  categories:                 # 启用的 PII 检测类别
    - mobile_phone          # 阿拉伯数字手机号
    - mobile_phone_zh       # 中文数字手机号（如"幺三八..."）
    - id_card              # 阿拉伯数字身份证
    - id_card_zh           # 中文数字身份证
    - bank_card            # 银行卡号
    - email                # 邮箱地址

llm_corrector:               # F3: LLM 语义纠错配置
  enabled: false             # 是否启用 LLM 模式（需要 transformers）
  model_name: "gpt2"        # transformers 模型名
  device: "cpu"             # "cpu" 或 "cuda"
  timeout: 3.0              # LLM 推理超时（秒），超时自动降级到规则引擎
  confidence_threshold: 0.7  # 低于此置信度的片段才启用 LLM 纠错

fusion:                      # 融合引擎配置
  timeline_tolerance: 0.1    # 时间轴对齐容差（秒）

ui:                          # Web UI 配置
  title: "ASR 安全审计系统"  # 页面标题
  share: false               # 是否生成公网分享链接
  server_port: 7860          # 服务端口
```

### 5.1 Whisper 模型选择

| 模型 | 参数量 | 磁盘占用 | CPU 推理时间（10s音频）| 推荐场景 |
|------|--------|---------|-------------------|---------|
| `tiny` | 39M | ~75 MB | ~3s | 快速演示、CPU |
| `base` | 74M | ~140 MB | ~6s | 日常使用 |
| `small` | 244M | ~460 MB | ~15s | 高精度需求 |
| `medium` | 769M | ~1.5 GB | ~40s | 研究级精度 |
| `large` | 1550M | ~2.9 GB | ~80s | 最高精度 |

---

## 6. 模型下载

### 6.1 Whisper 模型

Whisper 模型会在首次运行时**自动下载**（从 Hugging Face）。

```bash
# 手动预下载（可选）
python3 -c "import whisper; model = whisper.load_model('base')"
```

> 如果网络受限，可以使用 `pip install faster-whisper` 并在 config 中将 `model_name` 改为 `.onnx` 路径。

### 6.2 LLM 模型（可选）

如需启用 F3 LLM 模式，需要下载 transformers 模型：

```bash
# 推荐使用 gpt2（最小，~500MB）
python3 -c "from transformers import pipeline; pipe = pipeline('text-generation', model='gpt2')"
```

> 注意：LLM 推理在 CPU 上较慢，建议使用 GPU 或使用 `device: "cpu"` + `timeout: 10.0`

---

## 7. 常见问题 FAQ

### Q1: 启动报错 `ModuleNotFoundError: No module named 'whisper'`

**原因**：`openai-whisper` 未安装。
```bash
pip install openai-whisper
```

### Q2: 启动报错 `RuntimeError: CUDA not available`

**原因**：config 中设置了 `device: "cuda"` 但没有 NVIDIA GPU。
```yaml
# 修改 config/default.yaml
asr:
  device: "cpu"      # 改为 cpu
llm_corrector:
  device: "cpu"
```

### Q3: 分析很慢，Whisper 模型下载失败

**原因**：网络无法访问 Hugging Face。

**解决**：设置镜像源：
```bash
export HF_ENDPOINT=https://hf-mirror.com
# 或 Windows:
set HF_ENDPOINT=https://hf-mirror.com
```

### Q4: 上传音频后报错 `encoding error 5: Image size exceeds WebP limit`

**原因**：已修复。音频过长导致频谱图超过 WebP 16383 像素限制。
> 当前版本已内置频谱图尺寸自动缩放，不再报此错误。

### Q5: 脱敏后手机号仍能显示

**原因**：ASR 输出使用了未收录的方言数字读音。

当前支持的中文数字：幺、两、二、三、四、五、六、七、八、九、零、一

如果 ASR 将"1"读作"幺"，系统已覆盖。如有个别数字无法识别，可通过添加正则到 `core/privacy_guard.py` 的 `_PII_PATTERNS` 字典中。

### Q6: LLM 纠错没有效果

**原因**：未启用 LLM 模式，或 transformers 未安装。

```bash
pip install transformers torch
# 然后修改 config/default.yaml:
llm_corrector:
  enabled: true
  device: "cpu"      # 或 "cuda"（GPU）
```

### Q7: 如何批量处理多个音频？

当前版本为单文件 UI 处理。如需批量处理，可使用 Python 脚本：

```python
import yaml
from core.asr_engine import ASREngine
from core.privacy_guard import PrivacyGuard
from core.enhancer import AudioEnhancer
from core.llm_corrector import LLMCorrector
from core.fusion_engine import FusionEngine

config = yaml.safe_load(open("config/default.yaml"))
asr   = ASREngine(config)
guard = PrivacyGuard(config)
enh   = AudioEnhancer(config)
llm   = LLMCorrector(config)
fuse  = FusionEngine(config)

files = ["audio1.wav", "audio2.wav", "audio3.wav"]
for path in files:
    segs  = asr.transcribe(path)
    rdocs = guard.analyze(" ".join(s.text for s in segs))
    evts  = enh.detect_anomalies(*enh.load_waveform(path))
    corr  = llm.fix_low_confidence(segs)
    rep   = fuse.assemble(segs, rdocs, evts, corr)
    print(f"{path}: {rep.summary}")
```

### Q8: 测试失败 `pytest: command not found`

```bash
pip install pytest pytest-cov
python3 -m pytest tests/ -v
```

### Q9: 如何查看测试覆盖率？

```bash
python3 -m pytest tests/ -v --cov=core --cov=ui --cov-report=term-missing
```

### Q10: 音频文件太大超过 300 秒限制

在 `utils/audio_utils.py` 中修改 `MAX_DURATION`：

```python
MAX_DURATION = 600  # 改为 10 分钟
```
