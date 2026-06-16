# 跨模态分析案例候选

本文件从 50 条统一记录中筛选适合课程设计 PRE 展示的案例。筛选优先考虑：

- `modal_consistency = partial`；
- OCR 和视觉语义均提供补充信息；
- 正文、NLP、OCR、视觉语义四类模态齐全；
- `multimodal_score = 0.90`；
- 图像侧信息与正文具有可解释的主题关联；
- 尽量避免依赖明显噪声或难以解释的视觉标签。

以下分析属于跨模态辅助分析，不代表最终内容安全判定。

## 候选一：央视曝养生馆围猎老年人

**推荐等级：A（最适合 PPT）**

### 基本信息

- topic：央视曝养生馆围猎老年人
- post_id：5308969155298302
- category：社会
- sentiment：中性
- multimodal_score：0.90
- modal_consistency：partial

### 原始信息摘要

- 微博正文：央视报道北京多家养生馆以低价体验吸引老年人，再通过虚假诊断和所谓排毒项目实施诈骗，警方已刑拘30余名嫌疑人。
- OCR 文本：画面识别出央视新闻标识、养生项目价目表、护理项目价格，以及“北京警方捣毁20余家套路养生馆”等新闻字幕。
- 视觉语义：媒体被识别为新闻报道和视频画面截图，语义标签集中在社会民生、财经消费、诈骗及违法犯罪等场景。

### 跨模态分析结果

- ocr_adds_information：true
- vision_adds_information：true
- text_ocr_overlap_score：0.21
- text_vision_overlap_score：0.10
- extra_information_sources：ocr、vision
- analysis_notes：
  - OCR提供了正文之外的图片文字信息
  - 视觉语义提供了场景、人物或类别信息
  - 图像侧信息与正文部分相关，同时包含补充信息

### 适合 PPT 展示的原因

正文说明诈骗事件经过，OCR进一步补充了节目字幕、价目表和警方行动等画面
证据，视觉语义则提供“新闻报道、社会民生、诈骗”的场景判断。三类信息
关系清晰，能够直观说明多模态融合不只是重复正文，而是在补充事件细节。

## 候选二：影石对大疆发起反诉

**推荐等级：A（适合 PPT）**

### 基本信息

- topic：影石对大疆发起反诉
- post_id：5308985706283276
- category：其他
- sentiment：中性
- multimodal_score：0.90
- modal_consistency：partial

### 原始信息摘要

- 微博正文：大疆在美国起诉影石侵犯多项相机专利，影石随后发起反诉，争议涉及云台控制、防抖算法和全景相机等技术。
- OCR 文本：视频画面识别出“大疆在美国起诉影石抄袭”“影石发起专利反诉”，以及Luna相机、Insta360等产品名称。
- 视觉语义：视频抽帧整体被识别为视频画面截图，语义标签包含企业品牌和争议舆情，并检测到人物手持设备的画面。

### 跨模态分析结果

- ocr_adds_information：true
- vision_adds_information：true
- text_ocr_overlap_score：0.25
- text_vision_overlap_score：0.04
- extra_information_sources：ocr、vision
- analysis_notes：
  - OCR提供了正文之外的图片文字信息
  - 视觉语义提供了场景、人物或类别信息
  - 图像侧信息与正文部分相关，同时包含补充信息

### 适合 PPT 展示的原因

OCR文本较短且关键实体清晰，能够从视频画面中补充诉讼标题、品牌和产品
名称；视觉语义补充企业品牌和视频场景。该案例结构简洁、噪声相对较少，
适合用于讲解正文、画面文字和视觉场景如何统一关联。

## 候选三：白鹿方六连辟谣

**推荐等级：B（可选）**

### 基本信息

- topic：白鹿方六连辟谣
- post_id：5308930985039343
- category：娱乐
- sentiment：中性
- multimodal_score：0.90
- modal_consistency：partial

### 原始信息摘要

- 微博正文：白鹿工作室和律师事务所回应学历、恋情、番位等网络传言，表示相关信息属于诽谤并将依法追究传播者责任。
- OCR 文本：多张社交媒体和律师声明截图补充了白鹿本名、维权程序、被指侵权账号，以及多项被否认传言的具体内容。
- 视觉语义：图片被识别为微博截图、律师或公告类材料，标签集中在争议舆情、谣言辟谣、政策公告及娱乐明星。

### 跨模态分析结果

- ocr_adds_information：true
- vision_adds_information：true
- text_ocr_overlap_score：0.18
- text_vision_overlap_score：0.09
- extra_information_sources：ocr、vision
- analysis_notes：
  - OCR提供了正文之外的图片文字信息
  - 视觉语义提供了场景、人物或类别信息
  - 图像侧信息与正文部分相关，同时包含补充信息

### 适合 PPT 展示的原因

该案例能展示正文、声明截图和辟谣场景标签之间的对应关系，且与内容安全
系统中的谣言治理主题较贴合。其不足是 OCR 来源图片较多、文本较长，因此
更适合作为补充案例，而不是首选主案例。

## 推荐顺序

1. **央视曝养生馆围猎老年人**：事件清晰，OCR和视觉语义补充价值最容易解释。
2. **影石对大疆发起反诉**：OCR较简洁，适合展示企业新闻的多模态关联。
3. **白鹿方六连辟谣**：贴合辟谣主题，但OCR文本较长，建议作为备选。
