# 多模块整合数据构建报告

## 构建结果

- 爬虫主表记录数：50
- 生成父记录数：50
- 重复 `parent_document_id`：0
- `merged_text` 非空记录数：50
- 含视觉摘要的父记录数：50

## OCR 与视觉语义

- 媒体结果总数：151
- 图片记录数：121
- 视频记录数：30
- 对齐成功：151
- 对齐失败：0
- 非空 OCR 文本：132
- 非空视觉摘要：151
- 整合后的 OCR 文本数：132
- 整合后的视觉摘要数：151

## NLP 与 ASR

- NLP 输入记录数：50
- NLP 对齐成功：50
- NLP 对齐失败：0
- ASR 当前缺失，但所有记录均预留 `asr_texts: []` 和 `asr_count: 0`。
- ASR 缺失不会中断整合流程。

## 多模态融合字段

- 每条记录均新增 `available_modalities`、`missing_modalities`、
  `multimodal_score`、`fused_text`、`safety_indicators` 和
  `cross_modal_analysis`。
- 互动量 75 分位阈值：0.0
- PRE 阶段建议关注记录数：7
- `needs_review` 仅表示候选关注，不代表最终内容安全判定。
- 跨模态一致性分布：{"consistent": 7, "partial": 29, "weak": 14, "unknown": 0}

## 校验结论

- 父键是否唯一：是
- `merged_text` 是否全部非空：是
- 图像处理结果是否全部对齐：是
- NLP 结果是否全部对齐：是
- 视觉摘要是否进入统一数据集：是

## 无法对齐的结果

### 图像处理

```json
[]
```

### NLP

```json
[]
```
