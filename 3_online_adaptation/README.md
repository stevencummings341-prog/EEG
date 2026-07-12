---
title: "Online Adaptation"
tags:
  - "#modality/eeg"
  - "#method/domain_generalization"
created: "2026-06-10"
updated: "2026-07-12"
status: "active"
---

# 3_online_adaptation — 在线适应框架

按 **数据集并列**：每个数据集一个子目录。

```text
3_online_adaptation/
├── PRETRAINED_MODEL_INTEGRATION_CONTRACT.md  # 学长/预训练模型接入契约（权威）
├── PHASE3_TTA_DESIGN.md
├── wbci_shu/    # WBCIC-SHU 在线 test-then-update 设计/结果
└── shu/         # SHU 对应内容（待运行）
```

每个数据集子目录均有 `README.md`。系统架构文档为只读参考。

## Pretrained model integration

**Authoritative contract:** [`PRETRAINED_MODEL_INTEGRATION_CONTRACT.md`](PRETRAINED_MODEL_INTEGRATION_CONTRACT.md)

Round-1 scaffold complete; **real pretrained model not yet integrated**. Mock / baseline
adapters under `code/tta/` are fixtures only.
