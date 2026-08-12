---
title: "WBCIC-SHU foundation cross-subject (3C)"
tags:
  - "#modality/eeg"
created: "2026-08-07"
updated: "2026-08-07"
status: "active"
---

# WBCIC-SHU foundation cross-subject

端到端 5 模型 × 跨被试。当前正式 run = **三分类 11 人、对齐 DSGNet/SHUv5 协议**。

| 文件 / 目录 | 说明 |
|:---|:---|
| `DSGNET_SHUv5_3C_ANCHOR.md` | 论文协议与 Table II 对标数字（Acc 0.6856） |
| 重结果 | `outputs/experiments/wbci_shu/foundation_3c_loso_paper_v1/` |
| 旧结果（不对齐论文） | `outputs/experiments/wbci_shu/foundation_3c_loso_v1/` |

Config：`code/configs/experiments/foundation_cross_subject_wbci_3c.yaml`。
