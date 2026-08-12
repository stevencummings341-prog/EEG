---
title: "WBCIC-SHU foundation cross-subject (3C)"
tags:
  - "#modality/eeg"
created: "2026-08-07"
updated: "2026-08-12"
status: "active"
---

# WBCIC-SHU foundation cross-subject

端到端 5 模型 × 跨被试。当前正式 run = **三分类 11 人、对齐 DSGNet/SHUv5 协议**。

| 文件 / 目录 | 说明 |
|:---|:---|
| `DSGNET_SHUv5_3C_ANCHOR.md` | 论文协议与 Table II 对标数字（Acc 0.6856） |
| `HANDOFF_821_RUN.md` | **`paper_baseline_3c_821_v1` 交接文档**：跑什么、当前 50/77 进度、换账号要拷哪些产物、如何续跑、坑 |
| 重结果 | `outputs/experiments/wbci_shu/foundation_3c_loso_paper_v1/` |
| 旧结果（不对齐论文） | `outputs/experiments/wbci_shu/foundation_3c_loso_v1/` |
| 8:2:1 统一对比 run（进行中） | `outputs/experiments/wbci_shu/paper_baseline_3c_821_v1/` |

Config：`code/configs/experiments/foundation_cross_subject_wbci_3c.yaml`。
