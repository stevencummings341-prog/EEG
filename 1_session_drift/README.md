---
title: "Phase 0 Session Drift"
tags:
  - "#modality/eeg"
  - "#pipeline/4_analysis"
created: "2026-06-10"
updated: "2026-06-11"
status: "active"
---

# 1_session_drift — Phase 0 漂移诊断

按 **数据集并列**：每个数据集一个子目录，统一分为 `report / tables / figures`。

## 结构

```text
1_session_drift/
├── wbci_shu/        # WBCIC-SHU 2025
│   ├── report/  tables/  figures/
└── shu/             # SHU 2022 (待运行)
    ├── report/  tables/  figures/
```

每一层目录均有 `README.md`。

| 子目录 | 内容 |
|:---|:---|
| `<dataset>/report/` | `SESSION_DRIFT_REPORT.md` 等文字报告。 |
| `<dataset>/tables/` | `session_drift_report.csv`、`per_subject_drift_summary.csv`、`summary.json`。 |
| `<dataset>/figures/` | 诊断图。 |

## 核心结论（WBCIC-SHU）

漂移主要是空间模式 + μ/β 频谱分布，不是幅值。复跑用新 run_id，不覆盖。
SHU 结果待运行。
