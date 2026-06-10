---
title: "Phase 0 Session Drift"
tags:
  - "#modality/eeg"
  - "#pipeline/4_analysis"
created: "2026-06-10"
updated: "2026-06-10"
status: "active"
---

# 1_session_drift — Phase 0 漂移诊断

Phase 0 跨 session 漂移诊断的真实结果，统一分为 report / tables / figures。

## 结构

```text
1_session_drift/
├── report/    # 文字报告
├── tables/    # CSV / JSON 数据表
└── figures/   # 诊断图
```

| 子目录 | 内容 |
|:---|:---|
| `report/` | `SESSION_DRIFT_REPORT.md`、`SESSION_DRIFT_SUMMARY_CN.md`、`per_subject_drift_summary.md`。 |
| `tables/` | `session_drift_report.csv`、`session_pair_summary.csv`、`per_subject_drift_summary.csv`、`summary.json`。 |
| `figures/` | 14 张诊断图。 |

## 核心结论

漂移主要是空间模式 + μ/β 频谱分布，不是幅值。原始 run 在 `backup/root_archive_2026-06-10/outputs/analysis/session_drift_v1/`。复跑用新 run_id，不覆盖。
