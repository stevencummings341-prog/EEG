---
title: "Phase 1/2a/2b Baseline"
tags:
  - "#modality/eeg"
  - "#pipeline/5_dl_model"
created: "2026-06-10"
updated: "2026-06-10"
status: "active"
---

# 2_baseline — Baseline 与 Alignment

分为两个实验：无对齐 baseline 与 有对齐 baseline，各自统一为 report / tables / figures。

## 结构

```text
2_baseline/
├── no_alignment_baseline/   # Phase 1 within/cross + Phase 2a multi-source（无对齐）
│   ├── report/
│   ├── tables/
│   └── figures/
└── alignment_baseline/      # Phase 2b no-learning alignment（有对齐）
    ├── report/
    ├── tables/
    └── figures/
```

| 子目录 | 内容 |
|:---|:---|
| `no_alignment_baseline/report/` | `BASELINE_REPORT.md` + 配置/来源快照。 |
| `no_alignment_baseline/tables/` | within + cross + multi-source 汇总表。 |
| `no_alignment_baseline/figures/` | baseline 与 multi-source 图。 |
| `alignment_baseline/report/` | `ALIGNMENT_BASELINE_REPORT.md`、`RUN_STATUS.md` + 配置/来源。 |
| `alignment_baseline/tables/` | 各方法/模型/方向/漂移等级对齐结果表。 |
| `alignment_baseline/figures/` | 对齐对比图。 |

## 核心结论

- 跨 session drop 约 10pp（EEGNet within 0.807 vs cross 0.711）。
- 多源 ses-01+02 → ses-03 优于最强单源。
- 无学习对齐不足：BN-stats 小幅正收益，无方法超过 +2pp。

原始 per-run / checkpoint 在 `backup/root_archive_2026-06-10/`。已完成结果只读，复跑用新 run_id。
