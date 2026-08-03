---
title: "Phase 1/2a/2b Baseline"
tags:
  - "#modality/eeg"
  - "#pipeline/5_dl_model"
created: "2026-06-10"
updated: "2026-06-11"
status: "active"
---

# 2_baseline — Baseline 与 Alignment

按 **数据集并列**：每个数据集一个子目录，下分「无对齐 baseline」与「有对齐 baseline」，
各自统一为 `report / tables / figures`。

## 结构

```text
2_baseline/
├── wbci_shu/                       # WBCIC-SHU 2025 (58ch, 3 session)
│   ├── no_alignment_baseline/      # Phase 1 within/cross + Phase 2a multi-source
│   │   ├── report/  tables/  figures/
│   └── alignment_baseline/         # Phase 2b no-learning alignment
│       ├── report/  tables/  figures/
└── shu/                            # SHU 2022 (32ch, 5 session)
    ├── no_alignment_baseline/
    │   ├── report/  tables/  figures/
    └── alignment_baseline/
        ├── report/  tables/  figures/
```

每一层目录均有 `README.md`。叶子层（report/tables/figures）说明该层放什么。

## 核心结论（WBCIC-SHU）

- 跨 session drop 约 10pp（EEGNet within 0.807 vs cross 0.711）。
- 多源 ses-01+02 → ses-03 优于最强单源。
- 无学习对齐不足：BN-stats 小幅正收益，无方法超过 +2pp。

SHU 结果待运行（见 `code/configs/experiments/shu_*.yaml`）。
原始 per-run / checkpoint 在 `backup/` 与 `outputs/`。已完成结果只读，复跑用新 run_id。
