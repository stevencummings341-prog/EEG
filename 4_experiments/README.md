---
title: "New Experiments"
tags:
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-06-11"
status: "active"
---

# 4_experiments — Phase 2c+ 新实验

按 **数据集并列**：每个数据集一个子目录，下分各实验（如 prototype_drift），
每个实验统一为 `report / tables / figures`。

## 结构

```text
4_experiments/
├── wbci_shu/
│   └── prototype_drift/
│       ├── report/  tables/  figures/
└── shu/
    └── prototype_drift/
        ├── report/  tables/  figures/
```

每一层目录均有 `README.md`。

## 约定

- 每个实验一个子目录，包含 README、输出清单、核心结论；未运行不要写 done。
- 已完成结果只读；复跑用新 `run_id`。
- 命名规范见 `AGENTS.md` / `CLAUDE.md` 第 9 节。
