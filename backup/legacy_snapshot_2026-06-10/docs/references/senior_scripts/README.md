---
title: "脚本与自动化目录索引"
tags:
  - "#scripts/index"
created: 2026-06-06
---

# 05_Scripts_Automations — 脚本目录索引

## 目录结构

```
05_Scripts_Automations/
├── data_validation/           # 数据规范检查脚本
├── preprocessing/             # 预处理管道
├── feature_extraction/        # 特征提取
├── model_training/            # 模型训练入口脚本
├── paper_utils/               # 论文工具
├── obsidian_utils/            # 仓库维护脚本
└── README.md                  # 本文件
```

---

## 脚本清单

### data_validation/

| 脚本 | 说明 | 项目 |
|:---|:---|:---|
| `session_drift_diagnostic.py` | Session 漂移诊断：MMD、CORAL、频域/空间/可分性指标 | [[P10_MI泛化研究/proposal\|P10 MI 泛化]] |

### model_training/

| 脚本 | 说明 | 项目 |
|:---|:---|:---|
| `eegnet_cross_session.py` | EEGNet cross-session baseline（within/cross/loso 三种协议） | [[P10_MI泛化研究/proposal\|P10 MI 泛化]] |

---

## 使用说明

每个脚本目录下有对应的 `.md` 说明文档，包含：
- 功能概述
- 使用方式和参数说明
- 数据格式要求
- 输出文件说明
- 依赖列表
- 服务器运行指南
