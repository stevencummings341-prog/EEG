---
title: "P10_MI泛化研究 - 实验日志"
tags:
  - "#experiment/log"
  - "#paradigm/motor_imagery"
created: 2026-06-06
---

# P10: MI 泛化研究 — 实验日志

## 实验记录格式

每次实验记录包含：
- **日期**：实验执行日期
- **目标**：本次实验要验证的假设或要解决的问题
- **方法**：使用的算法、参数、数据集
- **结果**：定量指标 + 定性观察
- **结论**：结果说明了什么，下一步做什么

---

## 实验日志

### 2026-06-06: 项目初始化与脚本编写

**目标**：建立项目框架，编写诊断和基线脚本

**完成**：
- [x] 数据集论文解读归档（`2025_WBCIC_MI_Dataset_Scientific_Data.md`）
- [x] QC 质量报告确认（148/153 session 合格）
- [x] 编写 session 漂移诊断脚本（`session_drift_diagnostic.py`）
- [x] 编写 EEGNet cross-session baseline 脚本（`eegnet_cross_session.py`）
- [x] 更新 proposal.md（调整为跨 session DG 方向）

**待完成**：
- [ ] 将脚本 scp 到远程服务器
- [ ] 运行 session 漂移诊断
- [ ] 运行 EEGNet baseline（within/cross/loso）
- [ ] 分析诊断结果，确定漂移主要来源
- [ ] 基于漂移诊断设计自适应方法

---

## 基线结果记录表

| 方法 | 协议 | 数据集 | 准确率 | 备注 |
|:---|:---|:---|:---|:---|
| EEGNet | Within-session CV | WBCIC 2C | -- | 待测试 |
| EEGNet | Cross-session | WBCIC 2C | -- | 待测试 |
| EEGNet | LOSO | WBCIC 2C | -- | 待测试 |
| CSP+LDA | Within-session CV | WBCIC 2C | -- | 待测试 |

## Session 漂移诊断结果

（待运行后填写）

---

## 关键发现

（随实验进展更新）

---

<!-- Obsidian 格式硬规则
1. 图片嵌入必须使用 ![[filename.png|800]]
2. Wikilink 必须带 display alias
3. 禁止用 --- 作为占位符，用 --
4. YAML 字符串值必须用双引号包围
-->
