---
title: "P10 项目交接文档"
tags:
  - "#pipeline/4_analysis"
  - "#method/domain_generalization"
  - "#modality/eeg"
created: 2026-06-06
---

# P10 MI 泛化研究 — 项目交接文档

> 本文档用于将项目上下文完整传递给新的 AI 智能体。包含项目背景、已完成工作、待执行任务和所有相关文件索引。

---

## 一、项目概述

### 核心研究问题

**跨 session 运动想象 EEG 泛化为什么困难？如何量化和缓解？**

同一被试不同 session 的 EEG 分布存在显著漂移（论文报告学习效应：Session 1→3 精度从 81.8% 提升到 88.9%，+7.1%），这意味着 session 间分布不一致，直接跨 session 训练/测试会受漂移影响。

### 研究路线图

```
Phase 0: Session 漂移诊断（方向 A — 当前阶段）
    ↓ 告诉我们"变了什么"
Phase 1: Baseline 建立（方向 B）
    ↓ 告诉我们"难到什么程度"
Phase 2: 自适应方法设计（方向 C — 基于 A 的发现）
    ↓ 解决"怎么应对漂移"
Phase 3: 论文撰写
```

- **方向 A + B 可并行**，方向 C 依赖 A 的结论
- 最终目标是设计一个**跨 session 自适应框架**，可能基于多子模块深度学习预训练方案

---

## 二、数据集信息

### WBCIC-SHU MI 数据集

| 维度 | 内容 |
|:---|:---|
| 被试数 | 51（2C 左右手 MI） |
| Session 数 | 3（不同天采集） |
| 通道数 | 58（去除 ECG/EOG 后） |
| 采样率 | 250 Hz |
| Trial 数 | 200/session（100 左 + 100 右） |
| 数据维度 | X: [200, 58, 1000], y: [200] |
| 数据位置 | 远程服务器 `/share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean/` |
| QC 状态 | 148/153 session 合格（5 个因触发缺失排除） |

### 已排除的 5 个 failed session

| session | 原因 |
|:---|:---|
| sub-023/ses-01 | 少 1 个 trial（199/200） |
| sub-024/ses-02 | 少 1 个 trial |
| sub-024/ses-03 | 少 5 个 trial（195/200） |
| sub-026/ses-01 | 少 1 个 trial |
| sub-032/ses-02 | 少 1 个 trial |

### 论文报告的 baseline（EEGNet 10-fold CV）

| Session | 准确率 |
|:---|:---|
| Session 1 | 81.77% |
| Session 2 | 86.63% |
| Session 3 | 88.90% |

---

## 三、已完成工作

### 3.1 文献与数据

| 产出 | 文件路径 | 说明 |
|:---|:---|:---|
| 数据集论文解读 | `02_Literature/Motor_BCI_2026/2025_WBCIC_MI_Dataset_Scientific_Data.md` | 完整的 SOP 5 解读笔记 |
| 论文提取文本 | `07_Assets/s41597-025-04826-y.txt` | PDF 全文文本 |
| QC 质量报告 | `04_Research_Projects/P10_MI泛化研究/QC_SUMMARY_CN.md` | 同学的预处理质量总结 |
| ChatGPT 方案 | `04_Research_Projects/P10_MI泛化研究/ChatGPT-EEG MI 预训练任务.md` | 多子模块预训练框架设计 |

### 3.2 项目文档

| 产出 | 文件路径 | 说明 |
|:---|:---|:---|
| 项目提案 | `04_Research_Projects/P10_MI泛化研究/proposal.md` | 已更新为跨 session DG 方向 |
| 实验日志 | `04_Research_Projects/P10_MI泛化研究/experiment_log.md` | 待填写实验结果 |
| 本交接文档 | `04_Research_Projects/P10_MI泛化研究/HANDOFF.md` | 项目上下文打包 |

### 3.3 脚本代码

| 产出 | 文件路径 | 说明 |
|:---|:---|:---|
| Session 漂移诊断脚本 | `05_Scripts_Automations/data_validation/session_drift_diagnostic.py` | 方向 A，计算 8 类漂移指标 |
| 诊断脚本说明 | `05_Scripts_Automations/data_validation/session_drift_diagnostic.md` | 使用文档 |
| EEGNet baseline 脚本 | `05_Scripts_Automations/model_training/eegnet_cross_session.py` | 方向 B，三种评估协议 |
| Baseline 脚本说明 | `05_Scripts_Automations/model_training/eegnet_cross_session.md` | 使用文档 |
| 脚本目录索引 | `05_Scripts_Automations/README.md` | 脚本总览 |

---

## 四、待执行任务（按优先级）

### 优先级 1：运行 Session 漂移诊断（方向 A）

```bash
# 在远程服务器上运行
python session_drift_diagnostic.py \
    --data_dir /share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean/ \
    --output_dir ./drift_report
```

**预期产出**：
- `session_drift_report.csv` — 153 个 session pair 的漂移画像
- `figures/` — 8 张可视化（热力图、分布图、散点图）

**分析要点**：
- MMD/CORAL 分布 → 整体漂移程度
- μ/β 功率漂移 → 频域是否稳定
- CSP 模式相似度 → 空间模式是否一致
- Fisher 判别比变化 → MI 可分性是否随 session 变化
- 指标相关性矩阵 → 哪些指标共同变化

### 优先级 2：运行 EEGNet Baseline（方向 B）

```bash
# 先跑 within 验证环境
python eegnet_cross_session.py --data_dir /path/to/data --protocol within --device cuda

# 再跑全部
python eegnet_cross_session.py --data_dir /path/to/data --protocol all --device cuda
```

**预期产出**：
- `results_within_session.csv` — within-session 准确率（应接近论文的 85.3%）
- `results_cross_session.csv` — cross-session 准确率（预期低于 within）
- `results_loso.csv` — LOSO 跨被试准确率（预期显著低于 within）

### 优先级 3：分析结果并设计方法（方向 C）

基于方向 A 的漂移诊断结果，确定漂移主要来源：

| 如果漂移主要来自... | 对应方法 |
|:---|:---|
| 协方差旋转（空间模式变化） | Euclidean Alignment / CORAL 对齐 |
| 均值漂移（幅值变化） | BatchNorm 自适应 / z-score 重归一化 |
| 频段功率漂移 | 频段自适应 / Filter-bank 对齐 |
| 学习效应（正向漂移） | 在线学习 / Test-Time Adaptation |
| 以上均有 | 多子模块自适应框架 |

---

## 五、技术方案备忘

### 5.1 Session 漂移诊断指标体系

| 类别 | 指标 | 回答的问题 |
|:---|:---|:---|
| 分布距离 | MMD (RBF kernel), CORAL | 两个 session 的整体分布差多远？ |
| 频域漂移 | μ/β 带功率变化, KS 检验 | MI 核心频段是否稳定？ |
| 空间漂移 | CSP 模式余弦相似度, 通道 RMS 比值 | 运动皮层激活模式是否一致？ |
| ERD/ERS | 空间模式相关系数 | MI 事件相关去同步是否可重复？ |
| 可分性 | Fisher 判别比 | MI 类别可分性是否随 session 变化？ |
| 信号质量 | 高幅 trial 比例, RMS 统计 | 信号质量是否一致？ |

### 5.2 EEGNet 评估协议

| 协议 | 训练集 | 测试集 | 回答的问题 |
|:---|:---|:---|:---|
| Within-session | 同 session 10-fold | 同 session | 上界：无漂移时的最佳性能 |
| Cross-session | Session i | Session j | 漂移导致的性能损失 |
| LOSO | 50 被试 | 1 被试 | 跨被试泛化能力 |

### 5.3 候选自适应框架（ChatGPT 方案摘要）

多子模块深度学习预训练框架：
1. **Neural Subagent Toolkit**：时频/空间/熵/连接/Prototype 子网络
2. **Dataset-aware Router**：数据集感知路由，自动选择子模块权重
3. **Confidence-aware Fusion**：置信度感知融合，动态调整贡献
4. **Online Adaptation**：test-then-update 在线学习（只更新 adapter/prototype/BN）

详见 `04_Research_Projects/P10_MI泛化研究/ChatGPT-EEG MI 预训练任务.md`

---

## 六、知识库相关文件

| 文件 | 与 P10 的关系 |
|:---|:---|
| `03_Knowledge_Base/1_01_EEG时变性与跨被试差异的数学建模.md` | MMD 分解框架、session 漂移 SDE 建模 |
| `03_Knowledge_Base/MOC_域泛化前沿汇总.md` | DG 方法索引、性能 ceiling 表 |
| `03_Knowledge_Base/2_02_在线BCI系统_预训练到自适应更新完整架构.md` | 在线适应三层架构 |
| `02_Literature/Motor_BCI_2026/2025_WBCIC_MI_Dataset_Scientific_Data.md` | 数据集论文解读 |
| `04_Research_Projects/P10_MI泛化研究/ChatGPT-EEG MI 预训练任务.md` | 多子模块方案设计 |

---

## 七、文件完整性检查清单

以下是 P10 项目应包含的所有文件：

- [x] `proposal.md` — 项目提案（已更新）
- [x] `experiment_log.md` — 实验日志（已初始化）
- [x] `QC_SUMMARY_CN.md` — 数据质量报告（同学提供）
- [x] `ChatGPT-EEG MI 预训练任务.md` — 方案设计（ChatGPT 对话）
- [x] `WBCIC_SHU_MI_EEG_dataset_论文汇报.pptx` — 数据集汇报 PPT
- [x] `s41597-025-04826-y(1).pdf` — 数据集论文 PDF
- [x] `HANDOFF.md` — 本交接文档
- [ ] `results.md` — 实验结果（待创建）
- [ ] `figures/` — 可视化目录（待运行脚本后生成）

脚本文件：
- [x] `05_Scripts_Automations/data_validation/session_drift_diagnostic.py`
- [x] `05_Scripts_Automations/data_validation/session_drift_diagnostic.md`
- [x] `05_Scripts_Automations/model_training/eegnet_cross_session.py`
- [x] `05_Scripts_Automations/model_training/eegnet_cross_session.md`
- [x] `05_Scripts_Automations/README.md`

---

<!-- Obsidian 格式硬规则
1. 图片嵌入必须使用 ![[filename.png|800]]
2. Wikilink 必须带 display alias
3. 禁止用 --- 作为占位符，用 --
4. YAML 字符串值必须用双引号包围
-->
