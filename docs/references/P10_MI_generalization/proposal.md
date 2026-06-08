---
title: "P10_MI泛化研究：运动想象跨 Session 域泛化"
tags:
  - "#pipeline/4_analysis"
  - "#method/domain_generalization"
  - "#modality/eeg"
  - "#paradigm/motor_imagery"
created: 2026-06-06
status: "in-progress"
---

# P10: 运动想象跨 Session 域泛化研究

## 核心问题

> **跨 session 泛化为什么困难？** 同一被试不同 session 的 EEG 分布存在显著漂移，导致模型在新 session 上性能下降。本研究旨在：(1) 量化跨 session 漂移的具体来源和程度；(2) 建立跨 session 分类基线；(3) 基于漂移诊断结果设计针对性的自适应方法。

---

## 1. 研究背景与动机

### 1.1 数据集：WBCIC-SHU MI

详见 [[2025_WBCIC_MI_Dataset_Scientific_Data|WBCIC-SHU 数据集论文]]。

| 维度 | 内容 |
|:---|:---|
| 被试数 | 51（2C 左右手 MI） |
| Session 数 | 3（不同天采集） |
| 通道数 | 58（去除 ECG/EOG 后） |
| 采样率 | 250 Hz |
| Trial 数 | 200/session（100 左 + 100 右） |
| 预处理 | EOG/ECG ICA 清理完成，148/153 session 合格 |

### 1.2 已知的跨 Session 现象

论文报告的 **学习效应**（EEGNet 10-fold CV）：

| Session | 准确率 |
|:---|:---|
| Session 1 | 81.77% |
| Session 2 | 86.63% |
| Session 3 | 88.90% |

**关键观察**：
- 精度逐 session 提升 **+7.1%**，反映 MI 技能学习效应
- 但这意味着 session 间分布不一致 → 直接跨 session 训练/测试会受漂移影响
- 个体差异大：部分被试 S3 反而下降（S9, S16）

### 1.3 研究目标

1. **量化漂移**：系统测量跨 session 的分布距离、频域变化、空间模式变化、可分性变化
2. **建立基线**：within-session / cross-session / LOSO 三种协议的准确率对比
3. **设计方法**：基于漂移诊断结果，设计针对性的跨 session 自适应框架

---

## 2. 研究路线图

```
Phase 0: Session 漂移诊断（方向 A — 当前阶段）
    ↓ 告诉我们"变了什么"
Phase 1: Baseline 建立（方向 B）
    ↓ 告诉我们"难到什么程度"
Phase 2: 自适应方法设计（方向 C — 基于 A 的发现）
    ↓ 解决"怎么应对漂移"
Phase 3: 论文撰写
```

### Phase 0: Session 漂移诊断

**脚本**：`05_Scripts_Automations/data_validation/session_drift_diagnostic.py`

**指标体系**：

| 类别 | 指标 | 回答的问题 |
|:---|:---|:---|
| 分布距离 | MMD (RBF kernel), CORAL | 两个 session 的整体分布差多远？ |
| 频域漂移 | μ/β 带功率变化, KS 检验 | MI 核心频段是否稳定？ |
| 空间漂移 | CSP 模式余弦相似度, 通道 RMS 比值 | 运动皮层激活模式是否一致？ |
| ERD/ERS | 空间模式相关系数 | MI 事件相关去同步是否可重复？ |
| 可分性 | Fisher 判别比 (类间/类内方差) | MI 类别可分性是否随 session 变化？ |
| 信号质量 | 高幅 trial 比例, RMS 统计 | 信号质量是否一致？ |

**产出**：
- `session_drift_report.csv` — 51 被试 × 3 session pair 的漂移画像
- `figures/` — 8 张可视化（热力图、分布图、散点图、相关矩阵）

### Phase 1: Baseline 建立

**脚本**：`05_Scripts_Automations/model_training/eegnet_cross_session.py`

**三种评估协议**：

| 协议 | 训练集 | 测试集 | 回答的问题 |
|:---|:---|:---|:---|
| Within-session | 同 session 10-fold | 同 session | 上界：无漂移时的最佳性能 |
| Cross-session | Session i | Session j | 漂移导致的性能损失 |
| LOSO | 41 被试 | 10 被试 | 跨被试泛化能力 |

**对比方法**：
- EEGNet（深度学习 baseline）
- CSP + LDA（传统 baseline）

**产出**：
- `results_within_session.csv`, `results_cross_session.csv`, `results_loso.csv`
- 三种协议的准确率对比图

### Phase 2: 自适应方法设计（待 Phase 0/1 结论）

基于漂移诊断结果，可能的方向：

| 如果漂移主要来自... | 对应方法 |
|:---|:---|
| 协方差旋转（空间模式变化） | Euclidean Alignment / CORAL 对齐 |
| 均值漂移（幅值变化） | BatchNorm 自适应 / z-score 重归一化 |
| 频段功率漂移 | 频段自适应 / Filter-bank 对齐 |
| 学习效应（正向漂移） | 在线学习 / Test-Time Adaptation |
| 以上均有 | 多子模块自适应框架（ChatGPT 方案） |

**候选框架**：多子模块深度学习预训练（详见 `ChatGPT-EEG MI 预训练任务.md`）
- Neural Subagent Toolkit（时频/空间/熵/连接/Prototype 子网络）
- Dataset-aware Router（数据集感知路由）
- Confidence-aware Fusion（置信度感知融合）
- Online Adaptation（test-then-update 在线学习）

---

## 3. 实验设计细节

### 3.1 数据划分

```
51 被试 × 3 session = 153 个 session
  - 5 个 failed session 排除
  - 148 个 ok session 可用

Phase 0 (诊断): 全部 148 个 session，51 被试 × C(3,2) = 153 个 session pair
Phase 1 (基线):
  - Within: 每个 session 独立 10-fold CV
  - Cross: 6 种 train-test 组合 × 51 被试
  - LOSO: 51 轮，每轮留 1 个被试
Phase 2 (方法): 同 Phase 1 协议，对比改进效果
```

### 3.2 评估指标

| 指标 | 说明 |
|:---|:---|
| Accuracy | 主要指标 |
| Balanced Accuracy | 处理类别不平衡 |
| F1 Score | 综合 precision/recall |
| 准确率衰减比 | $1 - \text{Acc}_{cross} / \text{Acc}_{within}$ |
| 统计显著性 | McNemar 检验（配对比较） |

### 3.3 可视化

| 图表 | 说明 |
|:---|:---|
| 漂移热力图 | MMD/CORAL 的 subject × session 矩阵 |
| 频域漂移直方图 | μ/β 功率变化分布 |
| CSP 模式一致性 | 余弦相似度分布 |
| Fisher 漂移散点 | 可分性 session 间变化 |
| 准确率对比图 | 三种协议的 bar chart |
| 学习效应曲线 | 准确率随 session 变化 |

---

## 4. 技术栈

| 组件 | 技术选型 |
|:---|:---|
| 数据格式 | numpy (.npz) |
| 频域分析 | scipy.signal, numpy.fft |
| 机器学习 | scikit-learn（CSP, LDA, 评估指标） |
| 深度学习 | PyTorch（EEGNet） |
| 可视化 | matplotlib, seaborn |
| 运行环境 | 远程服务器（/share/workspace2/...） |

---

## 5. 预期产出

### 5.1 学术产出

- [ ] **论文 1**：跨 Session 漂移诊断（分析性论文，量化漂移来源）
- [ ] **论文 2**：跨 Session 自适应方法（基于诊断结果的新方法）

### 5.2 技术产出

- [ ] Session 漂移诊断工具（可复用的分析脚本）
- [ ] EEGNet Cross-Session Baseline（benchmark 代码）
- [ ] 跨 Session 自适应模型

### 5.3 与其他项目的联动

| 项目 | 联动方式 |
|:---|:---|
| [[P2_跨被试泛化研究/proposal\|P2 跨被试泛化研究]] | 跨 session 是跨被试的子问题；方法可迁移 |
| [[P5_在线BCI自适应系统/proposal\|P5 在线 BCI 自适应系统]] | 漂移诊断为在线适应提供触发条件 |
| [[P4_EEG降噪与任务特征保真/proposal\|P4 EEG 降噪]] | 降噪是否减少 session 漂移？ |

---

## 6. 当前进度

| 阶段 | 状态 | 产出 |
|:---|:---|:---|
| 数据集论文解读 | ✅ 完成 | `02_Literature/Motor_BCI_2026/2025_WBCIC_MI_Dataset_Scientific_Data.md` |
| QC 质量报告 | ✅ 完成 | `QC_SUMMARY_CN.md`（148/153 session 合格） |
| 漂移诊断脚本 | ✅ 完成 | `05_Scripts_Automations/data_validation/session_drift_diagnostic.py` |
| Baseline 脚本 | ✅ 完成 | `05_Scripts_Automations/model_training/eegnet_cross_session.py` |
| 服务器运行诊断 | ⏳ 待执行 | 需 scp 到服务器运行 |
| Baseline 训练 | ⏳ 待执行 | 需服务器 GPU |
| 方法设计 | ⏳ 待 Phase 0 结论 | 基于漂移诊断结果 |

---

<!-- Obsidian 格式硬规则
1. 图片嵌入必须使用 ![[filename.png|800]]
2. Wikilink 必须带 display alias
3. 禁止用 --- 作为占位符，用 --
4. YAML 字符串值必须用双引号包围
-->
