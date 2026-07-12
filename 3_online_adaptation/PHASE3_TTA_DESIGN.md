---
title: "Phase 3 — Test-Time Adaptation (T3A) Design & Route"
tags:
  - "#modality/eeg"
  - "#method/test_time_adaptation"
  - "#method/prototype"
created: "2026-07-06"
updated: "2026-07-06"
status: "planned"
---

# Phase 3 — Prototype-based Test-Time Adaptation (T3A) for Cross-session MI

> 这是 Phase 3 的**权威路线文件（指示文件）**。后续 Agent/人类照本文件执行。
> 状态：**planned / not run**（config 骨架已就绪，runner 待实现，见 §7 Step A）。
> 数字区暂空——未跑不得填 done、不得引用/编造结果。

## 0. Core conclusion (why this is the right next step)

- 参考论文：**Lee, Pradeepkumar, Sun. “Test-Time Adaptation for EEG Foundation Models:
  A Systematic Study under Real-World Distribution Shifts.” arXiv:2604.16926v1 (2026)**
  （NeuroAdapt-Bench）。其系统对比 Tent / SHOT / T3A 三种 TTA，核心结论：
  **只有 T3A（无优化、基于原型）在 in-distribution / OOD / 极端漂移三种设定下平均
  平衡准确率为正增益且最稳定；梯度类（Tent/SHOT）经常大幅掉点（负迁移）。**
  T3A 对 batch size 不敏感，在类不平衡时收益最大（REVE-Base 在 CHB-MIT +18.9pp）。
- 与本项目高度契合：Phase 2c 已证明跨 session 掉点机制 = **within-class scatter 膨胀 /
  Fisher collapse（非 centroid collapse）**，class prototype 确实漂移，且 **cosine 几何优于
  euclidean**，下游 prototype 方法应在归一化空间做。**T3A 正是"用 target 特征重估 class
  prototype、直接调整分类器几何、无需 target 标签"的可部署方法**——它是 **Phase B cosine
  Oracle 上限诊断**的**无标签、可落地对应物**。
- 因此 Phase 3 统一为「跨 session 修复」，含两条互补线（**v2：Oracle 先裁决**）：
  1. **Oracle 上限（label-informed 诊断，Phase B 裁决门，先做）**：已知 target 真标签时用 target
     prototype 能回收多少 acc（只做离线诊断，不训练/不部署）——裁决要不要扩大 T3A。
  2. **T3A（label-free 方法，本文主线）+ Tent/SHOT（论文对比复现）**：无标签、用伪标签 +
     熵筛选重估原型，实际提升跨 session 泛化。

## 1. Goal

在**不使用 target 标签、不改动主干网络**的前提下，用 T3A 在跨 session（同被试、同任务、
同蒙太奇）场景恢复 Phase 1/2c 观测到的掉点，并回答：

- **主问题**：T3A 相对 No-TTA 基线能否带来稳定正增益（预注册成功线：均值 Δbalanced-acc
  > +2pp，且在多数 (subject,direction) cell 上净正）？
- **机制验证**：Phase 2c 说 cosine>euclidean、scatter 膨胀——那么 **cosine 空间 T3A 是否
  优于论文原版 dot-product T3A**？T3A 是否在 prototype 可回收（centroid-recoverable）的 cell
  上收益最大、而在 FBCNet（几何异常）上失效？
- **方法学复现**：在 MI 数据上，论文的"T3A 稳、Tent/SHOT 不稳"结论是否成立？
- **组合优选（用户诉求）**：不同预训练 backbone（EEGNet/DeepConvNet/FBCNet）× 筛选/几何/
  filter_k 组合中，哪个组合最优，且相对 baseline 有稳健提升。

## 2. Method — T3A as we implement it

源模型 `f = h_w ∘ g_φ`（Phase 2c 训好的 source-only 冻结模型）。target session 的 trial
按 batch 到达（模拟在线流；episodic=False，support 跨整段累积）：

```text
初始化：每类 support set S_k ← 源分类头权重向量 ω_k = model.classifier.weight[k]
        （论文原版初始化；变体：用 source-train 类原型 mean(z|y=k) 初始化）
对每个 target batch：
  1. z = g_φ(x)（冻结前向）；logits/probs/entropy 由当前"调整后原型"给出
  2. 伪标签 ŷ = argmax_k  s(z, c_k)          # s = dot 或 cosine 相似度
  3. 把 (z, ŷ) 加入 S_ŷ
  4. 【不确定性筛选 filter_k】每类只保留熵最低（置信最高）的 top-k 个 support 向量
  5. 重估原型 c_k = mean(S_k)
  6. 预测 p(y=k|z) ∝ exp( s(z, c_k) / τ )    # dot: 论文原版；cosine: 本项目改进
```

**关键：本项目复用已存嵌入，T3A 全程可离线 CPU 跑。** Phase 2c 已把每个
(subject, direction, model, seed) 的 `target_test__{z,logits,probs,pred,conf,y}` 存进 npz
（`outputs/experiments/{wbci_shu,shu}/prototype_drift_v1/embeddings/`）。T3A 无优化、只在
特征空间算原型，因此**不需重训、不需 GPU**：直接按 trial 顺序 replay target 嵌入成 mini-batch
即可。初始权重 ω_k 从 Phase 2c 保存的 checkpoint（`checkpoints/{ds}/prototype_drift_v1/`）读，
或用 source-train 原型初始化（npz 里已有）。这让 1 个月的全量 ablation 极其轻量。

## 3. Protocol & fairness

- **No-leakage 铁律（沿用 Phase 2c）**：模型只在 source session 训练；target 标签**只**用于
  最终评测，绝不进训练/验证/早停/调参/适应。T3A 用**伪标签**（模型预测），不碰真标签。
  记录 `used_target_labels_for_adaptation ≡ False`。
- **No-TTA 基线** = Phase 2c 里同一冻结模型的 `acc_target`（同 seed/同 cell 对齐）。
- **Δ_TTA** = metric_TTA − metric_No-TTA，**逐 (subject,direction,seed) 先算差再聚合**（论文口径），
  这样波动反映的是适应效果而非绝对精度。
- **稳定性轴（论文核心）**：除均值 Δ 外，报告"净正 cell 比例""最差 cell 掉点"——TTA 的价值
  在稳定，不只在均值。
- Split：按 subject/session，绝不按 trial 泄漏；target 全 session 作评测流。

## 4. Experiment matrix (1 month)

| 轴 | 取值 | 说明 |
|:---|:---|:---|
| 数据集 | wbci_shu(58ch)、shu(32ch) | 均复用 Phase 2c 已存嵌入/权重 |
| backbone（"预训练模型"） | eegnet、deepconvnet、fbcnet | 用户诉求的"不同预训练模型" |
| 方法 | **No-TTA(基线)**、**T3A(主)**、Tent、SHOT | Tent/SHOT 需活模型→GPU；T3A/No-TTA→CPU 离线 |
| 几何 s(·) | **cosine**、dot | 验证 Phase 2c 的 cosine>euclidean 假设 |
| filter_k | 5, 10, 20(论文默认), 50, ∞ | support 上限 |
| 原型初始化 | clf_weights(论文)、src_proto | |
| 更新门控 | pseudo-label(全加)、entropy-gated(H<τ 才加) | |
| 不确定性度量 | entropy、margin、soft(置信加权原型) | ⚠️见 §6 二分类退化说明 |
| seeds | 0–4（5 seeds） | 对齐项目 + 论文 |
| batch size | 64、128、256 | 论文口径；MI session 小，256≈整段 |
| 评测指标 | balanced_acc(主)、accuracy、macro_f1、auc | Δ_TTA 逐 cell 先算 |

## 5. Baselines & positioning

1. **主对比**：T3A vs No-TTA（Phase 1/2c 冻结跨 session 基线）→ Δ_TTA。
2. **项目内对比**：vs Phase 2b 无学习对齐最优（WBCIC=BN-stats / SHU=z-score，均 <+2pp）——
   看 T3A 能否越过统计对齐做不到的 +2pp 线。
3. **上限对比**：vs Phase 3 Oracle（label-informed 上限）——label-free T3A 回收了多少天花板？
4. **复现论文**：T3A 稳/正、Tent/SHOT 不稳/负——在 MI 上是否重现。

## 6. Honest caveats（必须在报告写明）

- **本项目 backbone ≠ 论文的 EEG 基础模型**。论文用 CBraMod/REVE/TFM-Tokenizer（冻结预训练
  编码器 + 共享线性头）跑**跨数据集/跨任务/跨模态**漂移；我们用**从零训练的小模型**跑
  **跨 session**（更温和但真实）漂移。T3A 算法与架构无关（只需特征提取器 + 线性头），可直接迁移，
  但**不得声称基础模型结果**。集成真实 EEG 基础模型（CBraMod 等）= 可选延伸（§8），非主线。
- **二分类熵≡最大置信度退化**：K=2 时预测熵是最大 softmax 概率的严格单调函数，故"熵筛选"与
  "最大置信度筛选"**排序完全等价**。因此在本 MI 任务上，真正有区分度的"不确定性/筛选"轴是：
  **filter_k、几何(cosine vs dot)、软硬(soft 加权 vs hard top-k)、margin**——报告须点明，
  不能把 entropy vs max-conf 当作两个不同设置来邀功。
- SHU 跨 session 近 chance（地板效应），T3A 伪标签本身噪声大，预期收益有限——如实报告。
- FBCNet 在两数据集几何都异常（Phase 2c），T3A 很可能失效，须单独讨论、不并入主结论。

## 7. Implementation route（照此执行）—— v2：Oracle 先裁决

> **重排原则（2026-07-08）**：Phase 2c 权威结论是 prototype drift 只解释部分掉点、主机制是 scatter
> 膨胀 / Fisher collapse。因此 **Oracle 上限实验必须提前到 T3A 大扫之前作为科研裁决门**：先看"已知
> 目标原型 / scatter-aware 几何"的理论天花板，再决定要不要大规模上 T3A。详见根目录 `PHASE3_ROUTE_PLAN.md`。
> **T3A 现阶段只作无梯度 prototype baseline，不得提前写成"最终修复方案"。**

- **Phase 0 — 修正项目状态（文档）**：AGENTS/STATUS/README/results/progress 同步为"Phase 2c done →
  Phase 3A replay+smoke → Phase 3B Oracle 裁决"；修复 embed_index 失效路径 bug（消费端重拼路径）。
- **Phase A0 — Embedding Replay + No-TTA 精确复现（CPU，第一关）**
  - `code/experiments/session_tta.py`：读 Phase 2c `embed_index`/`prototype_drift_metrics` + npz →
    逐 cell 组织 target 流 → 算 No-TTA。`code/runners.py` 注册 `phase3_tta`。
  - **硬验收：No-TTA 复算 acc_target 逐 cell 对齐 Phase 2c，|Δ|<1e-6。不过则后续全不可信。**
- **Phase A1 — Minimal T3A Smoke（CPU，只证 pipeline）**
  - `code/methods/t3a.py`：source-prototype 初始化 + cosine + filter_k=[20]，纯 numpy 无梯度。
  - 范围：WBCIC + EEGNet + 2 被试（1 stable / 1 high-drift）+ seed 0 + `no_tta`+`t3a_src_proto_cosine`。
  - 报告只写"pipeline 可跑"，不写"T3A 有效"。
- **Phase B — ⭐ Oracle 上限（提前！裁决门，CPU）⭐**
  - 方法组：`source_proto` / `target_label_oracle_proto` / `cosine_oracle_proto` /
    `mahalanobis_oracle` / `shrinkage_oracle` / `reliability_weighted_oracle`（复用 Phase 2c 嵌入 + target
    真标签做离线上限诊断，显式标 `used_target_labels=True (oracle only)`）。
  - 指标：balanced acc、negative transfer rate、within-class scatter、Fisher、negative-margin、
    class-collapse、按 drift tertile 分层。
  - **决策门槛**：Oracle > +3pp → 进 Phase C；< +1pp → 停大规模 T3A，转 scatter/reliability/decision-boundary。
- **Phase C — T3A 小规模 ablation（仅当 Phase B 支持）**：init×geometry×filter_k；分层报告，不加 safety。
- **Phase D — Safe / reliability-weighted T3A**：仅当原始 T3A 有收益但负迁移明显时加安全门（EEG 特色）。
- **Phase E — 全量离线 sweep**：WBCIC 全量为主，SHU 外部验证，FBCNet 单列。
- **Phase F — Tent/SHOT/CoTTA 对照（GPU，后置）**：梯度类对照复现。
- **Phase G — 论文叙事收敛**：按 Oracle/T3A 结果三选一（见 `PHASE3_ROUTE_PLAN.md` §3 Phase G）。

### 时间线（1 个月，指示性）
- W1：Phase 0 + A0 + A1（文档修正 + replay 精确复现 + minimal T3A smoke，纯 CPU）。
- W2：**Phase B Oracle 裁决**（这是本月最关键节点，决定后续是否值得扩大 T3A）。
- W3：按裁决走 Phase C（+ 必要时 D）或转向 scatter/reliability 机制。
- W4：Phase E 全量（若通过）+ Phase G 叙事收敛 + 报告/图 + 同步 handoff。（Phase F 视需要顺延）

## 8. Out of scope / future

- 集成真实 EEG 基础模型（CBraMod/REVE/TFM-Tokenizer）跑 T3A（大工程，可选延伸）。
- 跨被试（41/10）/ 跨数据集 / 跨模态 TTA；full CAP-EEGNet 在线 test-then-update 全模块。
- 任何写 `/share/workspace2`、任何登录节点跑重活。

## 9. File list（本 Phase 新增/规划）

| 路径 | 状态 | 作用 |
|:---|:---|:---|
| `3_online_adaptation/PHASE3_TTA_DESIGN.md` | 本文件 | Phase 3 路线/方法/矩阵 |
| `code/configs/experiments/phase3_tta.yaml` | 骨架已建 | WBCIC T3A/Tent/SHOT 配置（runner 待实现） |
| `code/configs/experiments/shu_phase3_tta.yaml` | 骨架已建 | SHU 对应配置 |
| `code/methods/t3a.py` | 待建(Step A) | T3A 核心 |
| `code/methods/{tent,shot}.py` | 待建(Step B) | 梯度类 TTA |
| `code/experiments/session_tta.py` | 待建(Step A) | TTA 协议（离线 replay + 活模型） |
| `code/experiments/tta_summarize.py` | 待建(Step C) | Phase 3 汇总 |
| `4_experiments/{wbci_shu,shu}/tta/{report,tables,figures}/` | 待产出 | 可读结果区（**硬约束1：放 4_experiments，不放 3_online_adaptation**） |
| `outputs/experiments/{wbci_shu,shu}/tta_v1/` | 待产出 | heavy 层（每 run CSV + cell_id） |
