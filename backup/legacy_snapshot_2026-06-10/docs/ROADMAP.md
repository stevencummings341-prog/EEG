# ROADMAP — CAP-EEGNet full vision & staged plan

> **本文件是「项目方向对齐」的权威文档。** 它把学长聊天记录
> (`docs/references/ChatGPT-EEG-MI-pretraining.md`，尤其 sec.13–16) 的最终方案，落成一份
> 分阶段路线图，并明确区分「现在跑通的 Stage 0 minimal」与「论文最终的 full CAP-EEGNet」。
> 配合 `docs/MODEL_PLAN.md`（模型细节）、`docs/EXPERIMENT_PROTOCOL.md`（实验细节）、
> `docs/ALIGNMENT_CHECKLIST.md`（逐项对齐状态）阅读。

---

## 0. 最终目标（一句话，不可弱化）

**Confidence-aware Online Adaptive Multi-Subagent Pretraining Framework for
Cross-subject MI EEG Decoding**

中文：**面向跨被试运动想象 EEG 的「置信度感知 + 在线自适应 + 多神经子模块」预训练框架。**

> 这 **不是** 一个普通 EEGNet / DeepConvNet 分类器。普通 EEGNet 分类只是 Stage 0 的链路
> 验证 baseline。最终方法的卖点是：多神经子模块（neural experts）+ 数据集感知路由 +
> 多源样本置信度 + 多级原型记忆 + 轻量适配 + 跨被试对齐 + 在线 test-then-update。

核心约束（贯穿全程，任何阶段都不许违反）：
- 按 **被试** 划分，绝不按 session/trial；target 绝不泄漏进 source。
- 训练入口 = 自己预处理的 `status==ok` 的 `.npz`；derivatives 的 `.mat` 只做 QC，**绝不训练**。
- confidence **绝不**简化成 softmax 最大值。
- 在线 **必须** test-then-update；默认 **不**更新 full backbone。
- 所有子模块都是 **可微分深度网络**，不是手工特征（CSP/DE/TE）直接拼接。

---

## 1. full CAP-EEGNet 应包含的模块

| # | 模块 | 说明 | 实现状态 |
|---|---|---|---|
| 1 | **Main encoder** | EEGNet-style 主干（时间卷积→depthwise 空间→separable），[B,1,58,1000]→特征 | ✅ Stage 0 已实现 |
| 2 | **Neural subagents** | 多神经子模块/neural experts：时频、空间拓扑、熵/复杂度、动态连接等视角，端到端可微，可靠性加权融合 | 🚧 未实现（接口+flag 已留） |
| 3 | **Confidence head** | 多源置信度：predictive entropy + prototype margin + augmentation consistency + OOD/calibration（**非 softmax max**） | 🚧 未实现 |
| 4 | **Prototype memory** | global / subject / session 三级原型；高置信样本动量更新；输出 distance/margin | 🚧 未实现 |
| 5 | **Adapter** | 轻量 bottleneck/LoRA；target 微调与在线适应的主要可训练部分（backbone 冻结） | 🚧 未实现 |
| 6 | **Domain alignment** | subject/session 对齐（adversarial/CORAL/MMD/prototype align）：min H(Y\|Z) 且 max H(S\|Z) | 🚧 未实现 |
| 7 | **Online update** | test-then-update；置信度门控；只更新轻量模块；EMA teacher/replay/动量/蒸馏稳定机制 | 🚧 未实现 |
| + | **Dataset-aware router**（可选 v2） | 按数据集元特征/统计/probe 给各 subagent 分配权重 | 🚧 未实现（v2） |

> 代码位置：`src/models/cap_eegnet.py` 已含上述全部组件的 **config flag + 占位类 +
> 明确 NotImplementedError**（提示语统一为「Reserved for full CAP-EEGNet … NOT implemented
> in the minimal sanity model」）。`src/models/eegnet.py` 是已实现的 main encoder。

预训练多任务损失（最终形态，sec.12）：
`L = L_cls + λ1·L_mask + λ2·L_proto + λ3·L_conf + λ4·L_cons + λ5·L_domain`。
在线损失：`L_online = c_i·L_pseudo + λ1·L_proto + λ2·L_distill + λ3·L_calib`（`c_i`=预测置信度）。

---

## 2. minimal vs full CAP-EEGNet（务必区分）

| | Stage 0 **minimal**（现在） | **full** CAP-EEGNet（最终） |
|---|---|---|
| 组成 | EEGNet encoder + 线性分类头 | 上表 1–7 (+router) 全部 |
| forward 返回 | `{logits, features, proto_dist=None, confidence=None}` | `proto_dist`/`confidence` 真正填充，含多源置信度 |
| 置信度 | 无（None） | 多源融合，非 softmax max |
| 适配 | 无 | adapter + prototype + domain，backbone 冻结 |
| 在线 | 无 | test-then-update，轻量更新 |
| 定位 | **只验证数据/训练链路，是 baseline，不是论文方法** | **论文最终方法** |

**铁律：任何报告/论文都不得把 minimal 当作最终方法。** minimal 只在 Stage 0 与
ablation 里作为「无 confidence/prototype/adapter/online」的下界对照。

---

## 3. 分阶段路线（Stage 0 → 5）

### Stage 0 — Training infrastructure / sanity ✅ 已完成
- 41/10 subject-wise split（seeds 2026–2030）、`SHUTrialDataset`、minimal EEGNet encoder+
  classifier、小规模 sanity training。
- 目的：**只验证数据链路与训练链路能跑通**，不是论文最终模型。
- 产物：`src/data/splits.py`、`splits/cap_eegnet_4110_seed*.json`、`tests/test_splits.py`、
  `tests/test_dataset_smoke.py`、`src/models/{eegnet,cap_eegnet}.py`(minimal)、
  `scripts/sanity_train.py`、`outputs/sanity_check/sanity_check_metrics.json`。
- 阻塞：`mi_torch` torch 仍是 CPU-only → 正式 GPU 训练前需装 cu118 版（见 `docs/ENVIRONMENT.md`）。

### Stage 1 — Experiment 1：cross-subject pretraining + zero-shot 🚧
- 对 seeds 2026–2030 各做一次 repeated 41/10 subject-wise split，**每个 split 训练一个模型**。
- source 全部 ok session 训练；target 全部 session **zero-shot** 测试。
- 报告 **mean ± std**（跨 5 个 seed）。target 绝不进入 source 训练。
- 指标：Accuracy / Balanced Accuracy / Macro-F1 / AUC。
- 注：Stage 1 可先用 minimal 或「main encoder + 分类头」跑通跨被试 pipeline，但论文主结果
  必须用含 Stage 2 组件的模型。

### Stage 2 — 加 confidence / prototype / adapter（+domain）🚧
- **confidence**：多源（predictive entropy, prototype margin, consistency, OOD/calibration），
  **不许只用 max softmax**；训练加 calibration loss。
- **prototype**：至少 global class / subject / session 三级原型。
- **adapter**：用于 target 微调与在线适应。
- **domain alignment**：subject/session 不变表征（可与 prototype 协同）。
- minimal EEGNet+classifier **只能**作为 baseline / Stage 0，不是最终 CAP-EEGNet。

### Stage 3 — Experiment 2：target Session 1 fine-tuning 🚧
- 每个 target：用 **Session 1** 微调/校准，**Session 2+3** 测试。
- 对比：zero-shot / classifier-only / adapter / prototype / full-model。
- **优先 adapter + prototype + confidence calibration，不默认 full-model fine-tune。**
- 数据预算研究：full S1 / 前 20% / few-shot 5·10·20 每类 / 无标签。

### Stage 4 — Experiment 3：online test-then-update 🚧
- Session 2/3 **逐 trial**，严格顺序：①predict → ②record(pred/conf/correct) → ③then update。
- **禁止**先用整个 session 更新再测同一 session（未来信息泄漏）。
- 默认只更新轻量模块：prototype memory / adapter / confidence calibration head / BN(因果) /
  classifier(可选，有标签时)。**默认不更新 full backbone。**
- 两种标签模式：supervised / unsupervised（pseudo-label + 置信度门控）。
- 稳定机制：confidence threshold, EMA teacher, replay memory, prototype momentum,
  entropy reg, feature distillation, class-balanced memory, lr decay。
- 产物：`outputs/<run_id>/per_trial.csv`、online curve、before/after metrics。

### Stage 5 — Experiment 4：ablation & interpretability 🚧
- 至少：without confidence head / without prototype / without adapter / without online update /
  softmax-confidence only / no confidence threshold / update full backbone /
  不同 confidence threshold。
- 指标：acc / balanced acc / F1 / AUC + ECE / NLL / Brier / risk-coverage /
  confidence-accuracy curve。
- 可选 v2 ablation：without dataset router / without domain alignment。

---

## 4. 数据流（最终形态）

```
外部 raw BDF (51×3)
  → eog_ecg_clean 预处理 → status==ok 的 .npz（148 ok / 5 failed，QC PASS）
  → 41/10 subject-wise split (seeds 2026–2030, target 需 3 session 全 ok)
  → CAP-EEGNet：Neural Subagents → (Dataset Router) → Reliability-aware Fusion
                → Main Encoder → {Classification, Prototype, Confidence, Domain} heads
  → Exp1 zero-shot → Exp2 Session1 微调 → Exp3 Session2/3 在线 test-then-update → Exp4 消融
```

---

## 5. 现在不做的事（纪律）

- ❌ 不启动正式训练；❌ 不提交 sbatch 训练任务；❌ 不安装/改动 GPU 环境。
- ❌ 不把 derivatives `.mat` 当训练入口；❌ 不把 confidence 简化成 softmax max。
- ❌ 不实现会悄悄 full-backbone 在线更新的流程；❌ 不大改已通过的 preprocessing/split/dataset。
- ✅ 下一步：先解决 GPU torch（cu118 专用环境，见 ENVIRONMENT.md），正式训练需用户确认后再开始。
