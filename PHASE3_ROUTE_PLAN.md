---
title: "Phase 3 路线计划 v2.1（Oracle 先裁决版，学长已批准 + 7 条硬约束）"
tags:
  - "#modality/eeg"
  - "#method/test_time_adaptation"
  - "#method/prototype"
created: "2026-07-08"
updated: "2026-07-08"
status: "APPROVED (2026-07-08) — Phase 0 + A0/A1 可开工，含 7 条硬约束"
---

# Phase 3 路线计划 v2.1 —— Oracle 先裁决（学长已批准，含 7 条硬约束）

> **审批状态（2026-07-08）**：学长已批准 **Phase 0 + Phase A0/A1**，主路线 v2 不推翻。
> 附带 **7 条上机前必须落实的硬约束**（见 §2.5），其中 **cell_id、Oracle 泄漏标识、Mahalanobis 数值稳定性**
> 三条不补后面结果无法 defend——本版已全部写入对应 Phase。
>
> 本版按我们项目真实结构（`AGENTS.md` 灵魂记忆、`code/runners.py` 的 `PHASE_RUNNERS`、
> `code/summaries/`、双数据集 `{wbci_shu,shu}` 并列、`4_experiments/.../prototype_drift/tables/`）
> 落地，**不照搬学长文档里的 P10 / CLAUDE.md / `run_phase()` 等不同架构的名字**。
> 技术总纲（方法/公式/矩阵）见 `3_online_adaptation/PHASE3_TTA_DESIGN.md`（已同步重排）。

---

## 0. 最新总原则（为什么要重排）

Phase 2c 的权威结论（两数据集一致，见 `4_experiments/{wbci_shu,shu}/prototype_drift/report/AI_ANALYSIS.md`）：

- **prototype drift 只解释了部分掉点**（多元 R²≈0.35）；
- **主机制是 within-class scatter 膨胀 / Fisher collapse**（WBCIC 4.58→1.57；SHU 1.96→0.79），
  **不是 centroid collapse**（类中心反而更远）；
- **cosine 几何优于 euclidean**。

⟹ **推论**：如果连"已知目标真原型 / scatter-aware 几何"这种**理论上限（Oracle）**都救不回多少 acc，
那么 T3A 这种"用带噪伪标签重估原型"的方法更不可能有大收益。**所以必须先做 Oracle 上限裁决，
再决定要不要大规模上 T3A。** 这就是本版最重要的调整。

> **一句话**：先校验能否正确 replay Phase 2c 的 embedding → 再跑 T3A 最小 smoke（只证 pipeline 能跑）
> → **然后由 Oracle 上限实验裁决方向** → 通过才做 T3A ablation / safe-T3A / 全量 → 最后收敛论文叙事。
> **T3A 现在只能作为"无梯度 prototype baseline"，不得提前写成"最终修复方案"。**

---

## 1. 项目结构映射（学长文档 → 我们项目）

| 学长文档里的东西 | 我们项目里的对应 |
|:---|:---|
| `CLAUDE.md`（当前任务/状态） | **`AGENTS.md`**（唯一灵魂记忆）+ `0_docs/STATUS.md` + `README.md`；`CLAUDE.md` 只是指针 |
| `code/run.py` 的 `run_phase()` | `code/run.py` + **`code/runners.py` 的 `PHASE_RUNNERS` 字典**（已有 phase0–2c） |
| 无 `summaries/` | 我们**已有** `code/summaries/`（session/multisource/alignment/canonical/summarize） |
| 需新建 `PHASE3_TTA_DESIGN.md` | **已存在** `3_online_adaptation/PHASE3_TTA_DESIGN.md`（本版重排 Oracle 顺序） |
| `0_docs/prototype_drift/tables/trial_embeddings_index.csv` | `4_experiments/{wbci_shu,shu}/prototype_drift/tables/trial_embeddings_index.csv`（可读汇总层）<br>+ per-seed `outputs/experiments/{wbci_shu,shu}/prototype_drift_v1/runs/embed_index__{model}__seed{n}.csv`（重算层） |
| `4_experiments/phase3_tta/` | **`4_experiments/{wbci_shu,shu}/tta/`（已定死，见 §2.5 硬约束 1）**；`3_online_adaptation/` 只放设计文档，不放实验结果 |
| `.npz` embedding | `outputs/experiments/{wbci_shu,shu}/prototype_drift_v1/embeddings/{model}/seed{n}/{subj}_{src}-to-{tgt}.npz` |

---

## 2. 🐞 必须先修的已知 bug（我已定位，实现时自动处理）

1. **WBCIC embed_index 的 `npz_path` 是失效旧路径**：
   `outputs/experiments/wbci_shu/prototype_drift_v1/runs/embed_index__*.csv` 里的 `npz_path` 列写的是
   `outputs/experiments/prototype_drift_v1/embeddings/...`（**缺 `wbci_shu/`，文件已迁移，旧路径不存在**）；
   SHU 的 index 路径是对的。
   **修复方式**：Phase 3 runner **绝不信任 CSV 里的 `npz_path` 列**，而是用 config
   `source_embeddings.embeddings_dir` + `{model}/seed{n}/{subj}_{src}-to-{tgt}.npz` **重新拼路径**，
   并加"文件缺失就 fail-fast 报清楚是哪一格"。（我不会去改写 Phase 2c 的已完成产物，只在消费端修。）
   > 已确认 `code/configs/experiments/phase3_tta.yaml` 里的 `embeddings_dir` 指向的是**正确的新路径**。
2. **旧的散落 `outputs/experiments/prototype_drift_v1/`（无数据集前缀）**：属于迁移前残留引用。
   Phase 3 一律走 `{wbci_shu,shu}/` 作用域，不碰旧路径。
3. **文档状态过期**（Phase 0 修正，见下）：个别文档还留着 "Step 3 / Step 4 / qualified go" 等旧措辞，
   会让其它 agent 误判进度；本轮统一改成"Phase 2c 已完成 → Phase 3A（replay+smoke）→ Phase 3B（Oracle 裁决）"。

---

## 2.5 ⭐ 上机前必须落实的 7 条硬约束（学长批准补强，强制遵守）⭐

> 这 7 条是**上机前置条件**。任何 Phase 3 代码/报告都必须满足；尤其 **#2 cell_id、#4 Oracle 泄漏标识、
> #5 Mahalanobis 数值稳定性**——不补，后面结果无法 defend。下面每条都已落进对应 Phase。

1. **结果目录定死**：Phase 3 全部实验结果放 **`4_experiments/{wbci_shu,shu}/tta/`**（与 `prototype_drift` 同级）。
   **`3_online_adaptation/` 只放设计文档，不放任何实验结果。** config 里 `output.readable_dir` 必须相应改为
   `4_experiments/{wbci_shu,shu}/tta`。

2. **A0 必须定义 canonical `cell_id`**：每一格至少固定这些字段并作为 join key，
   否则 No-TTA 数值即使对上也可能是**错行 join**：
   ```text
   dataset, model, seed, subject, source_session, target_session,
   cell_id, npz_path_resolved, n_target_trials
   ```
   `cell_id` 建议 = `{dataset}__{model}__seed{seed}__{subject}__{source}->{target}`（唯一、可读、可复算）。
   No-TTA 复现按 `cell_id` 对齐 Phase 2c 的 `acc_target`，不靠行号/顺序。

3. **A1 被试自动选，禁止手选**：stable / high-drift 被试**必须从 Phase 2c drift tertile 表自动选**（读
   `1_session_drift/wbci_shu/tables/` 的 per-subject drift 分层），并落盘 `selected_smoke_cells.csv`
   记录选取依据（drift 分层 + 数值），避免 reviewer 质疑 cherry-pick。

4. **Oracle 命名必须带泄漏标识**：任何用了 target 真标签的方法，字段/命名显式标 `used_target_labels=True`
   （如 `target_label_oracle_proto.used_target_labels=True`）。报告里 **Oracle 单列为 "Oracle diagnostic only"**，
   **绝不与 deployable T3A 放进同一张主效果表**。

5. **Mahalanobis / shrinkage 先防数值不稳**：EEG embedding 维度（如 496）常 > 每类 target trials（~100），
   直接估协方差易病态。默认必须：**shrinkage covariance + ridge epsilon + 记录 condition number**；
   不满足条件（如 cond number 超阈 / 样本不足）就 **fail-fast 或降级到 cosine oracle**，并在输出里标注降级。

6. **Phase B 裁决门要同时看收益和风险**：不只看 `+3pp / +1pp` 均值，还要看
   **negative transfer rate、drift tertile 分层收益、Fisher ratio 恢复**。
   即使均值 > +3pp，但**若 high-drift 或某个模型负迁移很高，不得直接进入 full T3A sweep**（降级为小范围验证）。

7. **FBCNet 与 SHU 单列**：FBCNet 几何异常、SHU 近 chance（地板效应）。二者用于**验证外推性**，
   **不得与 WBCIC 的 EEGNet/DeepConvNet 混成一个主结论**。主结论以 WBCIC × {EEGNet, DeepConvNet} 为准。

---

## 3. 完整路线（Phase 0 → G）

### Phase 0 —— 修正项目状态（文档，低风险，本轮我直接做）
**目标**：任何 agent 读根目录就知道"Phase 2c 已完成，下一步是 replay 校验 + T3A smoke + Oracle 裁决"，不再误以为还停在 Step 3。

- 更新 `AGENTS.md`：当前第一优先改为 **Phase 3A（replay + No-TTA 精确复现 + minimal T3A smoke）→ Phase 3B（Oracle 裁决）**；逻辑链里把"Oracle"标为 T3A 大扫**之前**的裁决门。
- 更新 `0_docs/STATUS.md`：一句话现状 + 下一步同步为 Oracle-先裁决。
- 更新 `README.md`：§0 定位补"两数据集 Phase 0–2c 全部完成；下一步 Phase 3"。
- 更新 `results.md`：把 WBCIC Prototype Drift 行的 "Step 4 Oracle" 措辞对齐为 "Phase 3B Oracle（裁决门）"。
- 更新 `progress.md`：加一条本次计划重排的日记。
- 同步 `3_online_adaptation/PHASE3_TTA_DESIGN.md`：Oracle 从 Step E 提前为 **Phase B（裁决门）**。
- （历史产物 `AI_ANALYSIS.md` 里的 "Step 4" 是冻结记录，不改。）

**验收**：根目录任一入口文档都能读出"Phase 2c done → 下一步 replay/smoke/Oracle"。

---

### Phase A0 —— Embedding Replay + No-TTA 精确复现（第一关，纯 CPU）
**目标**：先证明我们能正确读取 Phase 2c 的 embedding 与 cell 元数据。**这一关不过，后面 T3A/Oracle 全部不可信。**

- 输入：
  - `4_experiments/wbci_shu/prototype_drift/tables/trial_embeddings_index.csv`（或 per-seed `outputs/.../runs/embed_index__*.csv`）
  - `4_experiments/wbci_shu/prototype_drift/tables/prototype_drift_metrics.csv`（含每格 `acc_target`）
  - Phase 2c 的 `.npz` embedding（路径按 §2 重拼）
- 实现：
  - 新建 `code/experiments/session_tta.py`（读 index + npz → 逐 cell 组织 target 流 → 算 No-TTA）。
  - 复用已确认存在的 `code/configs/experiments/phase3_tta.yaml`（骨架已就绪；`output.readable_dir` 改为 `4_experiments/wbci_shu/tta`，见硬约束 1）。
  - `code/runners.py` 里按现有风格注册 `phase3_tta` 到 `PHASE_RUNNERS`；`code/run.py` 无需大改（它按 experiment 名 dispatch）。
- **【硬约束 2】canonical `cell_id`**：每行输出至少含
  `dataset, model, seed, subject, source_session, target_session, cell_id, npz_path_resolved, n_target_trials`；
  `cell_id = {dataset}__{model}__seed{seed}__{subject}__{source}->{target}`。**No-TTA 复现按 `cell_id` join Phase 2c，不靠行号。**
- **【bug 修复】** `npz_path_resolved` 由 config `embeddings_dir` 重拼（不信任 index 的 `npz_path` 列，见 §2），缺文件 fail-fast。
- **硬验收**：
  ```text
  No-TTA 复算的 acc_target 必须按 cell_id 逐 cell 对齐 Phase 2c，|Δ| < 1e-6
  （对齐前先断言两侧 cell_id 集合完全一致，无缺失/多余，防错行 join）
  ```

---

### Phase A1 —— Minimal T3A Smoke（只证 pipeline 能跑，不出结论）
**目标**：只验证 T3A runner 能端到端跑，**绝不写成"T3A 有效"**。

- 范围：数据集 **WBCIC-SHU**；模型 **EEGNet**；被试 **2 个（一个 stable、一个 high-drift）**；seed **0**；方法 **`no_tta` + `t3a_src_proto_cosine`**。
- **【硬约束 3】被试自动选**：从 Phase 2c drift tertile 表（`1_session_drift/wbci_shu/tables/` per-subject drift 分层）
  **自动**各取 1 个 stable / 1 个 high-drift，**禁止手选**；落盘 `selected_smoke_cells.csv`（含 subject、drift 分层、依据数值）。
- 新建 `code/methods/t3a.py`：支持 **source prototype 初始化**、**cosine 相似度**、**filter_k=[20]**（先单点）。纯 numpy，无梯度。
- 输出（放 `4_experiments/{dataset}/tta/`，见硬约束 1）：
  ```text
  4_experiments/wbci_shu/tta/smoke/
    selected_smoke_cells.csv      # 硬约束 3：自动选被试的证据
    tta_cell_metrics.csv          # 含 cell_id（硬约束 2）
    tta_smoke_report.md
  ```
- **口径**：报告只写"pipeline 可跑 + No-TTA 对齐通过"，**不写科研结论**。

---

### Phase B —— ⭐ Oracle 上限实验（提前！科研裁决门，纯 CPU）⭐
**这是本版最关键的调整**：Oracle **不放最后**，而是放在 T3A 大扫**之前**，用来裁决"原型/几何修复方向到底有没有天花板"。

**要回答的问题**：
> 如果 target prototype / scatter-aware 几何是正确的修复方向，理论上最多能追回多少性能？

**方法组**（都复用 Phase 2c 已存 embedding + target 真标签做**离线诊断上限**，非部署方法）：

| 方法 | 目的 | 【硬约束 4】泄漏标识 |
|:---|:---|:---|
| `source_proto` | frozen source 几何基线（参照） | `used_target_labels=False` |
| `target_label_oracle_proto` | 用 target 真标签算原型 → 上限 | `used_target_labels=True` |
| `cosine_oracle_proto` | 验证归一化/cosine 几何（呼应 Phase 2c cosine>euclidean） | `used_target_labels=True` |
| `mahalanobis_oracle` | 验证 scatter-aware 是否优于纯 centroid（直击 scatter 膨胀主机制） | `used_target_labels=True` |
| `shrinkage_oracle` | 协方差收缩，降低 Mahalanobis 不稳定 | `used_target_labels=True` |
| `reliability_weighted_oracle` | 验证 trial reliability 加权是否有用 | `used_target_labels=True` |

**【硬约束 4】Oracle 泄漏隔离**：每行必须带 `used_target_labels` 字段；报告里 Oracle **单列为 "Oracle diagnostic only"**，
**绝不与 deployable T3A（伪标签、`used_target_labels=False`）放进同一张主效果表**。Oracle 是"天花板参照"，不是方法竞品。

**【硬约束 5】Mahalanobis / shrinkage 数值稳定性**（EEG embedding 维度 496 常 > 每类 target trials ~100，协方差易病态）：
- 默认 **shrinkage covariance（Ledoit-Wolf 或固定 α）+ ridge epsilon**；每格**记录 condition number**。
- 不满足条件（cond number 超阈 / 每类样本不足）→ **fail-fast 或降级到 `cosine_oracle`**，并在输出里显式标 `degraded_to=cosine`。
- 输出列至少含：`cov_shrinkage_alpha, ridge_eps, cond_number, degraded_flag`。

**关键指标（必须分层报告，不能只报 overall mean）**：
- balanced accuracy、**negative transfer rate（掉点 cell 比例）**
- within-class scatter、**Fisher ratio 恢复**、negative-margin rate、class-collapse rate
- 按 **drift tertile（stable/moderate/high）** 分组收益、**按 model 分组**

**【硬约束 6】裁决门 = 收益 + 风险，双条件**：
- **进入 full T3A（Phase C/E）需同时满足**：均值 balanced-acc 恢复 **> +3pp** **且**
  high-drift tertile 与各主力模型（EEGNet/DeepConvNet）**均无高 negative transfer**（无明显负迁移）**且** Fisher ratio 有实质恢复。
- **停止大规模 T3A（转 scatter/reliability/decision-boundary，论文路线 3）**：均值 **< +1pp**，或 Oracle 几乎不恢复 Fisher。
- **中间/割裂情形**（如均值 >+3pp 但 high-drift 或某模型负迁移高）：**不得直接 full sweep**，降级为小范围验证 + 定位可修子集。

> **注意**：`*_oracle_*` 用了 target 真标签，是**离线上限诊断**，**不是可部署方法**（见硬约束 4）。

---

### Phase C —— T3A 小规模 Ablation（仅当 Phase B 支持才做）
矩阵：

| 维度 | 候选 |
|:---|:---|
| init | `clf_weights`, `src_proto` |
| geometry | `dot`, `cosine` |
| filter_k | `5, 10, 20, 50, inf` |
| update | episodic=False，target stream 累积 |
| safety | **暂不加**，先看原始 T3A 的裸风险 |

报告必须按 **model / subject drift level / source→target direction / seed / negative transfer** 分层，不能只报 overall mean。

⚠️ **二分类退化提醒**：K=2 时预测熵是最大 softmax 概率的单调函数 → "熵筛选 ≡ 最大置信度筛选"，
entropy 与 max-conf **不是两个设置**；真正有区分度的是 filter_k / 几何 / 初始化 / 软硬 / margin。

---

### Phase D —— Safe / Reliability-weighted T3A（EEG 特色的起点）
仅当原始 T3A 有收益但**负迁移明显**时才加安全门：
- 高 entropy 不更新；class support 极不平衡不更新；confidence collapse 不更新；
- 低 reliability / 高伪迹 trial 不更新；rolling proxy 变差则 freeze 或 rollback。

> 定位：不是"把 T3A 搬到 EEG"，而是 **EEG-specific safe prototype adjustment**——这才是我们的差异化贡献点。

---

### Phase E —— 全量离线 Sweep（A/B/C/D 全过关后才做）
- **【硬约束 7】主结论 = WBCIC × {EEGNet, DeepConvNet}**。全 subject × directions × seeds。
- **FBCNet 单列**（Phase 2c 已提示其几何异常）——只作外推性验证，不并入主结论。
- **SHU 单列**（近 chance / 地板效应）——只作外部验证/补充，不一开始并入主结论。
- 报告必须按 model / drift tertile / direction / seed **分层**，并单列 `negative_transfer_analysis`。
- 输出：
  ```text
  4_experiments/wbci_shu/tta/full_sweep/
    tta_cell_metrics.csv                 # 含 cell_id + used_target_labels
    tta_summary_by_model.csv             # EEGNet/DeepConvNet 主结论；FBCNet 单列
    tta_summary_by_drift_level.csv
    negative_transfer_analysis.csv
    PHASE3_TTA_REPORT.md
  # SHU 平行输出到 4_experiments/shu/tta/full_sweep/（单列，不并入主结论）
  ```

---

### Phase F —— Tent / SHOT / CoTTA 对照（放后面，GPU）
需要活模型 + 梯度 + GPU，复杂度高，低 SNR EEG 风险更大。定位：
- Tent = gradient-based TTA 对照；SHOT = source-free 参考（非严格 online 主线）；CoTTA = continual 参考；
- **T3A / safe-T3A = 主线候选**。

---

### Phase G —— 论文叙事收敛（由结果裁决）
- **路线 1（Oracle + Safe-T3A 有效）**：*Cross-session MI degradation can be partially recovered by cosine-space, reliability-gated prototype adjustment.*
- **路线 2（Oracle 有效但 T3A 无效）**：*Target geometry is recoverable, but pseudo-label support construction is the bottleneck.*
- **路线 3（Oracle 也无效）**：*Cross-session degradation is dominated by representation diffusion beyond centroid correction; prototype TTA is insufficient.*

---

## 4. 决策项（学长已拍板，锁定）

| # | 选项 | 最终决定 |
|:--|:--|:--|
| 1 | Phase 3 结果放哪 | **锁定 `4_experiments/{wbci_shu,shu}/tta/`**（硬约束 1）。`3_online_adaptation/` 只放设计文档。 |
| 2 | A1 被试选谁 | **从 Phase 2c drift tertile 表自动选** 1 stable + 1 high-drift，落盘 `selected_smoke_cells.csv`（硬约束 3，禁手选）。 |
| 3 | Oracle 裁决门槛 | **收益+风险双条件**（硬约束 6）：均值 >+3pp 且 high-drift/各模型无高负迁移且 Fisher 有恢复 → 继续；<+1pp → 转向。 |
| 4 | 本轮做到哪 | Phase 0（文档修正）已完成；**Phase A0/A1 已获批可开工**（见 §5）。 |

---

## 5. 执行状态

- ✅ **Phase 0 已完成**（纯文档）：AGENTS/STATUS/README/results/progress + DESIGN 重排 + 本计划 + `PROJECT_ARCH_SYNC_FOR_ADVISOR.md`。
- ✅ **Phase A0/A1 Round-1 scaffold（2026-07-10）**：`code/tta/` + `session_tta.py` + `phase3_tta` runner；
  tests 14 passed；WBCIC smoke 子集 A0 replay **3/3 `|Δ|=0`**；minimal T3A + Oracle diagnostic 跑通。
  **口径**：framework/smoke runnable，**不是** full T3A experiment / 科研结论。
- ⏳ **待做**：全量 A0 replay；学长 pretrained model adapter；Phase B Oracle 全量裁决；catalog 复杂方法。
- 🔒 **纪律**：T3A 只作无梯度 prototype baseline；Oracle 未裁决前不做全量；FBCNet/SHU 单列不入主结论；
  未跑不写 done；不碰 `/share/workspace2`；不覆盖 Phase 2c 的 `*_v1` 产物。

---

## 6. 审批记录（学长口径，已生效）

```text
批准 Phase 0 + Phase A0/A1。
总路线按 v2：Oracle 提前作为 Phase B 裁决门；A0 先做 No-TTA 逐 cell 精确复现 Phase 2c acc_target，|Δ|<1e-6；
A1 只做 minimal T3A smoke，不写科研结论。
实现以真实仓库结构为准：/share/home/yuan/SYX/eeg-mi-online，AGENTS.md，code/runners.py 的 PHASE_RUNNERS，
code/summaries/，4_experiments/{wbci_shu,shu}/tta/。
```
+ 7 条硬约束（§2.5）：结果目录定死 / cell_id / A1 自动选被试 / Oracle 泄漏标识 / Mahalanobis 数值稳定 /
裁决门收益+风险双条件 / FBCNet 与 SHU 单列。
