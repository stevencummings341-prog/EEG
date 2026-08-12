---
title: "跨被试实验协议讨论备忘（发学长确认用）"
tags:
  - "#pipeline/5_dl_model"
  - "#modality/eeg"
created: "2026-08-04"
updated: "2026-08-04"
status: "partially_locked — WBCIC 3C LOSO paper-aligned (2026-08-07); 2C still open"
---

# 跨被试实验协议讨论备忘

> **用途**：发给学长确认协议；2026-08-04 学长已**先锁定 WBCIC 三分类**开跑顺序。
> **2026-08-07**：3C 划分已按 DSGNet 论文（SHUv5）对齐并重跑（见 §0）。
> 文献依据全文见 `inbox/cross_subject_protocol_research.md`；
> 工程落地见根目录 `FOUNDATION_E2E_ROUTE_PLAN.md`。

---

## 0. 学长已锁定（2026-08-04）+ 论文对齐（2026-08-07）— WBCIC 3C

| 项 | 决定 |
|:---|:---|
| 数据 | WBCIC-SHU **三分类 11 人**（= DSGNet 文中 SHUv5 3C；不是 SHU 2022；不是 2C 的 51 人） |
| 划分 | **LOSO** + **session val**：非测试被试 ses-01+02 train / ses-03 val；测试被试三 session 全测 |
| 模型 | **5 个 foundation**（本轮）；**DSGNet** 作论文数字对标（Acc 0.6856），本仓复现 deferred |
| checkpoint | 只存 **best + last**；支持断点续跑 |
| run_id | `foundation_3c_loso_paper_v1`（旧 `foundation_3c_loso_v1` = 留 2 个 val 被试，**不对齐论文**） |

对标论文：Lou et al., IEEE JBHI 2026, [doi:10.1109/jbhi.2026.3689121](https://doi.org/10.1109/jbhi.2026.3689121)。
PDF：`inbox/papers/dsgnet_jbhi2026_FullText.pdf`。Table II SHUv5：DSGNet Acc **0.6856** /
F1 **0.6833** / Kappa **0.5284**（次强 ATCNet Acc 0.6834）。

工程入口:

```bash
python scripts/preprocess_wbci_3c.py   # 官方 3C .mat -> npz（已跑过可跳过）
python code/run.py --config code/configs/experiments/foundation_cross_subject_wbci_3c.yaml --device cuda
```

**注意**：DSGNet 官方 GitHub 目前只有架构预览（`temp_model.py` / `temp_module.py`），
README 写 *full code upon acceptance*；完整复现需确认训练脚本/超参是否另有交付。本轮先用
5 个 foundation 在**同一协议**下出数，再与论文 DSGNet 数字对比。

二分类（51 人）协议（5 折 / seed 41–45 等）**尚未**作为本轮开跑条件；本轮优先 3C LOSO。

---

## 1. 进度一句话

学长给的 `models_eeg_foundation/`（5 个模型）**已完整融入项目框架**。
**2026-08-07**：3C 协议已对齐 DSGNet 论文并提交 `foundation_3c_loso_paper_v1` 全量重跑；
DSGNet 本仓接入仍待完整训练代码。

---

## 2. 文献调研的 4 个关键发现

1. **两个数据集都没有官方跨被试划分**，两篇数据集论文自己也都只做了 within-session /
   cross-session，**都没有做零样本跨被试**。所以我们没有现成协议可以照抄。

2. **WBCIC-SHU 2025 的跨被试文献实际上只有一篇**（该数据集共 18 次引用，逐条看过，绝大多数
   只是引用它但实验做在 BCI IV-2a 上）：

   > **EDAPT**, *J Neural Eng* 2026, <https://doi.org/10.1088/1741-2552/ae5689>
   > 协议：**2-fold 被试交叉验证（50% 被试训练 / 50% 被试测试）**，被试的多个 session 拼接，
   > 2–47 Hz 带通 + 平均重参考，可选 per-subject Euclidean Alignment，预训练 100 epoch /
   > Adam / lr 1e-4 / batch 64。
   >
   > **零样本跨被试 accuracy（mean ± std over test subjects）**：
   > EEGNet **0.81 ± 0.12** · ShallowConvNet **0.82 ± 0.12** · DeepConvNet **0.85 ± 0.12** ·
   > ATCNet **0.71 ± 0.12**。

   这是我们唯一的对标锚点。**注意它用的是 59ch / 1000Hz 原始数据，我们用的是 58ch / 250Hz
   官方预处理版，数字不能直接比，只能当量级参考。**

3. **SHU 2022 有地板效应，而且没有可验证的跨被试基准。** 数据集作者自己的跨 session 只有
   **53.7%**，与 chance（51.4–53.7%）**无显著差异（p>0.05）**。唯一声称在 SHU 上做 LOSO 的
   DSGNet（IEEE JBHI 2026）是 early access，分数据集的数字看不到。所以 SHU 跨被试大概率落在
   52–55%，**这是预期而不是 bug**，和我们 Phase 1（cross-session EEGNet 0.538）完全一致。

4. **WBCIC-SHU 有一个容易踩的混淆**：within-session accuracy 从 session 1 的 81.77% 升到
   session 3 的 88.90%，作者解释为**被试在学会做 MI**（技能习得），不是纯粹的漂移。
   把 3 个 session 全池化训练 = 把三种技能水平混进同一个训练分布。

   另外两条陷阱：WBCIC 官方 benchmark 的 stage-2 早停条件是**看 test loss**（论文原文），
   所以 85.32% 是乐观上限、不是干净的泛化估计；SHU 每 session 是 90–100 trial 不等
   （坏 trial 被剔除），**类别不一定 50/50**，要算 balanced accuracy。

---

## 3. 三个候选方案（算力用「训练次数」表示，避免瞎猜墙钟时间）

以 **5 个模型 × 1 seed** 计（seed 数一乘就翻倍）：

| 方案 | fold 数（WBCIC / SHU） | 训练次数 | 相对算力 | 每 fold 训练被试数（WBCIC） |
|:---|:---|---:|---:|---:|
| **A — LOSO** | 50 / 25 | 375 | 5.0× | ~49 被试 ≈ 29k trial |
| **B — 5-fold subject-grouped**（推荐） | 5 / 5 | 50 | **1.0×** | ~41 被试 ≈ 24k trial |
| **C — 固定 3 份划分** | 1 / 1 | 10 | 0.2× | 31 被试 ≈ 18k trial |

* **A 的优点**：对标文献（Kwon、BARN-DA、DSGNet 都用 LOSO），源被试最多，每个被试都有单独数字。
  **缺点**：贵 5 倍，而且 50 个 fold 产出大量近重复模型。
* **B 的优点**：MOABB 的 `CrossSubjectEvaluation(n_splits=5)` 就是这个（一等公民选项，不是野路子），
  **每个被试仍然恰好被测一次**，所以「per-subject 表格」和「mean ± std over subjects」的算法
  和 LOSO 完全一样、可直接对比。省下来的算力可以拿去跑 5 个模型 / 多 seed / EA 消融。
  **缺点**：每 fold 少 ~8 个训练被试，按 EDAPT 的 scaling 曲线结果会**略微偏保守**（偏保守是安全方向，
  但和文献 LOSO 数字比较时要说明）。
* **C**：适合当开发/消融台，测试集只有 10 个被试、方差大、没有 fold 级重复，**不能当主结论**。

**我们目前 config 里的默认 = 方案 B**（`kfold_subject`, `n_folds=5`, `split_seed=0`），
验证集用**留出的训练被试**（15%，≈8 个）——这跟 SHU 作者自己 CSA 里「留 3 个源被试做验证」
同一思路，保证模型选择完全不看测试被试。

---

## 4. 需要学长拍板的 7 个问题

| # | 问题 | 我的建议 | 影响 |
|:--:|:---|:---|:---|
| 1 | **LOSO 还是 5-fold subject-grouped？** | 主表用 B（5-fold）；如果要对标文献，就在 SHU（小数据集）上补一遍 LOSO 做一致性检查 | 算力 5× |
| 2 | **严格零样本，还是允许目标被试少量校准？** | 主结论做**严格零样本**（目标被试 0 数据）；校准作为后续 arm（EDAPT 证明 20 trial warm-up + 50 trial 滑窗就能补回大部分差距） | 决定论文的 claim 是什么 |
| 3 | **session 池化还是分层？** | 先池化（和 EDAPT 一致）；因为有技能习得混淆，建议**额外报一个「只用 session 1 训练」的对照** | +1 组实验 |
| 4 | **要不要加 Euclidean Alignment arm？** | 建议加，但作为独立 arm 不设为默认。文献明确推荐它为跨被试标准步骤（+4.33%，收敛快 70%+），而且它是 per-subject 无标签的，零样本协议下合法。我们 Phase 2b 已有现成实现 | 那部分网格 ×2 |
| 5 | **epoch 预算：100 还是 200–300？** | 先 100 + patience 25，用 GPU smoke 实测单 epoch 耗时后再决定要不要加到 200 | 直接线性影响总算力 |
| 6 | **5 个模型全跑，还是先跑便宜的看趋势？** | 先 `s4erp`（实测 1.37M @58ch / 0.94M @32ch）+ `dualcd_s4_pos`（3.17M / 2.32M）跑通看趋势，再放 `dualcd_s4_flatten`（66.9M / 66.0M）与 `dualcd_transformer`（67.9M / 67.0M） | 前两个只占总算力一小部分；后两个的参数几乎全在 DINO projection head |
| 7 | **SHU 近 chance 怎么办？** | 如实报（这是诚实的负结果，且与我们 Phase 1/2b 的 SHU 结论一致）；同时说明 chance band 是 51.4–53.7%，1–2pp 的「提升」在这个地板上是噪声 | 影响 SHU 结果怎么写 |

补充两个小问题：
* **通道要不要取运动区子集？** Kwon 在 OpenBMI 上用了 20 个运动区电极；但**我们这两个数据集
  都没有已发表的通道子集推荐**。可以在方案 C 上做个便宜的消融。
* **要不要复现官方 within-session benchmark 当参考行？** 如果要，得先决定怎么处理它
  stage-2 看 test loss 的泄漏问题（照抄并标注，还是修正后报一个更低的数字）。

---

## 5. 定完之后我要做的事

1. 把确认的参数写进两个 config（`foundation_cross_subject.yaml` / `shu_foundation_cross_subject.yaml`），
   并把本文件 status 从 `pending_advisor_confirmation` 改成 `approved` + 日期。
2. GPU 节点 smoke：1 fold × `s4erp`，实测单 epoch 耗时与显存，据此定 epoch 预算和 batch size。
3. 全量跑（Slurm，可断点续跑）→ summarize → 写 AI 分析报告到
   `4_experiments/{wbci_shu,shu}/foundation_cross_subject/report/`。
4. 报告口径固定为：**per-subject 表格 + mean ± std over subjects**，accuracy / balanced accuracy /
   macro-F1 / AUC + 校准指标，并且 **best.pt 与 last.pt 两套数字并排给**。
