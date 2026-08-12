---
title: "DSGNet SHUv5 3C comparison anchor (paper numbers)"
tags:
  - "#modality/eeg"
  - "#pipeline/5_dl_model"
created: "2026-08-07"
updated: "2026-08-10"
status: "active"
---

# DSGNet × WBCIC-SHU 三分类对标锚点

> 论文：Lou et al., *Subject-Independent Deep Learning Framework for Motor Imagery
> Electroencephalogram Decoding in Neurorehabilitation*, IEEE JBHI 2026.
> DOI: [10.1109/jbhi.2026.3689121](https://doi.org/10.1109/jbhi.2026.3689121)
> PDF：`inbox/papers/dsgnet_jbhi2026_FullText.pdf`

## Scope we match

| 项 | 论文 | 本仓 |
|:---|:---|:---|
| 数据 | SHUv5 三分类 11 人（WBCIC-SHU 3C） | `wbci_shu_3c` / `foundation_cross_subject_wbci_3c.yaml` |
| 通道 / 采样 | 58 ch，1000→250 Hz | 官方 derivatives `.mat` → 58×1000@250 Hz（不再二次滤波） |
| 协议 | LOSO；train ses1–2；val ses3；test=留一被试全 3 session | `val_mode=sessions`，同左 |
| 本轮模型 | DSGNet + 七个基线 | **5 个 foundation** + **ATCNet**（论文最强基线，已搬入复现）；DSGNet 复现 deferred |
| run_id | — | `foundation_3c_loso_paper_v1` |

## Paper Table II — SHUv5 (LOSO mean over subjects)

| Method | Acc | F1 | Kappa |
|:---|---:|---:|---:|
| EEGNet | 0.6492 | 0.6488 | 0.4737 |
| EEGNeX | 0.6488 | 0.6469 | 0.4731 |
| EEGInception | 0.5809 | 0.5778 | 0.3713 |
| ATCNet | 0.6834 | 0.6826 | 0.5275 |
| EEG-Deformer | 0.6529 | 0.6503 | 0.4793 |
| MDGEEG | 0.6688 | 0.6634 | 0.5031 |
| EEG-DG | 0.6509 | 0.6488 | 0.4763 |
| **DSGNet** | **0.6856** | **0.6833** | **0.5284** |

## Not matched (document only)

- 论文训练：Adam lr=1e-4，batch 128，500 epochs；本仓 foundation 用各自 recipe（batch 64，
  max 100 + patience 25，monitor macro_f1）。**协议对齐、超参不强行一致。**
  → 这条差距由 **ATCNet 两条 arm** 量化，见下。
- 论文从 raw 做 0.5–40 Hz；我们用官方已切段 derivatives `.mat`（与数据集发布一致）。
- Kappa：**决定不补**（2026-08-09），主对比只用 Acc / macro_f1。

## 怎么读这张表（claim 边界）

三个数字的差要分开归因：

| 差值 | 含义 |
|:---|:---|
| 我们的模型 − ATCNet(arm A) | **模型差距**（完全同条件，唯一可直接下结论的一项） |
| ATCNet(arm B) − ATCNet(arm A) | **训练配方差距**（同模型同划分，只换 recipe） |
| 论文 0.6834 − ATCNet(arm B) | 剩余的实现/预处理差距（移植、derivatives 入口等） |

* arm A = `foundation_3c_loso_paper_v1`（models=[atcnet]），我们的 recipe。
* arm B = `atcnet_3c_loso_paper_recipe_v1`，论文 recipe（Adam 1e-4 / batch 128 / 500 ep /
  patience 100 / 无 scheduler / 无 weight decay / 无梯度裁剪）。

**不要**直接写「我们的模型比 DSGNet 差 X pp」——除非 arm B ≈ arm A（即 recipe 差距可忽略）。

## Results (2026-08-10) — foundation 55/55 + ATCNet A/B 11/11 全 ok

| Method | Acc | F1 | ΔAcc vs DSGNet |
|:---|---:|---:|---:|
| **ATCNet arm A（我们的 recipe）** | **0.7129 ± 0.1322** | **0.7074 ± 0.1417** | **+2.73pp** |
| ATCNet arm B（论文 recipe） | 0.6891 ± 0.1284 | 0.6828 ± 0.1383 | +0.35pp |
| **DSGNet (paper)** | **0.6856** | **0.6833** | — |
| ATCNet (paper) | 0.6834 | 0.6826 | −0.22pp |
| dualcd_s4_flatten | 0.6599 ± 0.1365 | 0.6569 ± 0.1399 | −2.57pp |
| s4erp | 0.6477 ± 0.1388 | 0.6463 ± 0.1410 | −3.79pp |
| dualcd_transformer | 0.6340 ± 0.1323 | 0.6326 ± 0.1347 | −5.16pp |
| dualcd_s4_timepatch | 0.6107 ± 0.1024 | 0.6091 ± 0.1063 | −7.49pp |
| dualcd_s4_pos | 0.4511 ± 0.0666 | 0.4132 ± 0.0757 | −23.45pp |

CSV：`outputs/experiments/wbci_shu/foundation_3c_loso_paper_v1/runs/`；
arm B：`outputs/experiments/wbci_shu/atcnet_3c_loso_paper_recipe_v1/`。
**勿混入** `foundation_3c_loso_v1`（subject-val，非论文协议）。

### Gap 分解（Acc）

| 差值 | 数值 | 含义 |
|:---|---:|:---|
| 最好 foundation（flatten）− ATCNet(A) | **−5.30pp** | **模型差距**（同 recipe） |
| ATCNet(B) − ATCNet(A) | **−2.38pp** | **recipe 差距**（我们的 recipe 更好） |
| 论文 ATCNet − ATCNet(B) | **−0.57pp** | 剩余实现/预处理差距已基本闭合（B 略高于论文） |

结论要点：官方 ATCNet 移植在我们的 recipe 下超过论文 ATCNet/DSGNet；同条件下 5 个 foundation
都低于 ATCNet(A)，主差距是模型；论文 recipe 反而略弱于我们的 recipe；B≈论文 ATCNet，管线可信。
