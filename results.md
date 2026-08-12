---
title: "Results Quick Reference"
tags:
  - "#pipeline/4_analysis"
  - "#modality/eeg"
created: "2026-06-10"
updated: "2026-08-04"
status: "active"
---

# Results Quick Reference

> 下表为 WBCIC-SHU 结果。SHU 2022 主线 Phase 0/1/2a/2b/2c **已全部 summarize + AI 分析完成（2026-07-06）**，与 WBCIC 在 Phase 0–2c 上齐平。数字均来自各阶段 `tables/`。
>
> **端到端基础模型 × 跨被试（2026-08-04 起的主线）：还没有任何结果。** 代码已融合完成、可运行，
> 但协议参数待学长确认，正式实验一次都没跑（见 `FOUNDATION_E2E_ROUTE_PLAN.md`）。
> 文献对标区间（**别人的数字，不是我们的**）：WBCIC-SHU 零样本跨被试 EEGNet 0.81 / DeepConvNet 0.85
> （EDAPT, J Neural Eng 2026，2-fold 被试划分，59ch/1000Hz 原始数据）；SHU 2022 无可验证的已发表
> 跨被试数字，且 chance band 为 51.4–53.7%。

## SHU 2022（Phase 0–2c 全部完成）

| Stage | Result (SHU 2022) |
|:---|:---|
| Drift diagnostic | done（250 pairs / 25 subj）。空间+μ/β 频谱漂移主导，幅值稳定（RMS median 1.03），可分性未塌。MMD 0.356、CSP_sim 0.344、ERD μ/β 0.527/0.532。比 WBCIC 空间漂移更重（MMD 0.356>0.238、CSP 0.344<0.420）。high 9 / moderate 8 / stable 8。 |
| Baseline | done。within/cross（mean±std, 5 seeds）：EEGNet 0.611/0.538（drop 7.3pp）；DeepConvNet 0.606/0.536（7.0pp）；FBCNet 0.553/0.508（4.5pp）。cross 近 chance（地板效应，掉点 pp 小≠更稳）。排序与 WBCIC 一致。 |
| Multi-source | done。ses-01+02→ses-03：EEGNet 0.544、DeepConvNet 0.558、FBCNet 0.512。相对最强单源：DeepConvNet +2.6pp、EEGNet +0.7pp、FBCNet −0.2pp（方向同 WBCIC，多源≥最强单源）。 |
| Alignment | done。无对齐 0.5274；最佳 `session_zscore` +1.42pp（0.5416，≠WBCIC 的 BN-stats），4/5 净正，`filterbank` −1.47pp 有害。无方法过 +2pp。high-drift 受益最小（z-score stable +2.88 / moderate +1.06 / high +0.44pp）。 |
| Prototype Drift | done（7500 cells 全 ok）。机制同 WBCIC：within-class scatter 膨胀（15.7→38.3, +144%）/ Fisher collapse（1.96→0.79, −60%），非 centroid collapse（separation 6.6→23.9 反增）。最强预测子 fisher_change（ρ=0.43）、separation_change（cosine ρ=0.38/r²=0.16）；drift_mean/direction/margin 弱。cosine>euclidean。EEGNet/DeepConvNet 清晰、FBCNet 弱且几何异常。信号比 WBCIC 更噪（cross 近 chance）。 |

## WBCIC-SHU

| Stage | Result (WBCIC-SHU) |
|:---|:---|
| Drift diagnostic | Spatial + spectral drift dominates; amplitude is stable. |
| Baseline | EEGNet 0.807 within vs 0.711 cross; DeepConvNet 0.766 vs 0.681; FBCNet 0.720 vs 0.628. |
| Multi-source | `ses-01+02 -> ses-03` improves over best single source for all three models. |
| Alignment | No-learning alignment is insufficient; BN-stat is only a small positive gain. |
| Prototype Drift | Phase 2c done (4320 cells, all ok). Drift significant but moderate (separation_change rho=0.389, drift_mean 0.352, neg_margin 0.313); multivariate R2~0.35. Mechanism = within-class scatter inflation (Fisher 4.58->1.57), **not centroid collapse**. Cosine > euclidean. EEGNet/DeepConvNet show it, FBCNet weak. Verdict: **进入 Phase 3（路线 v2）——先做 Phase 3B Oracle 上限裁决（cosine / Mahalanobis-shrinkage / reliability-weighted）再决定是否扩大 T3A**；prototype drift 只解释部分掉点，主机制是 scatter 膨胀，故 Oracle 提前为裁决门。 |
