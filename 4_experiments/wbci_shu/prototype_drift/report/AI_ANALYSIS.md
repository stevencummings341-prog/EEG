---
title: "Phase 2c Prototype Drift -- AI Analysis"
tags:
  - "#pipeline/5_dl_model"
  - "#modality/eeg"
  - "#method/domain_generalization"
  - "#paradigm/motor_imagery"
created: "2026-06-11"
updated: "2026-06-11"
status: "active"
---

# Phase 2c -- AI Deep Analysis (Prototype Drift)

> Grounded read of `tables/*.csv` + scripted `prototype_drift_report.md`. No numbers
> are invented; every figure below is recomputed from the 25,920-row metric table
> (4,320 frozen-model cells: 50 subjects x directed pairs x 3 models x 5 seeds).
>
> **target labels are used only for offline diagnostic analysis, not for training or adaptation.**

## 1. Core conclusion (honest)

1. **Prototype drift is a real but only partial driver of the cross-session drop.**
   Every embedding-geometry signal correlates with `acc_drop` in the expected
   direction and is statistically significant ($p \ll 10^{-50}$, $n=4320$), but each
   single metric is only *moderate* (Spearman $\rho \in [0.24, 0.39]$). A standardized
   5-feature linear model explains $R^2 = 0.35$ of the variance in `acc_drop`. So
   embedding-space prototype geometry accounts for roughly **one third** of the drop,
   not all of it.

2. **The mechanism is within-class scatter inflation, not class-centroid collapse.**
   On target, the two class centroids actually move *farther* apart in raw euclidean
   space (source separation $8.45 \to$ target $11.78$), yet within-class scatter grows
   even faster (source $16.86 \to$ target $25.83$, $+53\%$), so the Fisher ratio
   collapses (source $4.58 \to$ target $1.57$, $-66\%$). The cross-session embedding
   does not lose its class means; it gets **fuzzier** — clusters smear out and overlap.

3. **The drop is concentrated in high-drift cells.** Splitting cells into drift
   tertiles: low-drift `acc_drop` $= 0.048$, mid $= 0.141$, high $= 0.155$. Low-drift
   transfers lose almost nothing (~5 pp); the ~11 pp average is driven by the
   mid/high-drift tail. This is the actionable part: drift is a usable *trigger*.

4. **Cosine geometry tracks the drop far better than euclidean.** Euclidean prototype
   distances are scale-inflated and only monotonic (Spearman), not linear (Pearson):
   `prototype_drift_mean` euclidean has Spearman $0.352$ but Pearson $0.067$
   ($R^2=0.005$). The same metrics in cosine space are much more *linear*:
   `separation_change` cosine Pearson $r=0.419$ ($R^2=0.176$) vs euclidean $R^2=0.001$.
   **Any downstream prototype method should operate in cosine / normalized space.**

5. **It is model-dependent.** EEGNet and DeepConvNet show the prototype-drift signature
   clearly (drift $\rho$ 0.56 / 0.49, separation 0.28 / 0.55). FBCNet does **not**
   (drift $\rho=0.28$, negative-margin $0.095$, direction-cosine $-0.036 \approx$ none).
   FBCNet's log-variance band-power embedding fails cross-session for a *different*
   geometric reason, so the prototype story must not be over-generalized.

6. **Verdict on the fork:** prototype drift (especially the Fisher-collapse /
   negative-margin / cosine-separation view) **partially** explains the drop and
   **justifies a prototype-based adaptation pilot**, but the dominant scatter-inflation
   mechanism means a pure centroid-shift correction will leave most of the gap. The
   honest next step is a **scoped Step 4 Oracle diagnostic in cosine space, plus a
   parallel look at scatter / reliability drift** — not an unconditional commitment to
   prototype adaptation.

## 2. Goal

Decide, from the frozen-model diagnostic, (a) whether the ~11 pp cross-session decode
drop is an embedding-space prototype-drift phenomenon, (b) which geometric signal best
predicts it, and (c) whether this evidence is strong and consistent enough to justify
Step 4 prototype-based adaptation, or whether a different drift mechanism should be
investigated first.

## 3. Method

As in the scripted report: train each model on the SOURCE session only (with a
source-train val slice for early stopping), freeze it, extract penultimate-layer
embeddings for source-train / source-val / target-test, build per-class prototypes
(`label_based`, `confidence_weighted`, `correct_only`), and compute six geometry
metrics in both euclidean and cosine distance. Target labels feed only the offline
prototype/metric computation; `used_target_labels_for_training` is `False` and
`n_target_labels_used_for_training` is `0` on every one of the 25,920 rows.

## 4. Protocol

WBCIC-SHU 2C, status=ok only, 50 eligible subjects (47 with 3 ok sessions -> 6 directed
pairs; 3 with 2 ok sessions -> 2 pairs; sub-024 with 1 ok session skipped, consistent
with Phase 1). Seeds 0-4. Phase 1 training recipe (Adam, lr 1e-3, batch 16, max 100
epochs, patience 20, val_fraction 0.2). 4,320 cells, all `ok` in `run_status.csv`.

## 5. Results (recomputed from the tables)

### 5.1 Headline correlations (ALL models, label_based, euclidean, n=4320)

| relationship | Spearman $\rho$ | Pearson $r$ | $R^2$ | verdict |
|:---|---:|---:|---:|:---|
| separation_change vs acc_drop | 0.389 | 0.036 | 0.001 | moderate, monotonic-only |
| fisher_change vs acc_drop | 0.359 | 0.247 | 0.061 | moderate |
| prototype_drift_mean vs acc_drop | 0.352 | 0.067 | 0.005 | moderate, monotonic-only |
| target_negative_margin_rate vs acc_drop | 0.313 | 0.365 | 0.133 | moderate, most linear |
| prototype_direction_cosine vs acc_drop | -0.237 | -0.264 | 0.070 | weak |
| target_margin_mean vs acc_drop | -0.159 | -0.176 | 0.031 | weak |

### 5.2 Euclidean vs cosine (ALL models, label_based)

| relationship | euclidean $R^2$ | cosine $R^2$ |
|:---|---:|---:|
| separation_change | 0.001 | **0.176** |
| target_negative_margin_rate | 0.133 | **0.137** |
| prototype_drift_mean | 0.005 | **0.065** |

Cosine consistently yields stronger *linear* structure; the best single linear
predictor is `separation_change` in cosine space ($R^2=0.176$).

### 5.3 Prototype type (ALL, euclidean)

`label_based` and `confidence_weighted` are nearly identical (e.g. drift $\rho$ 0.352
vs 0.356) — expected, because none of the baselines has a learned confidence head, so
the fallback confidence (max softmax) gives near-uniform weights. `correct_only`
distorts the separation metric (separation_change $\rho$ drops to $-0.013$) because
correct-only source prototypes are well-separated by construction; it should be treated
as a diagnostic of catastrophic cells, not a primary prototype.

### 5.4 Absolute geometry (canonical per-cell means)

| quantity | source | target | change |
|:---|---:|---:|---:|
| class separation (euclidean) | 8.45 | 11.78 | +39% (apart) |
| within-class scatter | 16.86 | 25.83 | **+53%** |
| Fisher ratio | 4.58 | 1.57 | **-66%** |

### 5.5 Drift tertiles, multivariate, directional asymmetry

- `acc_drop` by drift tertile: low 0.048 / mid 0.141 / high 0.155.
- Standardized 5-feature linear model: $R^2 = 0.348$; largest standardized coefficient is
  `target_negative_margin_rate` (0.090), then drift / Fisher / separation (~0.06),
  direction_cosine smallest (0.027).
- Directional asymmetry (mean acc_drop): `ses-03->ses-01` worst (0.150) and
  `ses-03->ses-02` (0.131) — i.e. ses-03 as **source** transfers worst; `ses-02->ses-03`
  best (0.086). This mirrors the Phase 1 directional asymmetry.

### 5.6 By-model robustness

| model | mean acc_drop | drift $\rho$ | direction_cos $\rho$ | neg_margin $\rho$ | separation $\rho$ | fisher $\rho$ |
|:---|---:|---:|---:|---:|---:|---:|
| eegnet | 0.106 | 0.558 | -0.364 | 0.411 | 0.275 | 0.409 |
| deepconvnet | 0.121 | 0.487 | -0.279 | 0.383 | 0.551 | 0.412 |
| fbcnet | 0.116 | 0.282 | -0.036 | 0.095 | 0.344 | 0.293 |

### 5.7 Degenerate cells

130 `correct_only` cells (≈9% of that prototype's cells) have a class with **zero**
correct target predictions — i.e. the model predicts a single class for all target
trials. These are exactly the catastrophic-collapse transfers and are flagged
`degenerate_empty_class` (not silently averaged).

### 5.8 Figures

![[drift_vs_acc_drop.png|640]]
![[separation_change_vs_acc_drop.png|640]]
![[negative_margin_vs_acc_drop.png|640]]
![[fisher_change_vs_acc_drop.png|640]]
![[correlation_summary.png|640]]
![[acc_drop_by_model.png|640]]

## 6. Analysis

**Is the cross-session drop a prototype-drift phenomenon?** Partially. Drift is
monotonically and significantly tied to the drop, and the drop concentrates in
high-drift cells, but a single drift number explains little variance linearly and the
joint model caps at $R^2 \approx 0.35$. Prototype drift is best read as a **reliable
trigger / risk signal**, not a complete causal account.

**What actually breaks the embedding?** The Fisher decomposition is decisive: class
centroids do not collapse (they drift *apart* in raw distance), but within-class scatter
grows by half, so discriminability ($\propto$ Fisher ratio) drops by two thirds. The
failure is **representation diffusion / cluster smearing**, not centroid translation.
This is consistent with Phase 0 (spatial + spectral drift, stable amplitude): a spatial/
spectral remapping scrambles where individual trials land without simply translating the
class means.

**Why direction cosine is weak.** A simple recentre-the-prototype correction relies on
the discriminative *direction* being preserved; direction_cosine is the weakest predictor
($\rho=-0.24$, std coef 0.027). That is bad news for the cheapest prototype fix (shift
target prototypes along a preserved axis) and good news for understanding *why* statistical
alignment failed in Phase 2b — the axis itself rotates and the clusters diffuse, which a
mean/cov alignment cannot repair.

**Why negative-margin rate is the most linear single signal.** It directly counts target
trials that fall on the wrong side of the source nearest-centroid rule ($R^2=0.133$, top
standardized coef). It is essentially a soft error rate, so its tight link to `acc_drop`
is partly mechanical — but it confirms the drop is dominated by trials migrating across
the source decision geometry, which is what scatter inflation produces.

**Anomaly: separation_change sign.** Mean `separation_change` is negative (target
centroids farther apart) yet it correlates *positively* with `acc_drop`. The correlation
is carried by the minority of cells where target separation genuinely shrinks; those are
the worst transfers. Reporting only the mean would hide this — the variance, not the mean,
is informative.

**FBCNet caveat.** FBCNet's embedding is log-variance band-power, not a learned spatial-
temporal feature map; its prototype geometry barely tracks the drop (direction cosine ≈ 0,
neg-margin $\rho=0.095$). Its cross-session failure likely lives in band-power scale/
covariance shift rather than prototype drift, so a prototype-adaptation method tuned on
EEGNet/DeepConvNet should not be assumed to transfer to FBCNet.

## 7. Relationship to previous phases

- **Phase 0** (spatial + spectral drift, stable amplitude) predicts exactly the scatter-
  inflation / axis-rotation picture seen here, not a clean centroid translation.
- **Phase 1** (~10 pp drop, directional asymmetry) is reproduced: mean drop ≈ 0.11, and
  ses-03-as-source is the hardest direction.
- **Phase 2b** (statistical alignment insufficient, high-drift subjects helped least) is
  now mechanistically explained: alignment corrects mean/cov but cannot undo within-class
  scatter growth or axis rotation, which is what dominates here.
- **Phase 2c** therefore advances the story from "alignment doesn't work" to "the failure
  is representation diffusion (Fisher collapse) concentrated in high-drift cells, only
  ~1/3 captured by prototype geometry, and model-dependent."

## 8. Next step (decision)

The evidence is **"qualified go"** for prototype work, with guardrails:

1. **Step 4 Oracle in cosine space (diagnostic upper bound).** Recompute target
   prototypes from target labels (offline) and replace the source nearest-centroid rule;
   measure the achievable recovery. Use cosine / normalized embeddings (5.2 shows that is
   where the geometry is linear). This sets the ceiling for any prototype method *before*
   building few-shot / pseudo-label variants.
2. **Pair it with a scatter / reliability probe.** Because the dominant mechanism is
   scatter inflation (Fisher $-66\%$), test whether per-trial reliability weighting or a
   scatter-aware (e.g. Mahalanobis / shrinkage) prototype beats a plain centroid. If the
   Oracle in cosine space already recovers most of the drop, prototype adaptation is
   justified; if it recovers little, pivot to decision-boundary / scatter-shrinkage or
   reliability-drift methods.
3. **Treat FBCNet separately.** Do not bundle it into a prototype-adaptation claim;
   investigate its band-power covariance shift on its own.
4. **Use drift as an online trigger, not a fix.** The clean drift-tertile separation
   (5 pp vs 15 pp) makes `prototype_drift_mean` (cosine) a good gate for *when* to adapt
   in a later online setting.

This keeps the project honest: Step 4 is justified as a **scoped Oracle diagnostic**, not
as a declared solution.

## 9. File list

- This report: `4_experiments/prototype_drift/report/AI_ANALYSIS.md`.
- Scripted report: `report/prototype_drift_report.md`; run status: `report/RUN_STATUS.md`.
- Tables: `tables/prototype_drift_metrics.csv`, `prototype_accuracy_correlation.csv`,
  `prototype_table.csv`, `trial_embeddings_index.csv`, `run_status.csv`.
- Figures: `figures/*.png` (embedded above).
- Heavy artifacts: `outputs/experiments/prototype_drift_v1/{runs,embeddings}/`,
  `checkpoints/prototype_drift_v1/`.
