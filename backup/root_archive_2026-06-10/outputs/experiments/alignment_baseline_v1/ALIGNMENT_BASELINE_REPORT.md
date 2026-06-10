# Alignment Baseline (Step 2) — Report

> No-learning / unsupervised **test-time alignment** baseline. The model trains ONLY on source session(s); the target session is used ONLY through its UNLABELED X (z-score / covariance / BN running stats / band power); `y_test` is used ONLY for the final evaluation. No `optimizer.step` on the target (only BN running-stat updates).

**Run status: COMPLETE** — total rows 30150, ok 30150, failed 0, NaN-acc among ok 0.

## 0. Headline conclusion (honest)

- **No-learning / unsupervised alignment is INSUFFICIENT.** No method reaches the pre-registered +2% (≥+0.02 mean Δacc) success line.
- `none_reference` (no alignment) mean cross-acc = **0.6818**.
- Best method = **`bn_statistics_adaptation`** with Δacc = **+0.0071** (below the +0.02 line; does NOT clear threshold). Only 1/5 methods are net-positive at all.
- BatchNorm-statistics adaptation gives a **small** positive gain; the covariance (Euclidean/Riemannian) methods slightly **hurt**; filter-bank/z-score are ≈ neutral. This is a **useful negative / diagnostic result**: pure statistic-only alignment cannot close the cross-session gap, which motivates (but does NOT itself perform) learning-based Step-3 adaptation.

## 1. Experiment goal

Test whether **unsupervised statistic-only alignment** (no target labels, no weight learning on target) recovers part of the cross-session accuracy drop measured by the static baseline (single-source ≈ 9–13% drop; multi-source ses-01+02→ses-03 helps).

## 2. Method definitions

- `none_reference` — no alignment; pulled from baseline_v1 cross rows (NOT re-run).
- `session_zscore` — per-channel mean/std normalization; source stats from source train, applied to source train+val; target uses its own unlabeled X stats.
- `euclidean_alignment` — whiten by `R^{-1/2}`, R = arithmetic mean of trial covariances (eigh inverse-sqrt, eps ridge + diagonal shrinkage). Source R from source train; target R from target X.
- `riemannian_alignment` — whiten by `G^{-1/2}`, G = **log-Euclidean** SPD mean `expm(mean_i logm(C_i))` (numpy/scipy only, no pyriemann). Source G from source train; target G from target X.
- `bn_statistics_adaptation` — train on source (early stop on source val), then forward the unlabeled target X to refresh BatchNorm running mean/var only (no loss/backward/optimizer).
- `filterbank_reweighting` — decompose into θ/μ/β/low-γ FIR sub-bands and reweight each band (single scalar gain, clipped) so the target band-power profile matches the **source** profile. Conservative version: per-band scalar gains from target X only.

## 3. Protocols

- single-source directed pairs: ses-i → ses-j (both ok) — 6 directions per 3-ok subject.
- multi-source: ses-01+ses-02 → ses-03 (all three ok).

## 4. No-leakage / no-target-label checks

- `used_target_y_for_training == False` for ALL ok rows: YES.
- `used_target_x_for_stats == True` for all trained-method ok rows: YES (25125/25125).
- n_train range [160, 320]; n_val [40, 80]; n_test [200, 200].
- Code guards: target trials never in train/val (separate session); val carved from source train only; BN method asserts no optimizer (running-stat only); alignment asserts shape unchanged + finite.

## 5. Comparison vs baseline_v1 `none_reference`

| method | mean Δacc vs none |
|---|---:|
| `session_zscore` | -0.0038 |
| `euclidean_alignment` | -0.0124 |
| `riemannian_alignment` | -0.0101 |
| `bn_statistics_adaptation` | +0.0071 |
| `filterbank_reweighting` | -0.0030 |

- **Largest average improvement: `bn_statistics_adaptation` (+0.0071)**.

## 6. Single-source direction results

| direction | `none_reference` | `session_zscore` | `euclidean_alignment` | `riemannian_alignment` | `bn_statistics_adaptation` | `filterbank_reweighting` |
|---|---|---|---|---|---|---|
| ses-01->ses-02 | 0.6698 | 0.6731 | 0.6581 | 0.6588 | 0.6839 | 0.6604 |
| ses-01->ses-03 | 0.6630 | 0.6642 | 0.6486 | 0.6523 | 0.6718 | 0.6609 |
| ses-02->ses-01 | 0.6724 | 0.6562 | 0.6558 | 0.6630 | 0.6701 | 0.6642 |
| ses-02->ses-03 | 0.7062 | 0.7030 | 0.6862 | 0.6859 | 0.7079 | 0.6972 |
| ses-03->ses-01 | 0.6499 | 0.6421 | 0.6378 | 0.6435 | 0.6545 | 0.6518 |
| ses-03->ses-02 | 0.6771 | 0.6807 | 0.6791 | 0.6773 | 0.6959 | 0.6922 |

## 7. Multi-source ses-01+02 → ses-03 results

| method | Acc | BalAcc | MacroF1 | AUC |
|---|---:|---:|---:|---:|
| `none_reference` | 0.7344±0.003 | 0.7344 | 0.7207 | 0.7942 |
| `session_zscore` | 0.7264±0.002 | 0.7264 | 0.7121 | 0.7922 |
| `euclidean_alignment` | 0.7197±0.005 | 0.7197 | 0.7056 | 0.7825 |
| `riemannian_alignment` | 0.7212±0.006 | 0.7212 | 0.7085 | 0.7802 |
| `bn_statistics_adaptation` | 0.7384±0.002 | 0.7384 | 0.7324 | 0.7953 |
| `filterbank_reweighting` | 0.7246±0.004 | 0.7246 | 0.7114 | 0.7800 |

## 8. Which method improves most

- Top method = **`bn_statistics_adaptation`**, mean Δacc = **+0.0071** over none_reference — **below** the +0.02 success line, so it is NOT a sufficient no-learning fix.
- Net-positive methods: 1/5. Covariance whitening (Euclidean/Riemannian) is the worst (slightly negative); z-score and filter-bank are ≈ neutral.

## 9. Which directions improve most

- For the best method `bn_statistics_adaptation`: most-improved direction = **ses-03->ses-02** (+0.0188); least = **ses-02->ses-01** (-0.0023).
- Overall best (method,direction) gain = `bn_statistics_adaptation` on ses-03->ses-02 (+0.0188); worst = `riemannian_alignment` on ses-02->ses-03 (-0.0204).
- Full per-direction means in `alignment_by_direction.csv`; per-direction gains derivable from `results_alignment_all.csv`.

## 10. Which subjects improve / regress

- Under the best method `bn_statistics_adaptation`: **43** subjects improve, **7** regress.
- Most improved: sub-006 (+0.055), sub-048 (+0.037), sub-027 (+0.029).
- Most regressed: sub-035 (-0.011), sub-037 (-0.041), sub-003 (-0.057).
- Full per-subject gains in `alignment_by_subject.csv` and `alignment_gain_by_subject.png`.

## 11. Effect by drift level

| drift_level | `session_zscore` | `euclidean_alignment` | `riemannian_alignment` | `bn_statistics_adaptation` | `filterbank_reweighting` |
|---|---|---|---|---|---|
| stable | -0.0050 | -0.0166 | -0.0142 | +0.0090 | +0.0106 |
| moderate | -0.0010 | -0.0141 | -0.0118 | +0.0076 | +0.0052 |
| high | -0.0051 | -0.0066 | -0.0044 | +0.0049 | -0.0244 |

## 12. Is online / agent adaptation warranted?

- **Yes — warranted, and this run is the evidence for it (but Step-3 is NOT run here).** No no-learning method clears the +0.02 line; the best (BN-stats) gives only a small positive gain, and the covariance methods slightly hurt. Crucially, on **high-drift** subjects the gains are smallest/negative (e.g. filter-bank is strongly negative on high drift), i.e. the subjects that need help most are the least helped by statistic-only alignment.
- Interpretation: closing the residual cross-session gap needs **learning-based** target adaptation (online update / adapter / prototype / memory), not just unsupervised statistics. That is the objective justification for the next stage.

## 13. Next-step suggestions (NOT executed here)

- Use BN-stats adaptation (cheap, mildly positive, never hurts much) as a default front-end, possibly combined with multi-source training.
- Explore learning-based Step-3 adaptation (online test-then-update / lightweight adapter / prototype-memory), focusing on high-drift subjects where no-learning alignment fails.
- Consider an UNLABELED per-subject/per-direction method-selection criterion (no target labels).
- These are suggestions only; no Step-3 / online / fine-tuning / 41-10 / CAP-EEGNet-full run is performed in this report.

## Files

- results_alignment_all: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/alignment_baseline_v1/cross_session/tables/results_alignment_all.csv`
- alignment_by_method: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/alignment_baseline_v1/cross_session/tables/alignment_by_method.csv`
- alignment_by_model: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/alignment_baseline_v1/cross_session/tables/alignment_by_model.csv`
- alignment_by_direction: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/alignment_baseline_v1/cross_session/figures/alignment_by_direction.png`
- alignment_by_protocol: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/alignment_baseline_v1/cross_session/tables/alignment_by_protocol.csv`
- alignment_by_subject: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/alignment_baseline_v1/cross_session/tables/alignment_by_subject.csv`
- alignment_vs_baseline: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/alignment_baseline_v1/cross_session/tables/alignment_vs_baseline.csv`
- alignment_gain_by_drift_level: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/alignment_baseline_v1/cross_session/figures/alignment_gain_by_drift_level.png`
- run_status: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/alignment_baseline_v1/cross_session/tables/run_status.csv`
- alignment_method_comparison: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/alignment_baseline_v1/cross_session/figures/alignment_method_comparison.png`
- alignment_vs_baseline_gain: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/alignment_baseline_v1/cross_session/figures/alignment_vs_baseline_gain.png`
- alignment_gain_by_subject: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/alignment_baseline_v1/cross_session/figures/alignment_gain_by_subject.png`
- alignment_protocol_comparison: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/alignment_baseline_v1/cross_session/figures/alignment_protocol_comparison.png`
