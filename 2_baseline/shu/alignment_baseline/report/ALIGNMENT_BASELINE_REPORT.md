# Alignment Baseline (Step 2) — Report

> No-learning / unsupervised **test-time alignment** baseline. The model trains ONLY on source session(s); the target session is used ONLY through its UNLABELED X (z-score / covariance / BN running stats / band power); `y_test` is used ONLY for the final evaluation. No `optimizer.step` on the target (only BN running-stat updates).

**Run status: COMPLETE** — total rows 45000, ok 45000, failed 0, NaN-acc among ok 0.

## 0. Headline conclusion (honest)

- **No-learning / unsupervised alignment is INSUFFICIENT.** No method reaches the pre-registered +2% (≥+0.02 mean Δacc) success line.
- `none_reference` (no alignment) mean cross-acc = **0.5274**.
- Best method = **`session_zscore`** with Δacc = **+0.0142** (below the +0.02 line; does NOT clear threshold). Only 4/5 methods are net-positive at all.
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
- `used_target_x_for_stats == True` for all trained-method ok rows: YES (37500/37500).
- n_train range [59, 80]; n_val [15, 20]; n_test [74, 100].
- Code guards: target trials never in train/val (separate session); val carved from source train only; BN method asserts no optimizer (running-stat only); alignment asserts shape unchanged + finite.

## 5. Comparison vs baseline_v1 `none_reference`

| method | mean Δacc vs none |
|---|---:|
| `session_zscore` | +0.0142 |
| `euclidean_alignment` | +0.0044 |
| `riemannian_alignment` | +0.0074 |
| `bn_statistics_adaptation` | +0.0056 |
| `filterbank_reweighting` | -0.0147 |

- **Largest average improvement: `session_zscore` (+0.0142)**.

## 6. Single-source direction results

| direction | `none_reference` | `session_zscore` | `euclidean_alignment` | `riemannian_alignment` | `bn_statistics_adaptation` | `filterbank_reweighting` |
|---|---|---|---|---|---|---|
| ses-01->ses-02 | 0.5213 | 0.5439 | 0.5271 | 0.5257 | 0.5268 | 0.5110 |
| ses-01->ses-03 | 0.5280 | 0.5472 | 0.5366 | 0.5355 | 0.5342 | 0.5166 |
| ses-01->ses-04 | 0.5278 | 0.5446 | 0.5351 | 0.5417 | 0.5384 | 0.5177 |
| ses-01->ses-05 | 0.5312 | 0.5342 | 0.5342 | 0.5343 | 0.5403 | 0.5107 |
| ses-02->ses-01 | 0.5233 | 0.5421 | 0.5280 | 0.5303 | 0.5259 | 0.5063 |
| ses-02->ses-03 | 0.5219 | 0.5375 | 0.5239 | 0.5274 | 0.5244 | 0.5029 |
| ses-02->ses-04 | 0.5386 | 0.5644 | 0.5466 | 0.5588 | 0.5481 | 0.5178 |
| ses-02->ses-05 | 0.5295 | 0.5424 | 0.5349 | 0.5340 | 0.5397 | 0.5143 |
| ses-03->ses-01 | 0.5193 | 0.5401 | 0.5282 | 0.5269 | 0.5270 | 0.5141 |
| ses-03->ses-02 | 0.5106 | 0.5366 | 0.5214 | 0.5205 | 0.5143 | 0.5055 |
| ses-03->ses-04 | 0.5073 | 0.5357 | 0.5169 | 0.5164 | 0.5179 | 0.5028 |
| ses-03->ses-05 | 0.5238 | 0.5320 | 0.5226 | 0.5346 | 0.5280 | 0.5077 |
| ses-04->ses-01 | 0.5283 | 0.5449 | 0.5350 | 0.5314 | 0.5254 | 0.5158 |
| ses-04->ses-02 | 0.5478 | 0.5609 | 0.5459 | 0.5489 | 0.5510 | 0.5273 |
| ses-04->ses-03 | 0.5235 | 0.5364 | 0.5333 | 0.5283 | 0.5284 | 0.5063 |
| ses-04->ses-05 | 0.5505 | 0.5604 | 0.5440 | 0.5550 | 0.5532 | 0.5245 |
| ses-05->ses-01 | 0.5211 | 0.5295 | 0.5248 | 0.5294 | 0.5212 | 0.5115 |
| ses-05->ses-02 | 0.5282 | 0.5284 | 0.5234 | 0.5310 | 0.5308 | 0.5143 |
| ses-05->ses-03 | 0.5268 | 0.5304 | 0.5276 | 0.5325 | 0.5311 | 0.5071 |
| ses-05->ses-04 | 0.5391 | 0.5397 | 0.5465 | 0.5534 | 0.5536 | 0.5199 |

## 7. Multi-source ses-01+02 → ses-03 results


## 8. Which method improves most

- Top method = **`session_zscore`**, mean Δacc = **+0.0142** over none_reference — **below** the +0.02 success line, so it is NOT a sufficient no-learning fix.
- Net-positive methods: 4/5. Covariance whitening (Euclidean/Riemannian) is the worst (slightly negative); z-score and filter-bank are ≈ neutral.

## 9. Which directions improve most

- For the best method `session_zscore`: most-improved direction = **ses-03->ses-04** (+0.0284); least = **ses-05->ses-02** (+0.0001).
- Overall best (method,direction) gain = `session_zscore` on ses-03->ses-04 (+0.0284); worst = `filterbank_reweighting` on ses-04->ses-05 (-0.0260).
- Full per-direction means in `alignment_by_direction.csv`; per-direction gains derivable from `results_alignment_all.csv`.

## 10. Which subjects improve / regress

- Under the best method `session_zscore`: **18** subjects improve, **7** regress.
- Most improved: sub-006 (+0.129), sub-002 (+0.062), sub-021 (+0.054).
- Most regressed: sub-013 (-0.003), sub-018 (-0.004), sub-015 (-0.008).
- Full per-subject gains in `alignment_by_subject.csv` and `alignment_gain_by_subject.png`.

## 11. Effect by drift level

| drift_level | `session_zscore` | `euclidean_alignment` | `riemannian_alignment` | `bn_statistics_adaptation` | `filterbank_reweighting` |
|---|---|---|---|---|---|
| stable | +0.0288 | +0.0087 | +0.0122 | +0.0098 | -0.0166 |
| moderate | +0.0106 | +0.0026 | +0.0063 | +0.0027 | -0.0146 |
| high | +0.0044 | +0.0022 | +0.0042 | +0.0044 | -0.0131 |

## 12. Is online / agent adaptation warranted?

- **Yes — warranted, and this run is the evidence for it (but Step-3 is NOT run here).** No no-learning method clears the +0.02 line; the best (BN-stats) gives only a small positive gain, and the covariance methods slightly hurt. Crucially, on **high-drift** subjects the gains are smallest/negative (e.g. filter-bank is strongly negative on high drift), i.e. the subjects that need help most are the least helped by statistic-only alignment.
- Interpretation: closing the residual cross-session gap needs **learning-based** target adaptation (online update / adapter / prototype / memory), not just unsupervised statistics. That is the objective justification for the next stage.

## 13. Next-step suggestions (NOT executed here)

- Use BN-stats adaptation (cheap, mildly positive, never hurts much) as a default front-end, possibly combined with multi-source training.
- Explore learning-based Step-3 adaptation (online test-then-update / lightweight adapter / prototype-memory), focusing on high-drift subjects where no-learning alignment fails.
- Consider an UNLABELED per-subject/per-direction method-selection criterion (no target labels).
- These are suggestions only; no Step-3 / online / fine-tuning / 41-10 / CAP-EEGNet-full run is performed in this report.

## Files

- results_alignment_all: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/shu/alignment_baseline_v1/cross_session/tables/results_alignment_all.csv`
- alignment_by_method: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/shu/alignment_baseline_v1/cross_session/tables/alignment_by_method.csv`
- alignment_by_model: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/shu/alignment_baseline_v1/cross_session/tables/alignment_by_model.csv`
- alignment_by_direction: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/shu/alignment_baseline_v1/cross_session/figures/alignment_by_direction.png`
- alignment_by_protocol: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/shu/alignment_baseline_v1/cross_session/tables/alignment_by_protocol.csv`
- alignment_by_subject: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/shu/alignment_baseline_v1/cross_session/tables/alignment_by_subject.csv`
- alignment_vs_baseline: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/shu/alignment_baseline_v1/cross_session/tables/alignment_vs_baseline.csv`
- alignment_gain_by_drift_level: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/shu/alignment_baseline_v1/cross_session/figures/alignment_gain_by_drift_level.png`
- run_status: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/shu/alignment_baseline_v1/cross_session/tables/run_status.csv`
- alignment_method_comparison: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/shu/alignment_baseline_v1/cross_session/figures/alignment_method_comparison.png`
- alignment_vs_baseline_gain: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/shu/alignment_baseline_v1/cross_session/figures/alignment_vs_baseline_gain.png`
- alignment_gain_by_subject: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/shu/alignment_baseline_v1/cross_session/figures/alignment_gain_by_subject.png`
- alignment_protocol_comparison: `/share/home/yuan/SYX/eeg-mi-online/outputs/experiments/shu/alignment_baseline_v1/cross_session/figures/alignment_protocol_comparison.png`
