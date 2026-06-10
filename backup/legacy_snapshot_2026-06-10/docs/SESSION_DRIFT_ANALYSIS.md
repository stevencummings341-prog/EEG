# Session Drift Analysis (Direction A)

Quantify **what changes across sessions** of the same subject in the WBCIC-SHU 2C
MI dataset, to explain why cross-session generalization is hard and to point at the
right adaptation mechanism. This is a CPU-only analysis (numpy/scipy/sklearn; no
torch/GPU). It is the first deliverable of the current cross-session DG mainline.

- Library: `src/analysis/session_drift.py`
- Entry script: `scripts/analysis/run_session_drift.py`
- Slurm (CPU): `scripts/slurm/session_drift_cpu.sbatch`
- Config: `configs/session_drift.yaml`
- Reference origin: `docs/references/senior_scripts/data_validation/session_drift_diagnostic.py`
  (re-implemented + hardened; **do not run the reference copy**).

## Input data

- Entry = the formal `eog_ecg_clean` per-session `.npz` listed in
  `processed_manifest.csv`, **only `status=ok` sessions** (148 ok / 5 failed; the 5
  failed are never loaded). Path comes from `configs/paths.yaml`
  (`processed_data.eog_ecg_clean_root` + `manifests.processed_manifest`).
- Each `.npz`: `X [n_trials, 58, 1000]` float32 (µV @ 250 Hz), `y [n_trials]`.
  Labels are normalized to **{0,1}** (the original reference assumed {1,2}; both work).
- Pairs: for every subject with ≥2 ok sessions, all **within-subject** undirected
  session pairs (i<j). Never compares across subjects (that is a different question).

## Metrics (per session pair)

| metric (CSV column) | category | meaning |
|---|---|---|
| `mmd` | distribution distance | RBF-kernel MMD on flattened trials (median-heuristic γ); higher = more drift |
| `coral` | distribution distance | Frobenius distance of channel-mean covariances; higher = more drift |
| `mu_power_shift` | spectral | log10 µ-band (8–13 Hz) power difference (ses_j − ses_i) |
| `beta_power_shift` | spectral | log10 β-band (13–30 Hz) power difference |
| `mu_ks_stat` | spectral | 2-sample KS statistic on µ-power distributions |
| `erd_mu_corr` | ERD/ERS | Pearson corr of µ ERD/ERS spatial pattern across sessions (1 = identical) |
| `erd_beta_corr` | ERD/ERS | same for β |
| `csp_similarity` | spatial | mean top-\|cosine\| similarity of per-session CSP filters (1 = identical) |
| `rms_ratio_median` / `rms_ratio_std` | amplitude | per-channel RMS ratio (ses_j/ses_i) median & std (1 = no change) |
| `fisher_i` / `fisher_j` / `fisher_shift` | separability | Fisher ratio (between/within class var) on µ log-power per session, and its shift |
| `high_amp_ratio_i/j`, `mean_rms_i/j` | signal quality | high-amplitude (>100 µV) trial fraction, mean RMS per session |

ERD/ERS baseline = first `erd_baseline_ratio` (default 25%) of each trial; MI window
= the rest. CSP uses a dependency-free 2-class implementation.

## Run commands

Always on a compute node (login node = light checks only):

```bash
# smoke test (subjects 1,2) — fast
sbatch scripts/slurm/session_drift_cpu.sbatch --subjects 1,2
# or interactively:
srun -p gpu2node -c 8 --mem 32G -t 00:30:00 bash -lc '
  source /share/software/anaconda3/2024.10/etc/profile.d/conda.sh && conda activate mi_torch &&
  cd /share/home/yuan/SYX/eeg-mi-online &&
  python scripts/analysis/run_session_drift.py --config configs/session_drift.yaml --subjects 1,2'

# full run (all ok subjects)
sbatch scripts/slurm/session_drift_cpu.sbatch

# cap the number of subjects (e.g. a 10-subject sample)
sbatch scripts/slurm/session_drift_cpu.sbatch --max-subjects 10
```

`--subjects` / `--max-subjects` override the `subset:` block in the config.

## Outputs (`outputs/analysis/session_drift_v1/`)

| file | content |
|---|---|
| `session_drift_report.csv` | one row per within-subject session pair (all metrics) — raw, never overwritten |
| `summary.json` | mean/median/std per metric + counts + params/bands |
| `figures/*.png` (8 base) | distribution_distance / band_power_shift / erd_ers_correlation / csp_similarity / fisher_ratio_scatter / rms_ratio / metric_correlation_matrix / session_pair_comparison |

### Report layer (built from the CSV, no recompute)

`scripts/analysis/build_drift_report.py` re-derives a full experiment report and
per-pair / per-subject tables + reporting figures **purely from the CSV** (does not touch
the npz, does not submit jobs). Re-run any time the narrative needs refreshing:

```bash
python scripts/analysis/build_drift_report.py   # light, CPU; safe on the login node
```

It adds (originals untouched):

| file | content |
|---|---|
| `SESSION_DRIFT_REPORT.md` | full experiment report: A purpose · B design (why 144 pairs / 50 subjects) · C metric meanings · D overall results · E per-session-pair (1-2/1-3/2-3) analysis · F per-subject (top-10 high/stable, partial subjects) · G figures · H conclusions |
| `SESSION_DRIFT_SUMMARY_CN.md` | one-page Chinese summary for the advisor |
| `session_pair_summary.csv` | per pair-type aggregate (1-2 / 1-3 / 2-3) |
| `per_subject_drift_summary.csv` / `.md` | per-subject profile + `drift_level` (high/moderate/stable by tertile of a composite drift score) |
| `figures/session_pair_metric_summary.png` | MMD/CSP/ERD-μ/μ-KS boxplots by pair |
| `figures/subject_mmd_heatmap.png` | subject × pair MMD heatmap |
| `figures/subject_csp_heatmap.png` | subject × pair CSP similarity heatmap |
| `figures/subject_erd_mu_heatmap.png` | subject × pair ERD-μ correlation heatmap |
| `figures/high_drift_subjects_bar.png` | top-10 high-drift subjects (MMD/μ-KS/CSP) |
| `figures/signal_quality_shift.png` | high-amp-trial-ratio shift + mean-RMS ratio |

> Note: KS is currently μ-band only (`mu_ks_stat`); β-band KS would need a (cheap) drift
> recompute over the npz and is intentionally not done here.

## Interpreting the result (→ method choice, Direction C / future work)

- High MMD/CORAL + low CSP/ERD-ERS correlation ⇒ spatial pattern drift → Euclidean
  Alignment / CORAL alignment.
- `rms_ratio` far from 1 or large µ/β power shift ⇒ amplitude/spectral drift → BN
  adaptation / re-normalization / filter-bank alignment.
- Positive `fisher_shift` on later sessions ⇒ MI separability improves (the paper's
  learning effect: S1 81.8% → S3 88.9%) → online / test-time adaptation.

## Dependencies

numpy, scipy, scikit-learn, pandas, matplotlib (all in `mi_torch`). **No seaborn**
(the original reference used it; here all figures are matplotlib-only).
