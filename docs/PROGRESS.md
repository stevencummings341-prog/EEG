# PROGRESS / Memory Log

The project's running memory. **Append a dated entry whenever you finish a
meaningful step or make a design decision.** Newest entries on top. Keep it factual.

Format per entry: date, what was done, decisions made, open questions, next step.

---

> **CURRENT STATUS (2026-06-08, Step 1 multi-source DONE):** The cross-session DG mainline
> now has: **A** session-drift diagnostic (144 pairs / 50 subjects) ✅; **B** static baseline
> EEGNet/DeepConvNet/FBCNet, 5 seeds, within-session 10-fold CV + single-source directed
> cross-session (26 520 trainings, 30/30 cells, no leakage/NaN) ✅; **Step 1 / C** the
> **multi-source `ses-01+ses-02 → ses-03`** static baseline ✅ (47 eligible subjects, 4 skipped,
> 705 rows, no NaN) — multi-source beats the best single source for all 3 models
> (EEGNet 0.7717 vs 0.7492, +0.0224; DeepConvNet 0.7564, +0.0353; FBCNet 0.6750, +0.0267;
> mean Δ +0.0281). Within Acc: EEGNet 0.807 / DeepConvNet 0.766 / FBCNet 0.720; single-source
> cross Acc 0.711 / 0.681 / 0.628 (drop 9–13%). We trail the paper by ~5–8pp in absolute within
> accuracy — a **training-recipe/data-budget** effect (within folds carve a 20% val slice →
> ~144 train trials/fold; dropout 0.25, no max-norm; shared un-tuned recipe), **not an
> architecture bug** (ranking + S1<S2<S3 trend match the paper).
> **Step 2 no-learning adaptation baseline = COMPLETE (2026-06-09).** 30,150 rows, 0 failed/0 NaN.
> Result is a **negative/diagnostic** one: no-learning alignment is INSUFFICIENT — none clears the
> +2pp line. none_reference acc 0.6818; best = bn_statistics_adaptation 0.6889 (Δ +0.0071); EA/RA
> slightly hurt; z-score/filter-bank ≈ neutral. High-drift subjects helped least → motivates (but
> does NOT run) learning-based Step-3 adaptation.
> **NEXT = Step 3 (learning-based) is FUTURE, NOT run.** CAP-EEGNet (v1/v2) + agent/toolkit/
> prototype/confidence/online/fine-tuning, LOSO, 41/10 remain **future work, not run**.
> GPU env `mi_torch_cu118` (torch 2.7.1+cu118).
> Sources of truth: `docs/PROJECT_STATUS_CURRENT.md` (status), `docs/MULTISOURCE_STEP1_REPORT.md`
> (Step 1), `docs/NEXT_EXPERIMENT_PLAN.md` + `docs/ADAPTATION_BASELINE_PLAN.md` (Step 2 plan),
> `docs/RESULTS_SUMMARY.md` (consolidated results), `docs/P10_INTEGRATION_SUMMARY.md` (P10 dir).

<!-- AUTORUN_STATUS_BELOW: baseline_report.py inserts entries here -->

## 2026-06-09 — Step 2 no-learning alignment baseline COMPLETE (negative/diagnostic result)

**Status: COMPLETE.** All 75 GPU training jobs (`21261-21335`) + the summarizer (`21336`) finished
COMPLETED; `results_alignment_all.csv` exists with **30,150 rows** (25,125 alignment + 5,025
none_reference), **0 failed**, **0 NaN/Inf** in metrics (the only all-empty column is the text
`error_message`). `used_target_y_for_training == False` for all rows; `used_target_x_for_stats ==
True` for all 25,125 trained-method rows. 5 trained methods + none_reference all present; models
eegnet/deepconvnet/fbcnet; seeds 0-4; all 6 single-source directions + multi-source ses-01+02→ses-03.

**Result (honest, mean cross-acc over directions/subjects/seeds, vs baseline_v1 `none_reference`):**

| method | Δacc vs none | note |
|---|---:|---|
| `bn_statistics_adaptation` | **+0.0071** | only net-positive method; well below +0.02 |
| `filterbank_reweighting` | −0.0030 | ≈ neutral |
| `session_zscore` | −0.0038 | ≈ neutral |
| `riemannian_alignment` | −0.0101 | slightly hurts |
| `euclidean_alignment` | −0.0124 | hurts most |

- Absolute acc (all scope): none_reference **0.6818**; bn_statistics **0.6889** (best); all others < none.
- By drift level: BN-stats small-positive at every level (stable +0.009 / moderate +0.008 / high
  +0.005); filter-bank positive on stable/moderate (+0.011/+0.005) but **−0.024 on high drift**;
  EA/RA negative everywhere. → the subjects that drift most are helped least.
- **Conclusion: no-learning / unsupervised statistic-only alignment is INSUFFICIENT** (no method
  reaches the pre-registered +2pp success line). BN-stats gives only a small positive gain. This is a
  **valuable negative/diagnostic result** that objectively justifies learning-based Step-3 adaptation
  (online / adapter / prototype / memory) — **which is NOT run here**.
- Verified the `alignment_vs_baseline.csv` join is correct: none_reference is unique per
  (model, seed, subject, train_sessions, test_session, training_scope); all 25,125 alignment rows
  matched; an independent re-merge reproduced the summarizer's Δacc exactly.
- Outputs: `outputs/experiments/alignment_baseline_v1/` — `ALIGNMENT_BASELINE_REPORT.md` (13 sections
  + headline), `RUN_STATUS.md`, `manifest_sources.json`, `cross_session/tables/` (9 CSVs incl.
  results_alignment_all.csv, alignment_vs_baseline.csv, alignment_by_method/model/direction/protocol/
  subject.csv, alignment_gain_by_drift_level.csv, run_status.csv), `cross_session/figures/` (6 PNGs).
  Checkpoints `checkpoints/alignment_baseline_v1/`.
- Closeout only added a richer, honest interpretation to the report (recomputed from the same CSVs);
  no new experiment, no new deps, baseline_v1 untouched.

## 2026-06-09 — Step 2 no-learning alignment baseline: implemented + smoke + FULL RUN SUBMITTED (results PENDING)

**Status (do NOT read as complete):** Step 2 **code implemented**, **smoke passed**, **full run
submitted**, **summarizer dependency submitted**, **results PENDING until the 75 GPU jobs finish.**
Will be marked complete only after the summarizer writes `results_alignment_all.csv`.

**Scope (per user): ONLY Step 2 alignment baseline.** No online, no 41/10, no LOSO, no fine-tuning,
no CAP-EEGNet full, no multi-agent/prototype/memory. No new deps, no shared-env changes, no
raw/workspace2 writes, baseline_v1 not overwritten.

**A — pre-checks (all pass):** baseline_v1 exists with within+cross standard tables; processed
manifest + `eog_ecg_clean` present; env `mi_torch_cu118` = torch 2.7.1+cu118, CUDA available on the
GPU node (RTX 4090 D); git working tree clean (HEAD `a239b43`).

**B — new code (nothing existing overwritten; reuses trainer / registry / session_splits / metrics
/ baseline split logic):**
- `src/adaptation/{__init__,session_alignment,bn_adaptation}.py` — fit/transform alignment
  transforms (ChannelZScore, EuclideanAlignment, RiemannianAlignment = log-Euclidean SPD mean via
  eigh, FilterBankReweight) + BN running-stat adaptation (no grad/backward/optimizer). EA/RA use
  eps ridge + diagonal shrinkage; inverse-sqrt/logm via symmetric eigh with eigenvalue clipping.
- `src/evaluation/session_alignment_protocols.py` — single-source + multi-source tasks; source
  transform fit on SOURCE TRAIN only (applied to source train+val); target aligned from target X
  (filterbank reweights target→source profile); BN method trains plain then refreshes BN from
  target X; per-run leakage asserts; 26-column result rows.
- `scripts/train_session_alignment.py`, `scripts/summarize_alignment_results.py` (pulls
  `none_reference` from baseline_v1 + drift levels), `scripts/build_alignment_baseline_outputs.py`
  (scaffold). `configs/session_alignment_compare.yaml`. Slurm:
  `scripts/slurm/{train_session_alignment_gpu,summarize_alignment_results_cpu}.sbatch`.

**C — unit checks (CPU, synthetic):** zscore finite + ~N(0,1); EA matrix 58×58; RA matrix 58×58 +
spd_mean=log_euclidean; aligned X shape unchanged; EA finite on near-singular cov; BN adapt leaves
all trainable weights UNCHANGED (no optimizer.step) but changes running_mean; eval mode after; all 3
models forward aligned X → logits [B,2]. ALL PASS.

**D — smoke (GPU srun, mi_torch_cu118):** subjects 1,2, eegnet, seed 0, 3 epochs, all 5 trained
methods, both protocol groups → 70/70 rows ok, 0 NaN/Inf, n_train/val/test = 160/40/200 (single) &
320/80/200 (multi), 70 checkpoints, split JSON sessions correct, `used_target_y_for_training`=False
& `used_target_x_for_stats`=True for all rows. Summarizer ran cleanly on the smoke rows. Smoke
artifacts removed afterward.

**E — full run (GPU Slurm):** **75 jobs `21261–21335`** = method × model × seed (each job runs both
single-source + multi-source). Per job = 335 trainings; total ≈ **25,125 trainings**. Partition
gpu2node, gpu:1, env mi_torch_cu118, fail-fast if no CUDA, logs → `logs/slurm/`. Job ids →
`outputs/experiments/alignment_baseline_v1/full_job_ids.txt`. (Some pending under
QOSMaxCpuPerUserLimit; drain as capacity frees.) Estimates + layout in
`outputs/experiments/alignment_baseline_v1/RUN_PLAN.md`.

**F — dependent summarizer:** CPU job **`21336`** with `--dependency=afterany:<all 75>` →
`scripts/summarize_alignment_results.py` → `cross_session/tables/*` (results_alignment_all.csv,
alignment_by_method/model/direction/protocol/subject.csv, alignment_vs_baseline.csv,
alignment_gain_by_drift_level.csv, run_status.csv) + `cross_session/figures/*` (6 figs) +
`ALIGNMENT_BASELINE_REPORT.md` + `RUN_STATUS.md` + `manifest_sources.json`.

**Next (after jobs finish):** check `sacct -j 21261-21336`, read `RUN_STATUS.md`, confirm
`results_alignment_all.csv` exists, then read `ALIGNMENT_BASELINE_REPORT.md`. Only then mark Step 2
complete. Step 3 (online / adapter / prototype) stays gated on these results.

## 2026-06-09 — Systematic documentation repair (no experiments)

**Why**: git HEAD is the 2026-06-04 scaffold commit; all work since (drift, baseline, P10
integration, Step 1) was never committed, so the working tree's docs/code lived only on disk.
During the Step-1 run a tooling/backend hiccup left the working tree missing several docs and
the new Step-1 code files. **Experiment results were never lost** (the 705-row Step-1 run +
the baseline summaries were intact on disk).

**Repaired (no results touched, no experiments run):**
- Re-materialized the Step-1 code that the run had used: `src/evaluation/session_multisource_protocols.py`,
  `scripts/train_session_multisource.py`, `scripts/summarize_multisource_results.py`,
  `configs/session_multisource_compare.yaml`, the two `scripts/slurm/*multisource*` jobs.
  Re-ran the summarizer on the existing 705 rows — reproduced identical numbers.
- Restored the full rich docs from user-provided backups (`/share/home/yuan/SYX/PROGRESS.md`,
  `/share/home/yuan/SYX/PROJECT_OVERVIEW.md`, "saved before the remaining Step 1") + this
  conversation's context, then brought them current (P10 integration + Step 1 done).
- Rewrote the lost docs: `BASELINE_PROTOCOL.md`, `P10_INTEGRATION_SUMMARY.md`,
  `ADAPTATION_BASELINE_PLAN.md`, `CODE_INTEGRATION_NOTES.md`,
  `references/P10_MI_generalization_README.md`; recovered+updated `AGENTS.md`,
  `references/README.md` from git HEAD.
- Wrote a deep Step-1 analysis: `docs/MULTISOURCE_STEP1_REPORT.md` (per-subject gains,
  drift-level breakdown, upper-bound gap recovery) + `multisource_by_subject.csv`.

**Recommendation**: commit the working tree soon so this can't recur (currently everything since
2026-06-04 is uncommitted). Not done automatically — awaiting user go-ahead on git.

## 2026-06-08 — Step 1 static baseline COMPLETE: multi-source ses-01+02 → ses-03

**Scope (per user): only the remaining Step-1 static-baseline item.** No Step-2 adaptation,
no online, no 41/10, no LOSO, no fine-tuning, no CAP-EEGNet full / multi-agent / prototype /
memory. No raw/workspace2 writes. No existing baseline results overwritten.

**Step A — checked for existing results**: none found under `outputs/experiments/` for
`multisource` / `0102_to_03`. Confirmed in code that only **single-source directed pairs**
existed (`session_splits.make_cross_session_pairs`, `session_protocols.run_cross_session`).
So the multi-source direction was genuinely missing → implemented it.

**Step B — new code (separate from existing baselines, nothing overwritten):**
- `src/evaluation/session_multisource_protocols.py` — eligibility (all of ses-01/02/03 ok),
  combine ses-01+ses-02 trials, carve val ONLY from the combined train, test = ses-03;
  per-run leakage assertions; `n_train/n_val/n_test` + checkpoint + status recorded.
- `scripts/train_session_multisource.py` — entry (reuses TrainSpec / trainer / registry /
  `load_ok_sessions`); writes per-seed CSV + meta JSON (used/skipped subjects).
- `scripts/summarize_multisource_results.py` — aggregates, builds the report + figure, and
  pulls the single-source ses-01→03 / ses-02→03 rows from the baseline summaries for comparison.
- `configs/session_multisource_compare.yaml`, `scripts/slurm/train_session_multisource_gpu.sbatch`,
  `scripts/slurm/summarize_multisource_results_cpu.sbatch`.

**Step C — smoke test (GPU node via srun, `mi_torch_cu118`):** subjects 1,2, EEGNet, seed 0,
3 epochs. Passed: train_sessions=ses-01+ses-02, test=ses-03, `n_train=320 / n_val=80 / n_test=200`,
train/val disjoint, val carved only from the combined train, no NaN, CSV + checkpoint + split JSON
written. Smoke output dir removed afterward.

**Step D — full run (GPU Slurm):** jobs `21240,21241,21242,21243,21244` (one per seed,
3 models each) + summarize `21245` (afterany). All **COMPLETED**, exit `0:0`. 705 rows
(3 models × 5 seeds × 47 used subjects), 0 failed, 0 NaN.

**Step E — results (mean±std over 5 seeds, test = ses-03):**
- EEGNet **0.7717±0.003** > DeepConvNet 0.7564±0.007 > FBCNet 0.6750±0.002.
- vs best single direction (`ses-02→03`): +0.0224 / +0.0353 / +0.0267 (mean **+0.0281**).
- vs per-subject best single (oracle source pick): +0.0139 / +0.0209 / +0.0097 (still positive).
- recovers ~30–53% of the single-source-cross → within-ses-03 upper-bound gap.
- per-subject: EEGNet improved 33/47; failures concentrate where the two source sessions differ
  a lot (e.g. sub-029 ses-01 0.441 vs ses-02 0.935) → motivates Step-2 alignment/weighting.
- Full analysis: `docs/MULTISOURCE_STEP1_REPORT.md`; machine report + CSVs under
  `outputs/experiments/baseline_v1/provenance/session_multisource_v1/summaries/`; checkpoints
  `checkpoints/session_multisource_v1/{eegnet,deepconvnet,fbcnet}/`.

**Note (doc loss + recovery):** during this work the (uncommitted) working tree lost several
docs and the new Step-1 code files (git HEAD is the 2026-06-04 scaffold; later work was never
committed). The experiment **results were intact**; the Step-1 code was re-materialized from the
run, the summarizer re-ran cleanly on the existing 705 rows, and the docs were restored from
backups + this conversation's context. See the 2026-06-09 doc-repair entry.

**Next**: Step 2 no-learning adaptation baseline (see `docs/ADAPTATION_BASELINE_PLAN.md`). Do not
run online / 41-10 / fine-tuning / CAP-EEGNet full until Step 2 is done.

## 2026-06-08 — P10 「MI 泛化研究」 integration + direction cleanup (NO experiments run)

Integrated the senior's updated P10 package (`/share/home/yuan/SYX/P10_MI泛化研究/`) into the
project's understanding/docs. P10 Phase-0 (drift) + Phase-1 (baseline) numbers **match this
project exactly** (same 148 ok sessions, same EEGNet 0.807/0.711) — same study, packaged for
handoff. P10 adds richer per-subject drift profiling (drift_level high/moderate/stable; sub-020
most drifted, sub-005 most stable), the cross-session **directional asymmetry** (ses-02→03 0.749
best vs ses-03→01 0.681 worst, 6.8pp), MMD↔accuracy Spearman ρ=−1.0 (n=3), and a **Phase-2 online
continual-learning framework that is design + draft code only, NOT validated** → future work.

New docs: `docs/PROJECT_STATUS_CURRENT.md`, `docs/P10_INTEGRATION_SUMMARY.md`,
`docs/NEXT_EXPERIMENT_PLAN.md` (Step 0/1/2/3), `docs/ADAPTATION_BASELINE_PLAN.md`,
`docs/CODE_INTEGRATION_NOTES.md`, `docs/references/P10_MI_generalization_README.md`. Updated
wording (done=done, unrun=future) across the core docs. **No experiment run; no raw/workspace2 writes.**

<!-- AUTORUN_STATUS_BELOW: baseline_report.py inserts entries here -->

## 2026-06-07 — Baseline 5-seed run COMPLETE + repo cleanup

**All 10 training jobs + the report job COMPLETED** (21161–21171, exit 0). Final summarizer
re-run after cleanup: **26 520 rows (within 22 200 + cross 4320), 30/30 cells,
`incomplete=False`, no NaN, no leakage**. (The earlier auto `RUN_STATUS.md`/log entry said
"INCOMPLETE" only because the report job counted *itself* (21171) as RUNNING — a false flag;
the summarizer logged `incomplete=False`.)

**Headline results** (mean±std over 5 seeds):
- within Acc: EEGNet **0.807±0.002** > DeepConvNet 0.766±0.002 > FBCNet 0.720±0.003.
- cross Acc: EEGNet **0.711±0.008** > DeepConvNet 0.681±0.002 > FBCNet 0.628±0.003.
- cross-session drop: 11.9% / 11.1% / 12.8%.
- vs paper within (85.32 / 84.47 / 78.40): we are −4.65 / −7.84 / −6.37 pp.

**Why below paper (see `docs/RESULTS_SUMMARY.md` §4)**: systematic training-recipe/data-budget
difference, NOT an architecture bug — chiefly the within-fold 20% val carve (≈144 vs 180 train
trials), early stopping on a tiny (~36) val set, dropout 0.25 + no max-norm + wd 0, and a shared
un-tuned recipe (FBCNet also simplified). Gap is consistent across all 3 models and the ranking
+ S1<S2<S3 trend match the paper; the within-vs-cross conclusion is fair and unaffected.

**New doc**: `docs/RESULTS_SUMMARY.md` (drift + baseline + paper-gap analysis + next steps).

**Repo cleanup (gitignored artifacts only; no source/results lost):** removed the archived
smoke output dir, legacy `outputs/processed_paper_style/sub-001`, `outputs/raw_check`,
`outputs/sanity_check`, and the stale smoke-only `checkpoints/.../cap_eegnet/` (18 files,
sub-001/002 seed0). Standardized run CSV naming: legacy `within__{model}.csv` →
`within__{model}__seed0.csv` (now all 30 files are uniform `{protocol}__{model}__seed{N}.csv`).
The 3 baseline checkpoint dirs (eegnet/deepconvnet/fbcnet, 2180 each, ~2.7G total) are kept as
legitimate results.


## 2026-06-07 — Overnight 5-seed baseline GRID submitted (EEGNet/DeepConvNet/FBCNet)

**Goal**: full 5-seed baseline over the 148 `status=ok` sessions, both protocols, seeds
0–4, three baseline models. CAP-EEGNet and all complex modules remain FUTURE (not run);
no LOSO, no 41/10.

**Overwrite fix**: result CSVs are now written per seed —
`runs/{protocol}__{model}__seed{seed}.csv` (was `{protocol}__{model}.csv`, which would have
let seeds overwrite each other). Job 21161 (already running, old code) keeps the legacy
`within__{model}.csv` (=seed 0); the summarizer globs all CSVs and reads the `seed` column,
so there is no collision. Checkpoints were already seed-namespaced.

**Jobs (1 GPU each, env `mi_torch_cu118`, fail-fast if no CUDA):**
- `21161` within seed 0 (already running) — `mi_sess_compare`
- `21162` cross s0, `21163` within s1, `21164` cross s1, `21165` within s2, `21166` cross s2,
  `21167` within s3, `21168` cross s3, `21169` within s4, `21170` cross s4
  (job names `mi_base_{protocol}_s{seed}`, logs `logs/slurm/mi_base_{protocol}_s{seed}-<jobid>.out/.err`)
- job-id list: `outputs/experiments/baseline_v1/provenance/session_model_compare_v1/overnight_job_ids.txt`

**Expected trainings** (1 model × 1 protocol × 1 seed): within = 148 sessions × 10 folds =
1480; cross = 288 directed pairs. Per seed (3 models): within 4440, cross 864. 5-seed total:
within 22 200 + cross 4320 = **26 520 trainings**.

**Dependent report job** (`mi_base_report`, CPU, `--dependency=afterany:` all 10 IDs): runs
`scripts/baseline_report.py` → sacct check + `scripts/summarize_session_results.py` →
`outputs/experiments/baseline_v1/provenance/session_model_compare_v1/summaries/` (results_within/cross_session.csv,
within_by_seed/cross_by_seed/within_session_wise/cross_by_direction.csv,
summary_by_model_protocol.csv, model_ranking.md, SESSION_MODEL_COMPARE_REPORT.md, RUN_STATUS.md,
3 figures) and appends auto status entries here + to `docs/EXPERIMENT_LOG.md`. Marks the
report INCOMPLETE if any training job is not COMPLETED or any output is missing.

Outputs: `outputs/experiments/baseline_v1/provenance/session_model_compare_v1/`; checkpoints
`checkpoints/session_model_compare_v1/<model>/{within_<subj>_<ses>_seed<k>.pt,
cross_<subj>_<tr>-to-<te>_seed<k>.pt}`; slurm logs `logs/slurm/`.

## 2026-06-07 — Mainline converged to the three baseline architectures (EEGNet/DeepConvNet/FBCNet)

**Decision (per user)**: narrow the active mainline to **the three baseline models** — run
EEGNet + DeepConvNet + FBCNet within-session 10-fold CV then cross-session on the 148
`status=ok` sessions; start by checking EEGNet against the WBCIC-SHU paper baseline.
**CAP-EEGNet (v1/v2) and ALL complex modules (agent/toolkit/neural-subagents, prototype,
multi-source confidence, online, fine-tuning) are kept in code but marked OPTIONAL /
FUTURE — not run.** Also not run: LOSO, 41/10.

**No code deleted.** Changes are config + docs only (the trainer/protocols stay
model-agnostic, so CAP-EEGNet still works when re-enabled):
- `configs/session_model_compare.yaml`: `models: ["eegnet","deepconvnet","fbcnet"]`
  (cap_eegnet commented as optional/future).
- `scripts/slurm/train_session_models_gpu.sbatch`: default `--models eegnet,deepconvnet,fbcnet`.
- Docs updated: `PROGRESS.md` (this entry + top status), `EXPERIMENT_PROTOCOL.md`
  (mainline = 3 baselines; CAP-EEGNet + experiments = future), `BASELINE_PROTOCOL.md`
  (top banner + run commands narrowed to the 3 baselines).

**Run plan (seed=0)**: (1) full within-session CV for the 3 baselines → (2) summarizer →
check EEGNet vs paper → (3) full cross-session for the 3 baselines → summarizer.
Submitting the full within-session CV now (see next entry / experiment log for results).

## 2026-06-06 — Mainline pivot: cross-session DG (drift diagnostics + within/cross model comparison)

**Why**: per the senior's P10 "MI 泛化研究" package (now in `docs/references/P10_MI_generalization/`),
the project pivots from 41/10 cross-subject pretraining to a **cross-session domain
generalization** study: (A) diagnose what drifts across sessions, (B) compare baselines +
our model under one fair protocol, (C) at Within-session CV and Cross-session levels, (D)
unified report. **No 41/10, LOSO, fine-tuning, or online now** — all kept as future work.

**Task 0 — references placed** (renamed, reference-only, not run from there):
`docs/references/P10_MI_generalization/` (HANDOFF/proposal/experiment_log/QC_SUMMARY_CN/
ChatGPT task md/paper PDF/slides) and `docs/references/senior_scripts/{data_validation,
model_training}/` (the originals). `docs/references/README.md` indexes them + maps each to
its in-tree implementation.

**New configs**: `configs/session_drift.yaml`, `configs/session_model_compare.yaml` (paths
from `configs/paths.yaml`; status=ok filter; within folds=10; protocols within+cross;
models eegnet/deepconvnet/fbcnet/cap_eegnet; seeds=[0]; bs16; lr1e-3; max_epochs100;
early-stopping).

**New code**:
- `src/data/session_splits.py` — `load_ok_sessions` (manifest, status=ok), label
  normalization to {0,1} (accepts {1,2}), `make_within_session_folds` (StratifiedKFold,
  label-balanced), `make_cross_session_pairs` (directed, both-ok), JSON persistence.
- `src/analysis/session_drift.py` + `scripts/analysis/run_session_drift.py` +
  `scripts/slurm/session_drift_cpu.sbatch` — MMD/CORAL/μ-β power shift/KS/ERD-ERS spatial
  corr/CSP similarity/RMS ratio/Fisher shift; vectorized FFT; **matplotlib-only** (no
  seaborn); status=ok via manifest; `--subjects`/`--max-subjects`. Outputs CSV + summary.json
  + SESSION_DRIFT_REPORT.md + 8 figures.
- `src/models/`: added `EEGNetClassifier` (eegnet.py), `deepconvnet.py` (Schirrmeister
  2017), `fbcnet.py` (Mane 2021, fixed FIR filter bank), `registry.py` (`build_model`).
  **CAP-EEGNet upgraded to v1**: encoder + classifier + lightweight **learned** confidence
  head (single-source scalar, calibration BCE); prototype/subagents/adapter/domain/online
  still raise `NotImplementedError`. All 4 share `forward -> {logits, features, confidence}`.
- `src/training/trainer.py` (CE + optional confidence-BCE, early stopping, predict),
  `src/evaluation/session_protocols.py` (`run_within_session` StratifiedKFold; 
  `run_cross_session` directed pairs; metrics acc/bacc/f1/auc/nll/brier/ece; val carved
  from train only; bounded checkpointing), `scripts/train_session_models.py` +
  `scripts/slurm/train_session_models_gpu.sbatch` (env `mi_torch_cu118`, cuda fail-fast).
- `scripts/summarize_session_results.py` + `..._cpu.sbatch` — merges `runs/*.csv` →
  results_within/cross CSVs, summary_by_model_protocol.csv, 3 figures, model_ranking.md,
  SESSION_MODEL_COMPARE_REPORT.md, cross-session drop + relative drop.

**Docs**: new `SESSION_DRIFT_ANALYSIS.md`, `BASELINE_PROTOCOL.md`; updated
`PROJECT_OVERVIEW.md` (top status), `EXPERIMENT_PROTOCOL.md` (mainline = within/cross;
41/10/LOSO/finetune/online = FUTURE), `.cursor/rules/30-model-experiments.mdc` (shared
protocol/filter/metrics + no-leakage), `AGENTS.md` (constraint 9).

**SMOKE TESTS — ALL PASS (compute nodes via srun, never login node):**
- GPU env check: `mi_torch_cu118` = torch **2.7.1+cu118**, `cuda.is_available()=True`,
  RTX 4090 D. (The earlier `mi_torch` CPU-only-torch blocker is resolved by this env.)
- Drift (subjects 1,2): 6 within-subject pairs in 12.6 s; CSV + summary.json + report +
  8 figures written to `outputs/analysis/session_drift_v1/`. MMD≈0.175, ERD/ERS corr≈0.44–0.49,
  RMS ratio≈0.90 — sane.
- Within (subjects 1,2; folds=2; epochs=3; all 4 models): 48 rows, 117.8 s, all CSVs +
  per-session split JSONs written.
- Cross (subjects 1,2; epochs=3; all 4 models): 48 rows (2 subj × 6 directed pairs × 4
  models), 37.7 s; n_train=160/n_val=40 (carved from train) / n_test=200 (full test session)
  → no leakage verified.
- Summarize: merged 96 rows → all 8 summary artifacts. (Numbers are smoke-noise at 3
  epochs/2 folds; pipeline correctness is the point.)

**Full-run estimates (1 seed; pending user go-ahead — NOT yet submitted):**
- Drift full ≈ **144** within-subject pairs (47 subjects×3 + 3 from partial-ok subjects;
  sub-024 has 1 ok session → 0 pairs), ~10–15 min CPU.
- Within full = 148 ok sessions × 10 folds × 4 models = **5920** trainings; ~**4–8 h wall**
  if the 4 models run as parallel 1-GPU jobs (deepconvnet is the long pole), less with
  early stopping.
- Cross full = **288** directed pairs × 4 models = **1152** trainings; ~**1–2 h wall**
  parallel. Seeds multiply linearly.
- Outputs: `outputs/analysis/session_drift_v1/`, `outputs/experiments/baseline_v1/provenance/session_model_compare_v1/`
  (`runs/`, `splits/`, `summaries/`); checkpoints `checkpoints/session_model_compare_v1/`;
  slurm logs `logs/slurm/`.

**Deliberately NOT done**: full-scale GPU submission (awaiting user OK per task 8.6); 41/10,
LOSO, fine-tuning, online; full multi-subagent CAP-EEGNet (v1 only). No raw/workspace2 writes.

## 2026-06-05 — Roadmap alignment (project direction = full multi-subagent CAP-EEGNet)

**Why**: ensure the project targets the chat record's FINAL design, not a plain EEGNet
classifier. No training, no GPU install, no sbatch — docs/rules/light-code alignment only.

**Final goal restated everywhere**: **Confidence-aware Online Adaptive Multi-Subagent
Pretraining Framework for Cross-subject MI EEG Decoding**. The minimal EEGNet+classifier is
explicitly a **Stage 0 pipeline-validation baseline**, NOT the paper method.

**Staged route fixed (docs/ROADMAP.md)**: Stage 0 infra/sanity (✅) → Stage 1 cross-subject
pretrain + zero-shot (5 seeds, one model each, mean±std) → Stage 2 confidence+prototype+
adapter(+domain, +neural subagents) → Stage 3 Session-1 fine-tune → Stage 4 online
test-then-update → Stage 5 ablation/interpretability.

**New docs**: `docs/ROADMAP.md` (final goal, full module table, minimal-vs-full, Stage 0–5,
do-not list) and `docs/ALIGNMENT_CHECKLIST.md` (chat-record→status mapping).

**Docs updated**: `PROJECT_OVERVIEW.md` (top status, §0 final-goal, completion table now
shows splits ✅ + minimal CAP-EEGNet ✅ + full CAP-EEGNet 🚧, next-steps), `MODEL_PLAN.md`
(minimal-vs-full, deep neural subagents not handcrafted, full component list, Stage-ordered
build order), `EXPERIMENT_PROTOCOL.md` (5-seed mean±std, Exp1–4 input/output/forbidden).

**Rules updated**: `00-project-context` (final goal + refreshed current-stage block),
`30-model-experiments` (minimal≠final; repeated multi-seed split mandatory; no target
leakage; stage note), `40-online-learning` (test-then-update; default-forbid full-backbone
online update; confidence-gate/prototype/adapter are core).

**Light code alignment** (`src/models/cap_eegnet.py`): kept the working minimal model;
added full-vision config flags (`use_subagents/use_dataset_router/use_adapter/use_prototype/
use_confidence/use_domain_align/use_online_update`), fail-fast `NotImplementedError`
("Reserved for full CAP-EEGNet … NOT implemented in the minimal sanity model") when any is
enabled, `predict_confidence()`/`online_update()` reserved methods, and documented stub
classes (`NeuralSubagentEncoder/DatasetAwareRouter/Adapter/PrototypeMemory/ConfidenceHead/
DomainAlignmentHead/OnlineUpdateModule`). Verified: minimal forward still returns
`{logits[B,2], features, proto_dist=None, confidence=None}`; enabling a full flag raises the
clear error. Also aligned `configs/finetune.yaml` + `configs/online_adaptation.yaml`
`variant→eog_ecg_clean` + `statuses:[ok]` for consistency with `train_cross_subject.yaml`.

**Unchanged on purpose**: preprocessing, splits, dataset, sanity outputs (no refactor). No
formal training, no GPU env changes, no sbatch, derivatives `.mat` never a training entry.

## 2026-06-05 — Pre-training prep: 41/10 splits + minimal CAP-EEGNet + sanity train

**Scope (per user): training-prep only, NOT formal training.** All heavy/GPU steps ran on
a compute node via `srun` (node gpu02), never the login node.

**1. `src/data/splits.py` (implemented).** Reads `processed_manifest.csv`, uses only
`status==ok` sessions. `SubjectSessionIndex` summarizes subject→session→status. Policy:
target subjects must have **all 3 sessions ok** (needed for Session-1 fine-tune +
Session-2/3 online); source = the other 41 subjects (may contain subjects with failed
sessions, but training uses ok sessions only). Subject-wise only — never session/trial.
Failed sessions → `excluded_sessions`. `make_subject_wise_split(seed=...)` uses a local
`random.Random(seed)` over the **47 fully-ok** subjects to pick 10 targets; the 4 partial
subjects (sub-023/024/026/032, owning the 5 failed sessions) are always forced into source.

**2. Splits generated** via `scripts/make_splits.py` for seeds **2026–2030** →
`splits/cap_eegnet_4110_seed<k>.json`. Each: source=41 (118 ok sessions, 23 600 trials,
11 800/11 800), target=10 (30 ok sessions, 6 000 trials, 3 000/3 000), excluded=5
(`sub-023/ses-01, sub-024/ses-02, sub-024/ses-03, sub-026/ses-01, sub-032/ses-02`). Each
JSON records source_subjects, target_subjects, excluded_sessions(+detail), source/target ok
sessions, `target_finetune_sessions=[1]`, `target_online_sessions=[2,3]`. Target sets per
seed (e.g. 2026: 007/008/015/021/030/037/040/043/046/049).

**3. Split smoke test** `tests/test_splits.py` (manifest-only, login-safe): asserts
source=41/target=10, no source∩target overlap, target subjects have ses-01/02/03 all ok,
all used sessions are ok, excluded are truly non-ok. **All 5 splits PASS**; prints per-split
trial/session counts + balanced 0/1 label dist.

**4. Minimal CAP-EEGNet** — `src/models/eegnet.py` now has a real `EEGNetEncoder`/`EEGNet`
(Lawhern 2018, F1=8/D=2/F2=16/K=64; depthwise spatial over 58 ch, separable temporal;
feature_dim inferred = 496). `src/models/cap_eegnet.py` `CAPEEGNet` (nn.Module) = encoder +
linear cls head; `forward(x)` accepts `[B,58,1000]` or `[B,1,58,1000]`, returns
`{logits[B,2], features[B,496], proto_dist=None, confidence=None}`. Adapter / Prototype /
Confidence kept as **TODO stubs** (use_*=False; enabling raises NotImplementedError). ~3 026
params. (`padding='same'` even-kernel UserWarning is benign.)

**5. SHUTrialDataset smoke test** `tests/test_dataset_smoke.py` (compute node): builds
source/target datasets from `processed_manifest.csv` + split JSON; asserts entry is `.npz`
only (no `derivatives/.mat`), item x=`[58,1000]` float32, DataLoader batch x=`[B,58,1000]`,
y dtype `torch.long` (int64), labels ⊆ {0,1}; feeds a batch through CAP-EEGNet → logits
`[B,2]` (internal 4D `[B,1,58,1000]`). **PASS** for both roles.

**6. CUDA check (srun, --gres=gpu:1 on gpu02)**: `nvidia-smi` shows an RTX 4090 D (24 GB,
driver CUDA 12.8), but **torch 2.6.0 is CPU-only**: `torch.version.cuda=None`,
`torch.cuda.is_available()=False`, `device_count=0`. **Fix plan (do NOT mutate shared
`mi_torch`)**: create a dedicated CUDA env, e.g. inside a GPU `srun`:
`conda create -n mi_torch_cu118 --clone mi_torch` then
`pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio`
(cu121/cu124 also compatible with the 12.8 driver), and point GPU jobs at it. Awaiting user
go-ahead before installing.

**7. Sanity training** `scripts/sanity_train.py` (+ `scripts/slurm/sanity_train_gpu.sbatch`):
3 source subjects (sub-001/002/003), 9 npz, 1 800 trials, 3 epochs, batch 128, Adam lr 1e-3.
Ran on CPU (no CUDA). **loss 0.7002 → 0.6474 → 0.6014 (decreased=True)**, acc 0.541 → 0.630 →
0.682, no shape errors, ~38 s. Output: `outputs/sanity_check/sanity_check_metrics.json`.
Device auto-selects cuda if available else cpu, so it will use GPU once cu-torch is installed.

**Deliberately NOT done**: formal cross-subject training loop, fine-tuning, online loops,
prototype/confidence/adapter heads, GPU torch install. Awaiting user go-ahead.

## 2026-06-05 — Full preprocessing QC + quality comparison vs official derivatives

**What ran**: full `eog_ecg_clean` preprocessing was already complete (153 sessions, written
to the external `processed/eog_ecg_clean/` tree). This step did **QC + a quality comparison
against the paper `derivatives/2C dataset_processeddata` .mat** (NO training, NO 41/10, NO
CAP-EEGNet). New code:
- `src/evaluation/data_quality.py` — reusable, numpy/scipy-only metrics: amplitude, Welch
  PSD + bandpowers (δ/θ/μ/β/low-γ/50 Hz), per-channel RMS/std, paired similarity
  (trial-wise + per-channel Pearson, MAE/RMSE/relRMSE), MI class separability (Cohen's d /
  Fisher on log-bandpower at C3/C4), official `.mat` locator (handles the one non-`eeg/`
  layout) + loader (data[58,1000,200]→[200,58,1000], labels 1/2→0/1), and
  `best_session_assignment()` (per-subject session matching).
- `src/visualization/quality_plots.py` (NEW package) — 13 QC/compare figures (Agg backend).
- `scripts/compare_processed_quality.py` — orchestrator: manifest QC (19 checks) →
  per-subject session-aligned comparison → CSV/JSON/MD + figures.
- `scripts/slurm/compare_quality_cpu.sbatch` — reusable CPU job.
Ran on a compute node via `srun` (~6 min, `-c 8 --mem 32G`). Outputs in
`processed/eog_ecg_clean/qc_vs_derivatives/`: `manifest_qc_summary.json`,
`session_quality_metrics.csv` (153), `paired_similarity_metrics.csv` (144),
`session_alignment.json`, `QC_REPORT.md`, `figures/*.png` (13).

**Manifest QC (153)**: status **148 ok / 5 failed**. Failures = trigger/试次 < 200
(sub-023/ses-01=199, sub-024/ses-02=199, sub-024/ses-03=195, sub-026/ses-01=199,
sub-032/ses-02=199) → shape/label/trigger checks fail for exactly those 5. The other 148:
shape [200,58,1000], y[200], 100/100, 250 Hz, 58 ch, 1000 times, **0 NaN/Inf**. aux cleaning
used 153/153; EOG ICs removed in 106 sessions (189 comps), ECG in 43 (74); ICA n_components
retry fired 10×; **no-aux-clean fallback 0×**. Total 6.15 GiB.

**KEY FINDING — official derivatives session ordering is permuted.** For **22/51 subjects**
the paper `.mat` stores sessions in a different order than BIDS sourcedata (our ses-YY).
Naive same-ses comparison produced absurd std ratios (e.g. sub-030 ses-01 264×, sub-018
ses-01 14×, with reciprocal <0.1 in another session of the same subject). We match each of
our sessions to its true official counterpart within a subject via a (std, max|·|) amplitude
fingerprint (robust to ICA), accepting a permutation only when it beats identity by a margin.
Cross-validated: after matching, those subjects' labels become **exact** and trial-corr jumps
(e.g. sub-001/ses-02 was "102/200 order-diff" under same-ses → it's actually our-ses-02 ↔
official-ses-03, exact + corr 0.92). This reinterprets the earlier "trial-order differences"
as session swaps. **Our (X,y) are correct** (sourcedata + evt.bdf; per-subject session
amplitude *set* matches official). Permuted subjects: 001,004,006,007,008,010,013,017,018,
023,025,027,030,036,038,039,042,044,045,047,049,051.

**Comparison verdict (after alignment) = PASS**:
- std ratio (ours/official) median **0.976** (111/153 ours lower → artifact removal), RMS
  ratio 0.976; μ/α bandpower ratio **0.898**, β ratio **0.941**.
- paired (144 exact-label sessions) trial-wise Pearson median **0.954**, rel-RMSE 0.251.
- PSD overlays (global + C3/C4/Cz) overlap through μ/β; only 40–50 Hz differs (MNE firwin +
  50 Hz notch vs EEGLAB rolloff — above MI bands, expected).
- MI class separability (|Cohen's d| on log-bandpower) ours≈official at C3/C4 μ/β
  (e.g. C3 μ 0.138 vs 0.150; C4 β 0.142 vs 0.144) → discriminative MI info preserved.
- 22 "attention" sessions = the 5 failed + a few heavy-clean low-ratio sessions (std ratio
  0.35–0.42, genuine artifact removal) + genuinely high-amplitude recordings (>500 µV trials,
  also high in official) — none are data bugs.

**Recommendation**: **proceed to 41/10 split + SHUTrialDataset** using the **148 ok** sessions
(`SHUTrialDataset.from_manifest(..., statuses=('ok',))` already filters). First decide the 5
failed sessions: exclude, or re-extract triggers from raw (their subjects have other ok
sessions, so subject-level 41/10 split is unaffected). **Still deliberately NOT done**:
training, 41/10 split, CAP-EEGNet, any GPU job — awaiting user go-ahead.

---

## 2026-06-04 — Small-batch validation: sub-001/002/003 × 3 = 9 sessions (9/9 ok)

**Run**: `srun ... python scripts/preprocess_all.py --subjects 1,2,3 --tag dryrun` (~313 s,
compute node). Outputs to the external eog_ecg_clean tree; wrote
`processed_manifest.dryrun.csv` + `preprocess_summary.dryrun.csv` (canonical names reserved
for the full run). **9/9 ok**, every session shape [200,58,1000], float32/int64, 100/100
labels, 58 ch, 1000 times, 250 Hz, no NaN/Inf.

**New fail policy (req): `evaluate_failure_reasons()`** in `eog_ecg_clean.py`, applied after
the `.mat` cross-check in `pipeline.py`. A session is FAILED iff: shape≠[200,58,1000], label
count≠100/100, trigger count≠200, NaN/Inf in X, or (when `.mat` exists)
`labels_multiset_match=False`. `labels_match_mat=False` (order-only diff) is NOT a failure.
Added `labels_multiset_match` + `n_labels_agree` columns to the processed manifest + summary.

**Per-session results (excluded ICs / match_mat / multiset)**
- sub-001: ses-01 [1,4,10] T/T; ses-02 [1] F/T (order diff, 102/200); ses-03 [1,2] —/— (no .mat).
- sub-002: ses-01/02/03 all [] (none) T/T. Top EOG |corr| only ~0.10–0.14 → genuinely little
  IC-level ocular contamination (NOT a near-miss); sub-002 ses-01 ECG flat/invalid. Removing
  nothing is the correct conservative outcome.
- sub-003: ses-01 [1] T/T; ses-02 **[0,1,3,9,12,15,16]** T/T — 4 EOG (|corr| 0.51–0.75) + **3
  ECG (|corr| 0.56–0.77)**: first session with real cardiac ICs → ECG path validated; ses-03
  [3] T/T (see below).

**ECG**: detected in every session, but only sub-003/ses-02 had ECG-correlated ICs (≥0.5).
Elsewhere max |corr|~0.002–0.03 → no cardiac IC, not removed. Confirms ECG cleaning fires
only when there's genuine contamination.

**Robustness fix — ICA n_components retry.** sub-003/ses-03 is an EXTREME-amplitude session
(std ≈ 660 µV, ~60× normal; the paper `.mat` agrees at 660 → real data). `n_components=0.99`
collapsed to 1 PCA comp → MNE `ICA.fit` raised. Added a retry: on fit failure, refit with a
fixed int `aux_cleaning.ica.n_components_fallback` (=15). After the fix, ses-03 fits with 15
comps and removes EOG IC [3] (|corr| 0.653); the warning self-documents the retry. If the
retry also fails, the existing no-aux-clean fallback still applies (recorded, never crash).

**Disk**: each npz ≈ 41 MiB (≈43 MB; float EEG barely compresses), 9 sessions ≈ 392 MB →
full 153 ≈ ~6.5–7 GB. 38 files = 9×{npz,meta.json,preprocess_report.json,manifest_row.json}
+ 2 CSVs.

**Open questions**: (1) sub-001/ses-02 (and likely others) store trials in a different order
than the paper `.mat` — multiset matches, our (X,y) are evt.bdf-consistent; treat exact
`labels_match_mat` as non-authoritative. (2) Some subjects (sub-002) yield no ocular ICs at
threshold 0.5 — confirmed genuine here; revisit only if many subjects look under-cleaned.

**Deliberately NOT done**: full 51×3 run, training, 41/10 split, GPU. Awaiting user go-ahead.

---

## 2026-06-04 — EOG/ECG-clean preprocessing, .npz output, dry-run (sub-001)

**What changed (files)**
- `configs/paths.yaml`: renamed processed keys to `*_root`; `eog_ecg_clean_root` +
  `manifests.processed_manifest` now point to the external
  `/share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean/...`.
  `paper_style_root` kept (legacy sanity-check only, not the formal output).
- `configs/preprocess.yaml`: `mode: eog_ecg_clean`; `output.{format=npz, npz_compress,
  save_debug_npy=false, write_manifest_row}`; `aux_cleaning.{enabled, method=ica,
  fallback, validity, ica{...}}`.
- NEW `src/preprocessing/eog_ecg_clean.py`: aux-channel validity check + MNE ICA cleaning
  (fit EEG on 1 Hz-highpass copy; detect via EOG/ECG; exclude; apply) + paper-style 2nd
  half + 11-point QC report. ICA/detection failures degrade to no-aux-clean (recorded,
  never crash).
- NEW `src/preprocessing/pipeline.py`: per-session process→save(.npz/meta/report/
  manifest_row)→build manifest row; richer `.mat` cross-check. Shared by both scripts.
- `src/utils/paths.py`: `*_root` fields, `session_npz_path()`, `assert_safe_output_dir()`
  (refuses sourcedata/derivatives/raw root), `ensure_writable_dir()` (errors if unwritable).
- `src/utils/io.py`: `save_session_npz()`/`load_session_npz()`/`save_debug_npy()`.
- `src/data/manifest.py`: `PROCESSED_MANIFEST_FIELDS` + `build_processed_manifest_row()`
  (records `npz_path`, NOT X/y paths).
- `scripts/preprocess_raw.py`: dispatch on `mode`; eog_ecg_clean → .npz + report + safety.
- `scripts/preprocess_all.py`: implemented (manifest-driven; `--subjects/--sessions/
  --limit/--tag`; per-session try/except → status+error_message; writes
  `processed_manifest.csv` + `preprocess_summary.csv` beside the npz tree).
- `src/data/shu_dataset.py`: implemented — reads X/y from `npz_path` (LRU-cached, lazy
  index from y), `from_manifest()` / `from_npz_paths()`.

**.npz keys**: `X` [200,58,1000] float32 (µV), `y` [200] int64 (0/1), `subject_id`,
`session_id`, `sfreq` (=250), `channel_names` (58).

**Dry-run (sub-001, `srun` compute node, ~36 s/session, mode=eog_ecg_clean)** — 3/3 ok:
- All sessions: shape [200,58,1000], dtype float32/int64, 100/100 labels, no NaN/Inf,
  58 ch, 1000 times, 250 Hz → every QC `all_passed=true`.
- Aux cleaning REAL (correlation-based IC detection, |corr|>0.5):
  - ses-01: valid EOG=HEOR/HEOL/VEOU (VEOL flat→invalid), ECG valid; excluded EOG ICs
    [1,4,10] (corr 0.66/0.65/0.61; gap to next 0.46). std 11.28→9.99.
  - ses-02: all 4 EOG + ECG valid; excluded EOG IC [1].
  - ses-03: all valid; excluded EOG ICs [1,2] (0.69/0.61).
  - ECG: detected in every session but max IC corr ~0.02–0.03 (both `ctps` and
    `correlation` agree) → no cardiac IC removed. Plausibly little cardiac contamination
    in this montage; recorded, not forced.
- Disk: ~42–68 MB/session (compressed npz), 150 MB for 3 → full 153 ≈ ~7–8 GB.

**KEY FINDING / open question — ses-02 `.mat` label order**
- ses-01 labels match the paper `.mat` 200/200 (exact). ses-02: exact match only 102/200
  (≈chance) BUT multiset identical (100/100) and triggers are monotonic/evenly-spaced
  (same structure as ses-01). ⇒ the paper `.mat` stores ses-02 trials in a DIFFERENT
  order than acquisition; our chronological evt.bdf order pairs each `(X_i, y_i)` from the
  SAME trigger (self-consistent, correct). Cross-check now reports `labels_match` (exact)
  + `labels_multiset_match` + `n_labels_agree`. Treat `labels_match_mat` (exact) as
  non-authoritative for ordering; multiset is the meaningful check.

**Decisions**
- IC detection uses absolute-correlation (`measure='correlation'`, thr 0.5) instead of the
  MNE z-score default (3.0), which detected nothing on ses-01/02; ECG default `correlation`
  (ctps found nothing). All thresholds live in `configs/preprocess.yaml`.
- npz is `savez_compressed`; `X.npy/y.npy` only if `output.save_debug_npy=true`.

**Deliberately NOT done (await user go-ahead)**: full 51×3 `preprocess_all` (only
`--tag dryrun` files written; canonical `processed_manifest.csv` not created), CAP-EEGNet,
training, 41/10 split, any GPU job.

**Next step (after user confirms)**: run full `preprocess_all` on a compute node
(`srun`/sbatch), write canonical `processed_manifest.csv` + `preprocess_summary.csv`,
review failures/`labels_match_mat`, then 41/10 splits + CAP-EEGNet.

---

## 2026-06-04 — Paused at framework + single-session sanity check

**State**
- Framework, rules (7), configs (6), docs (11), and src/scripts skeletons complete and
  committed; working tree clean.
- Raw manifest built: `manifests/shu_2c_raw_manifest.csv` (51 subjects, 153 sessions,
  0 missing BDFs).
- Single-session sanity check RETAINED: `outputs/processed_paper_style/sub-001/ses-01/`
  = {X.npy [200,58,1000], y.npy, meta.json, manifest_row.json}; validated vs the paper
  `.mat` (labels_match=True, corr ~0.994, std 11.28 vs 11.26).
- Slurm queue checked: no jobs running/queued (today's `mi_pp_test` srun checks all COMPLETED).

**Deliberately NOT done (await user go-ahead)**
- Full 51×3 `preprocess_all` (Slurm CPU job).
- Any Slurm submission / GPU job.
- Model training code, CAP-EEGNet implementation, 41/10 split.

**Next step (after user confirms)**: run full preprocessing via the Slurm CPU job,
write `manifests/shu_2c_processed_manifest.csv`, then 41/10 splits + CAP-EEGNet.

---

## 2026-06-04 — Architecture refactor: external paths + manifests + CAP-EEGNet

**Why**: new direction — drop the baseline-first plan; main model is **CAP-EEGNet**;
raw data live OUTSIDE the repo and all paths come from `configs/paths.yaml`; MATLAB is
reference-only; everything Python/PyTorch. Current stage stays data/paths/preprocessing.

**Done**
- Dirs: added `manifests/`, `splits/`, `outputs/processed_*`; removed `data/`, `notebooks/`.
- Rules renamed/rewritten: `00-project-context` (new content), `10-data-paths` (NEW),
  `20-preprocessing`, `30-model-experiments`, `40-online-learning`, `50-server-slurm`
  (+ updated `90-agent-behavior`). Deleted old 10/20/30/40.
- `src/utils/paths.py` refactored: no hardcoded dataset root; `load_paths()` reads
  `configs/paths.yaml` (env `SHU_2C_ROOT` overrides), validates, returns a `Paths` object.
- `configs/`: NEW `paths.yaml`; `preprocess.yaml` slimmed to params only; renamed
  cross_subject→`train_cross_subject.yaml`, online→`online_adaptation.yaml` (CAP-EEGNet).
- `scripts/build_manifest.py` + `src/data/manifest.py`: scan external raw root →
  `manifests/shu_2c_raw_manifest.csv`. Verified: **51 subjects, 153 sessions, 0 missing**.
- `check_raw_bdf.py` + `preprocess_raw.py` now read `configs/paths.yaml`; preprocess
  writes to the configured processed dir, writes `manifest_row.json`, and (optional)
  cross-checks labels vs the paper `.mat`.
- Skeletons (no complex code yet): `src/models/cap_eegnet.py` (encoder+adapter+
  prototype+confidence+cls), `src/data/splits.py`, `src/data/shu_dataset.py`.
- Re-validated end-to-end on a compute node (srun): sub-001/ses-01 →
  `outputs/processed_paper_style/...`, shape [200,58,1000], **mat labels_match=True**,
  std 11.28 vs 11.26.

**Decision**: `configs/paths.yaml` ships with the verified real raw root
(`/share/workspace2/moto_imagination/WBCIC_SHU`) since it's known; loader still
validates + supports `SHU_2C_ROOT` override. Change the YAML if data moves.

**Next step**: when ready, implement `src/data` (dataset + subject-wise split),
then CAP-EEGNet encoder. Still NO full 51×3 preprocessing / GPU jobs in this stage.

---

## 2026-06-04 — Task 2: raw preprocessing implemented & validated (sub-001/ses-01)

**Done**
- Cracked the evt.bdf event parsing (was the #1 blocker). The 200 MI triggers live
  in the `BDF Annotations` TAL channel of the Neuracle BDF+C file; MNE only surfaces
  the block markers {7,8}. Wrote `src/preprocessing/neuracle_events.py` to parse the
  BDF header + TAL bytes directly -> recovers 100x'1'(left) + 100x'2'(right), ~8s apart.
- Implemented `src/preprocessing/shu_preprocess.py` (paper-style), faithful to the
  authors' `code/pre-processed/preprocessed.m`:
  drop {ECG,HEOR,HEOL,VEOU,VEOL} -> reref Pz & drop Pz (58 EEG) -> 0.5-40 bandpass
  -> 50 notch -> epoch [0,4)s (baseline = whole-epoch demean) -> resample 250 -> [200,58,1000].
- Wired `scripts/preprocess_raw.py`; ran on a COMPUTE NODE via `srun` (not login node).

**Validated vs paper .mat (derivatives/)**
- Shape [200,58,1000], labels match **exactly** (element-wise, all 200 trials).
- Signal correlation 0.988-1.000 (mean 0.994). After fixing a unit bug, scale matches:
  our std 11.283 vs paper 11.263 (ratio 1.0017), RMSE 0.876 uV (~7.8% of std). The
  residual is expected from EEGLAB-vs-MNE filter implementation differences.

**Decisions / gotchas**
- UNIT QUIRK: BDF physical dim is the garbled `?V` (meant µV), so MNE does NOT apply
  µV->V scaling; `get_data()` already returns µV-magnitude values. We store as-is
  (µV) and do NOT multiply by 1e6 (doing so was a bug, fixed).
- reref target Pz = EEGLAB channel index 43 (confirmed by counting the EEG montage).
- Epoch window [0,4)s at 1000Hz (4000 samples) then resample -> 1000 samples.

**Next step**
- Generalize to all 51x3 via `scripts/preprocess_all.py` + `scripts/slurm/preprocess_cpu.sbatch`
  (collect per-session status into outputs/preprocess_summary.csv; don't silently skip
  failures). Then start the EEGNet baseline.

---

## 2026-06-04 — Project scaffold created

**Done**
- Created project skeleton at `/share/home/yuan/SYX/eeg-mi-online/`:
  `.cursor/rules/` (6 rules), `docs/` (8 docs + references), `configs/`,
  `scripts/` + `scripts/slurm/`, `src/` (7 packages), `data/`, `outputs/`,
  `logs/`, `checkpoints/`, `notebooks/`.
- Wrote the 6 Cursor rules, the docs (PROJECT_BRIEF, DATASET_SHU,
  PREPROCESSING_SPEC, EXPERIMENT_PROTOCOL, MODEL_PLAN, SERVER_RUNBOOK, ENVIRONMENT),
  AGENTS.md, this file, `.gitignore`, `requirements.txt`.
- Copied the senior's chat record to `docs/references/ChatGPT-EEG-MI-pretraining.md`.
- Implemented `scripts/check_raw_bdf.py` (raw BDF inspector, Task 1).
- Added documented stubs for the other scripts + `src/` modules.
- Wrote Slurm sbatch templates in `scripts/slurm/` (adapted from `run_test.sh`).
- Initialized git and made the first commit.

**Verified facts (from inspecting the real server/dataset)**
- Dataset root: `/share/workspace2/moto_imagination/WBCIC_SHU` (BIDS, READ-ONLY).
- 2C: **51** subject folders on disk; `participants_2C.tsv` lists 52; README says 53.
  -> Decision: always enumerate subjects from disk, never hardcode the count.
- Raw: 1000 Hz, **64 ch**. `check_raw_bdf.py` on sub-001/ses-01 VERIFIED the real
  layout = **59 EEG + 1 ECG (`ECG`) + 4 EOG (`HEOR/HEOL/VEOU/VEOL`)**. The
  `task-motorimagery_eeg.json` count "1 EOG, 4 ECG" is SWAPPED vs the actual names
  -> trust the names. (The generic plan's "ch60=ECG, ch61-64=EOG" was actually
  closer to reality than the JSON.)
- Events: 1=left, 2=right, 3=foot(3C only). 2C uses {1,2} -> internal {0,1}.
- Target processed shape `[200, 58, 1000]` (58 = 59 EEG minus Pz; 1000 = 4 s @ 250 Hz).
- Paper `.mat` confirmed: `data [58,1000,200]` float32, `labels [1,200]` in {1,2}.
- Env `mi_torch`: py3.10.18, torch 2.6.0, mne 1.10.0, numpy 2.2.5, scipy 1.15.3,
  sklearn 1.7.1, pandas 2.2.3, h5py 3.14.0, einops 0.8.1. No braindecode.
- Slurm: `gpu2node`(default)/`gpu3node`, each `gpu:8`/128 CPU/~773 GB; modules
  `cuda/11.8`, `anaconda3`.

**Decisions**
- Two preprocessing variants: paper-style (first, no ICA) and EOG/ECG-clean (later).
- First model = EEGNet encoder + classification head + prototype + confidence head
  + adapter (the chat record's "minimal version"). Baselines (EEGNet/DeepConvNet/
  FBCNet) come before that, to validate preprocessing.
- Git identity for commits: see commit log; tell the maintainer if it needs changing.

**check_raw_bdf.py first run (sub-001/ses-01) — verified**
- sfreq=1000 Hz, 64 ch, duration=2250 s (~37.5 min; ~11.25 s/trial for 200 trials).
- Channels: 59 EEG + 1 ECG(`ECG`) + 4 EOG(`HEOR/HEOL/VEOU/VEOL`). `other`=[] after
  improving the classifier to recognize H/V-EOG names.
- Aux validity: first 10 s of `ECG` and `VEOL` were all-zero (flat); `HEOR/HEOL/VEOU`
  active. -> must validate aux over the full recording before use (variant 2).
- Report saved at `outputs/raw_check/sub-001_ses-01_raw_check.json`.

**Open questions / TODO before scaling up**
- ⚠️ `mi_torch` torch is **CPU-only** (`torch.version.cuda is None`). Must install a
  cu118-matched torch (or make a `mi_torch_cu118` env) before real GPU training.
  (RESOLVED later: `mi_torch_cu118` env created, torch 2.7.1+cu118.)
- ⚠️ **Event triggers**: `evt.bdf` annotations via MNE gave only `{"7":1,"8":1}`
  (channel "Empty Event Data"), NOT 200 MI markers. (RESOLVED: TAL parser in
  `neuracle_events.py`.)
- Confirm cue timing for the 4 s epoch window + baseline interval (paper detail TBD).
- Check whether the flat ECG/VEOL is session-specific or dataset-wide.

**Next step**
- Run `scripts/check_raw_bdf.py` on `sub-001/ses-01` (login-node-safe: reads one
  file, prints + dumps JSON), read the report, then implement `preprocess_raw.py`
  for a single session and confirm the `[200, 58, 1000]` shape.
