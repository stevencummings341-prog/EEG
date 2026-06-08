# AGENTS.md — Read me first

Persistent context for AI agents (and humans). The project's "soul memory".
Keep it accurate; read it first. Authoritative status: `docs/PROJECT_STATUS_CURRENT.md`.
Master brief: `docs/PROJECT_BRIEF.md`. Full overview: `docs/PROJECT_OVERVIEW.md`.
Memory/journal: `docs/PROGRESS.md`. Core idea: `docs/references/ChatGPT-EEG-MI-pretraining.md`.

## What this project is

MI-EEG decoding on the Shanghai University **WBCIC-SHU 2C** dataset, fully in
**Python/PyTorch**. **Current mainline = cross-session domain generalization (DG):**
(A) diagnose what drifts across sessions, (B) static baselines (EEGNet/DeepConvNet/FBCNet)
at within-session CV + single-source cross-session, (Step 1) multi-source `ses-01+02 → ses-03`,
(Step 2, next) no-learning adaptation baseline, then (future) online continual learning.

The long-term vision (CAP-EEGNet: confidence-aware online adaptive multi-subagent framework)
is kept in `docs/ROADMAP.md` but is NOT the current runnable mainline.

## Behavior rules (read before acting)

1. **First read** `docs/PROJECT_STATUS_CURRENT.md`, then `docs/PROGRESS.md` + `docs/PROJECT_OVERVIEW.md`.
2. **Current mainline = cross-session DG.** Do exactly the requested step; **do NOT advance
   multiple steps at once** (no jumping ahead to Step 2/online without being asked).
3. **Never write "done" for things not run.** 41/10, online, fine-tuning, CAP-EEGNet full,
   multi-agent/prototype/memory = FUTURE. **Step 2 no-learning adaptation = NOT run.**
4. **Code first smoke-test, then full run.** Smoke on a GPU node via `srun` before any `sbatch`.
5. **Never overwrite completed results** (`outputs/.../*_v1/`, `checkpoints/.../*_v1/`); new
   experiments get a new `*_v2`/distinct run_id.
6. **After finishing a meaningful step, update `docs/PROGRESS.md` + `docs/EXPERIMENT_LOG.md`**
   (and the status page) — factually, no exaggeration.
7. **No heavy work on the login node**; **never write raw / workspace2 source data**.
8. P10 folder (`/share/home/yuan/SYX/P10_MI泛化研究/`) is **read-only reference**; do not run from
   it or copy its draft code over `src/`/`scripts/`.

## Non-negotiable constraints

1. **Everything is Python/PyTorch.** MATLAB (`preprocessed.m`) + Neuracle toolbox are
   a REFERENCE RECIPE only, never a runtime dependency.
2. **Never hard-code data paths.** Load from `configs/paths.yaml` via
   `src/utils/paths.load_paths()` (env `SHU_2C_ROOT` overrides). Unknown/missing path
   → raise a clear error and ask the user to fill it. Never guess.
3. Raw data live **outside** the repo and are READ-ONLY. **Never write into raw / workspace2
   source dirs.** Processed data is the external `eog_ecg_clean` `.npz` tree; run artifacts go to
   `outputs/`, weights to `checkpoints/`, logs to `logs/`.
4. The paper `.mat` in `derivatives/` is a label/event **cross-check truth only**, NOT the
   training data entry (that is our Python-preprocessed per-session `.npz`, `status=ok` only).
5. Tensor convention `[trials, channels, time]`, target `[200, 58, 1000]`; labels 1→0 (left),
   2→1 (right); normalized to {0,1}.
6. Split by subject/session, **never leak by trial**. For cross / multi-source / adaptation, the
   early-stopping val slice is carved **only from train**, and **the test session's labels are
   NEVER used** for train/val/early-stopping/tuning.
7. Online learning (future) is test-then-update (predict+record, THEN update).
8. Comment your code (docstrings + EEG-specific reasoning). Chinese or English.
9. Heavy/GPU work via **Slurm only** (GPU env `mi_torch_cu118`, torch 2.7.1+cu118), never on the
   login node. Smoke-test (subjects 1,2, few epochs) on a GPU node via `srun` before any full run.

## Current status (2026-06-08) — see `docs/PROJECT_STATUS_CURRENT.md`

- ✅ Full preprocessing (148 ok / 5 failed), QC PASS.
- ✅ **A** session-drift diagnostic (144 pairs / 50 subjects).
- ✅ **B** static baseline EEGNet/DeepConvNet/FBCNet, 5 seeds: within-session 10-fold CV +
  single-source directed cross-session.
- ✅ **Step 1** multi-source `ses-01+ses-02 → ses-03` (47 eligible subjects, 4 skipped, 705 rows;
  multi-source beats best single source for all 3 models). See `docs/MULTISOURCE_STEP1_REPORT.md`.
- 🔜 **Step 2 (next, NOT run)** = no-learning adaptation baseline (none / session_zscore /
  Euclidean Alignment / Riemannian Alignment / target BN-stats adaptation / filter-bank
  reweighting). Plan: `docs/ADAPTATION_BASELINE_PLAN.md`, `docs/NEXT_EXPERIMENT_PLAN.md`.
- 🚧 **future (Step 3+, not run / not validated)**: online learning, 41/10 cross-subject
  pretraining, target fine-tuning, LOSO, CAP-EEGNet full, multi-agent/prototype/memory (incl.
  P10's draft `multi_agent_*`/`online_drift_*` code). P10 dir (read-only, do NOT run / write):
  `/share/home/yuan/SYX/P10_MI泛化研究/` (see `docs/P10_INTEGRATION_SUMMARY.md`).

> ⚠️ git: HEAD is the 2026-06-04 scaffold commit; everything since is uncommitted. A tooling
> hiccup once dropped uncommitted docs/code (results were safe; restored 2026-06-09). Commit soon.

## Directory map

| Path | What |
| --- | --- |
| `.cursor/rules/` | 00-project-context, 10-data-paths, 20-preprocessing, 30-model-experiments, 40-online-learning, 50-server-slurm, 90-agent-behavior |
| `configs/` | `paths.yaml`, `preprocess.yaml`, `session_drift.yaml`, `session_model_compare.yaml`, `session_multisource_compare.yaml`; (future) `train_cross_subject/finetune/online_adaptation/eegnet_baseline.yaml` |
| `src/` | `analysis/` (drift), `data/` (session_splits, shu_dataset, splits, manifest), `models/` (eegnet, deepconvnet, fbcnet, cap_eegnet, registry), `training/` (trainer), `evaluation/` (session_protocols, session_multisource_protocols, metrics, data_quality), `preprocessing/`, `utils/`, `visualization/`, `online/` (future) |
| `scripts/` | preprocess + `train_session_models.py`, `train_session_multisource.py`, `summarize_session_results.py`, `summarize_multisource_results.py`, `analysis/`, `slurm/` |
| `docs/` | PROJECT_STATUS_CURRENT, PROJECT_OVERVIEW, RESULTS_SUMMARY, MULTISOURCE_STEP1_REPORT, BASELINE_PROTOCOL, SESSION_DRIFT_ANALYSIS, NEXT_EXPERIMENT_PLAN, ADAPTATION_BASELINE_PLAN, CODE_INTEGRATION_NOTES, P10_INTEGRATION_SUMMARY, PROGRESS, EXPERIMENT_LOG, references/ |
| `outputs/`, `logs/`, `checkpoints/` | run artifacts (gitignored); processed `.npz` live in the external workspace |

## Verified facts (so you don't re-derive)

- 2C: 51 subjects, 153 sessions (manifest confirms). 148 ok / 5 failed; failed sessions =
  sub-023/ses-01, sub-024/ses-02, sub-024/ses-03, sub-026/ses-01, sub-032/ses-02 (trigger<200).
  47 subjects have all 3 sessions ok; subjects 023/024/026/032 are the 4 partial ones.
- 200 trials/session (100/class), classes left(1)/right(2).
- Raw 1000 Hz, 64 ch = 59 EEG + 1 ECG(`ECG`) + 4 EOG(`HEOR/HEOL/VEOU/VEOL`)
  [`eeg.json` EOG/ECG counts are swapped]. Reref Pz → drop → 58 EEG. 4 s @ 250 Hz = 1000.
- Data unit is µV (BDF dim `?V`; MNE does not rescale — do NOT ×1e6).
- evt.bdf triggers are in the TAL channel; parse via `src/preprocessing/neuracle_events.py`.
- GPU env `mi_torch_cu118` (torch 2.7.1+cu118, cuda 11.8); RTX 4090 D. Slurm partitions
  `gpu2node`(default)/`gpu3node`, `gpu:8`.

## Models (current mainline)

EEGNet (Lawhern 2018) / DeepConvNet (Schirrmeister 2017) / FBCNet (Mane 2021), plus CAP-EEGNet
**v1** (encoder + classifier + learned confidence head; the rest raises `NotImplementedError`).
All share a `{logits, features, confidence}` forward contract + one trainer + one metric set for
fair comparison. CAP-EEGNet full + all complex modules = future (`docs/ROADMAP.md`).
