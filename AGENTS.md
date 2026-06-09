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
(Step 2, DONE) no-learning adaptation baseline, then (Step 3, future) learning-based / online
continual adaptation.

The long-term vision (CAP-EEGNet: confidence-aware online adaptive multi-subagent framework)
is kept in `docs/ROADMAP.md` but is NOT the current runnable mainline.

## Where the data lives (addresses)

All paths come from `configs/paths.yaml` (load via `src/utils/paths.load_paths()`, env
`SHU_2C_ROOT` overrides). For quick reference, the verified locations are:

- **Project root (this repo):** `/share/home/yuan/SYX/eeg-mi-online`
- **Raw 2C dataset (external, READ-ONLY):** `/share/workspace2/moto_imagination/WBCIC_SHU`
  (BIDS; raw under `sourcedata/2C dataset`, paper `.mat` under `derivatives/2C dataset_processeddata`).
- **Processed training entry (external, READ-ONLY for us — we only consume it):**
  `/share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean/` — per-session `.npz`
  (`X[200,58,1000]` µV @250 Hz, `y[200]∈{0,1}`), `status=ok` only (148 ok / 5 failed).
- **Processed manifest:** `.../processed/eog_ecg_clean/processed_manifest.csv` (one row per session,
  has `npz_path` + `status`).
- **Run artifacts:** `outputs/` (gitignored). **Weights:** `checkpoints/` (gitignored).
  **Logs:** `logs/slurm/`.

## Behavior rules (read before acting)

1. **First read** `docs/PROJECT_STATUS_CURRENT.md`, then `docs/PROGRESS.md` + `docs/PROJECT_OVERVIEW.md`.
2. **Current mainline = cross-session DG.** Do exactly the requested step; **do NOT advance
   multiple steps at once** (no jumping ahead to the next step / online without being asked).
3. **Never write "done" for things not run.** 41/10, online, fine-tuning, CAP-EEGNet full,
   multi-agent/prototype/memory = FUTURE (Step 3+, NOT run/validated). Mark a step complete ONLY
   after its results exist on disk (e.g. the summarizer's results CSV), never on submission alone.
4. **Code first smoke-test, then full run.** Smoke on a GPU node via `srun` before any `sbatch`.
5. **Never overwrite completed results** (`outputs/.../*_v1/`, `checkpoints/.../*_v1/`); new
   experiments get a new `*_v2`/distinct run_id.
6. **After finishing a meaningful step, update `docs/PROGRESS.md` + `docs/EXPERIMENT_LOG.md`**
   (and the status page) — factually, no exaggeration.
7. **No heavy work on the login node** (login = edit/git/inspect/submit + <~30 s checks only).
8. **Filesystem scope:** write ONLY inside the project root `/share/home/yuan/SYX/eeg-mi-online`.
   - NEVER write anywhere under `/share/workspace2/...` (raw + processed dataset) — it is READ-ONLY
     input; we *read* the `.npz`/manifest from there but never modify, add, or delete files there.
   - Do not write into other folders under `/share/home/yuan/SYX/` (e.g. handoff folders, see below)
     unless the user explicitly asks; treat them as read-only references.
   - Do not touch anything outside `/share/home/yuan/SYX/` at all.
9. **Handoff / reference folders (not hard-coded).** The user periodically drops a NEW folder into
   the workspace after talking with the senior (it may have any name/path, e.g. a `P10_...`-style
   research package). When such a folder is present and the user points you at it: **read it,
   understand the intent, reference its code/notes, and sensibly fold the useful parts into the
   CURRENT architecture** (`src/`, `scripts/`, `configs/`, `docs/`) following this repo's
   conventions. Treat the handoff folder itself as **read-only**: don't run from it and don't copy
   its draft code verbatim over `src/`/`scripts/`. No specific folder name is a permanent
   instruction — only act on it when the user references it.

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
   Activate the env with the real base on this cluster:
   `source /share/software/anaconda3/2024.10/etc/profile.d/conda.sh && conda activate mi_torch_cu118`.
   GPU jobs must **fail-fast if `torch.cuda.is_available()` is False** (don't silently fall back to CPU).

## Current status (2026-06-09) — see `docs/PROJECT_STATUS_CURRENT.md`

- ✅ Full preprocessing (148 ok / 5 failed), QC PASS.
- ✅ **A** session-drift diagnostic (144 pairs / 50 subjects; per-subject `drift_level` high/moderate/stable).
- ✅ **B** static baseline EEGNet/DeepConvNet/FBCNet, 5 seeds: within-session 10-fold CV +
  single-source directed cross-session.
- ✅ **Step 1** multi-source `ses-01+ses-02 → ses-03` (47 eligible subjects, 4 skipped, 705 rows;
  multi-source beats best single source for all 3 models). See `docs/MULTISOURCE_STEP1_REPORT.md`.
- ✅ **Step 2 (DONE 2026-06-09)** no-learning adaptation baseline (none_reference / session_zscore /
  euclidean_alignment / riemannian_alignment / bn_statistics_adaptation / filterbank_reweighting),
  3 models × 5 seeds × (288 single-source pairs + 47 multi-source). 30,150 rows, 0 failed/0 NaN.
  **Result = negative/diagnostic: no-learning alignment is INSUFFICIENT** (best `bn_statistics_adaptation`
  Δacc +0.0071 over none 0.6818→0.6889; EA/RA slightly hurt; no method clears +2pp; high-drift
  helped least). Outputs: `outputs/experiments/alignment_baseline_v1/`. See
  `docs/ADAPTATION_BASELINE_PLAN.md`.
- 🚧 **future (Step 3+, not run / not validated)**: learning-based / online adaptation (test-then-update),
  adapter/prototype/memory, 41/10 cross-subject pretraining, target fine-tuning, LOSO, CAP-EEGNet full,
  multi-agent. Step 2's negative result is the objective motivation, but Step 3 is NOT started.

> git: work is committed regularly (latest = Step-2 closeout). No `user.name/email` is set in this
> environment, so commit with a one-off identity override (do NOT edit git config), reusing the
> repo's existing author (`git log -1 --format='%an %ae'`).

## Directory map / architecture

Data flow: external raw BDF → (preprocessing, already done) external `eog_ecg_clean/*.npz` +
`processed_manifest.csv` → `src/data/session_splits.load_ok_sessions` (status=ok) → per-protocol
evaluation modules build splits + tensors → shared `src/training/trainer` trains a model from
`src/models/registry` → `src/evaluation/metrics` → per-run CSVs in `outputs/.../runs/` → a
`summarize_*` script aggregates to `tables/` + `figures/` + a report. Every model shares the
`{logits, features, confidence}` forward contract + one trainer + one metric set for fair comparison.

| Path | What |
| --- | --- |
| `.cursor/rules/` | 00-project-context, 10-data-paths, 20-preprocessing, 30-model-experiments, 40-online-learning, 50-server-slurm, 90-agent-behavior |
| `configs/` | `paths.yaml`, `preprocess.yaml`, `session_drift.yaml`, `session_model_compare.yaml`, `session_multisource_compare.yaml`, `session_alignment_compare.yaml` (Step 2); (future) `train_cross_subject/finetune/online_adaptation`.yaml |
| `src/data/` | `session_splits.py` (load_ok_sessions, within folds, directed cross pairs, label norm), `shu_dataset.py`, `splits.py`, `manifest.py` |
| `src/models/` | `eegnet.py`, `deepconvnet.py`, `fbcnet.py`, `cap_eegnet.py` (v1), `registry.py` (`build_model`) |
| `src/training/` | `trainer.py` (one CE trainer + early stopping + predict, model-agnostic) |
| `src/evaluation/` | `session_protocols.py` (within + single-source cross), `session_multisource_protocols.py` (Step 1), `session_alignment_protocols.py` (Step 2), `metrics.py`, `data_quality.py` |
| `src/adaptation/` | **Step 2** `session_alignment.py` (z-score / Euclidean / Riemannian log-Euclidean SPD mean / filter-bank), `bn_adaptation.py` (BN running-stat refresh, no optimizer) |
| `src/` (other) | `analysis/` (drift), `preprocessing/`, `utils/` (paths, io, config, seed, logging), `visualization/`, `online/` (future) |
| `scripts/` | preprocess + `train_session_models.py`, `train_session_multisource.py`, `train_session_alignment.py`, `summarize_session_results.py`, `summarize_multisource_results.py`, `summarize_alignment_results.py`, `build_alignment_baseline_outputs.py`, `analysis/`, `slurm/*.sbatch` |
| `docs/` | PROJECT_STATUS_CURRENT, PROJECT_OVERVIEW, RESULTS_SUMMARY, MULTISOURCE_STEP1_REPORT, BASELINE_PROTOCOL, SESSION_DRIFT_ANALYSIS, NEXT_EXPERIMENT_PLAN, ADAPTATION_BASELINE_PLAN, CODE_INTEGRATION_NOTES, PROGRESS, EXPERIMENT_LOG, references/ |
| `outputs/`, `logs/`, `checkpoints/` | run artifacts (gitignored). Canonical experiment dirs: `outputs/experiments/{baseline_v1, alignment_baseline_v1}/`, `outputs/analysis/session_drift_v1/`. Processed `.npz` live in the external workspace (read-only). |

## Verified facts (so you don't re-derive)

- 2C: 51 subjects, 153 sessions (manifest confirms). 148 ok / 5 failed; failed sessions =
  sub-023/ses-01, sub-024/ses-02, sub-024/ses-03, sub-026/ses-01, sub-032/ses-02 (trigger<200).
  47 subjects have all 3 sessions ok; subjects 023/024/026/032 are the 4 partial ones.
- 200 trials/session (100/class), classes left(1)/right(2).
- Raw 1000 Hz, 64 ch = 59 EEG + 1 ECG(`ECG`) + 4 EOG(`HEOR/HEOL/VEOU/VEOL`)
  [`eeg.json` EOG/ECG counts are swapped]. Reref Pz → drop → 58 EEG. 4 s @ 250 Hz = 1000.
- Data unit is µV (BDF dim `?V`; MNE does not rescale — do NOT ×1e6).
- evt.bdf triggers are in the TAL channel; parse via `src/preprocessing/neuracle_events.py`.
- GPU env `mi_torch_cu118` (torch 2.7.1+cu118, cuda 11.8) — CUDA works, GPU Slurm jobs are the
  normal path now (the old `mi_torch` was CPU-only; do not use it for GPU). RTX 4090 D.
  Slurm partitions `gpu2node`(default=gpu01,02)/`gpu3node`(gpu03,04,05), `gpu:8` per node; ALWAYS
  set `-t`. A per-user CPU/QOS cap means many submitted jobs sit `PD (QOSMaxCpuPerUserLimit)` and
  drain as capacity frees — normal, not an error. Conda base:
  `/share/software/anaconda3/2024.10/etc/profile.d/conda.sh`. Monitor with `slmwatch`/`gpuwatch` on
  login01. IO-heavy work (large copy/compress/backup) → `storge` node, not login01. Home quota
  512 GiB (`quota -s`). Official cluster doc: http://10.26.1.75:58080/ (see `docs/SERVER_RUNBOOK.md`).

## Models (current mainline)

EEGNet (Lawhern 2018) / DeepConvNet (Schirrmeister 2017) / FBCNet (Mane 2021), plus CAP-EEGNet
**v1** (encoder + classifier + learned confidence head; the rest raises `NotImplementedError`).
All share a `{logits, features, confidence}` forward contract + one trainer + one metric set for
fair comparison. CAP-EEGNet full + all complex modules = future (`docs/ROADMAP.md`).
