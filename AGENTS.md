# AGENTS.md — Read me first

Persistent context for AI agents (and humans). The project's "soul memory".
Keep it accurate; read it first. Master brief: `docs/PROJECT_BRIEF.md`.
Memory/journal: `docs/PROGRESS.md`. Core idea: `docs/references/ChatGPT-EEG-MI-pretraining.md`.

## What this project is

MI-EEG decoding on the Shanghai University **WBCIC-SHU 2C** dataset, fully in
**Python/PyTorch**: raw BDF → Python preprocessing → **41/10 cross-subject
pretraining of CAP-EEGNet** → target Session-1 fine-tuning → Session-2/3 online
test-then-update, with explicit sample confidence.

## Non-negotiable constraints

1. **Everything is Python/PyTorch.** MATLAB (`preprocessed.m`) + Neuracle toolbox are
   a REFERENCE RECIPE only, never a runtime dependency.
2. **Never hard-code data paths.** Load from `configs/paths.yaml` via
   `src/utils/paths.load_paths()` (env `SHU_2C_ROOT` overrides). Unknown/missing path
   → raise a clear error and ask the user to fill it. Never guess.
3. Raw data live **outside** the repo and are READ-ONLY. Never write into them.
   Processed data go to the configured `processed_*` dir (default `outputs/...`).
4. The paper `.mat` in `derivatives/` is a label/event **cross-check truth only**,
   NOT the training data entry point (that is our Python-preprocessed `.npy`).
5. Tensor convention `[trials, channels, time]`, target `[200, 58, 1000]`; labels
   1→0 (left), 2→1 (right).
6. Split by subject, never by trial. No target leakage into source training.
7. Online learning is test-then-update (predict+record, THEN update).
8. Comment your code (docstrings + EEG-specific reasoning). Chinese or English.
9. **Current stage = data/paths/preprocessing.** Do NOT build the full CAP-EEGNet /
   training / online loops yet (skeletons only), do NOT preprocess all 51×3, do NOT
   submit GPU jobs. Heavy/GPU work via Slurm only, never on the login node.

## Directory map

| Path | What |
| --- | --- |
| `.cursor/rules/` | 00-project-context, 10-data-paths, 20-preprocessing, 30-model-experiments, 40-online-learning, 50-server-slurm, 90-agent-behavior |
| `configs/` | `paths.yaml` (all paths), `preprocess.yaml` (params), `train_cross_subject.yaml`, `online_adaptation.yaml`, `finetune.yaml`, `eegnet_baseline.yaml` (optional) |
| `manifests/` | `shu_2c_raw_manifest.csv` (bridge to external raw), processed manifest |
| `splits/` | persisted subject-wise 41/10 splits (JSON) |
| `scripts/` | `build_manifest.py`, `check_raw_bdf.py`, `preprocess_raw.py`, `preprocess_all.py`, train/finetune/online (run from project root); `scripts/slurm/` |
| `src/` | data, preprocessing, models (cap_eegnet), training, evaluation, online, utils |
| `outputs/`, `logs/`, `checkpoints/` | run artifacts + processed data (gitignored) |

(No `data/` dir — raw is external; processed goes to `outputs/processed_*`.)

## Verified facts (so you don't re-derive)

- 2C: 51 subjects, 153 sessions on disk (manifest confirms, 0 missing). 3 sessions ea,
  200 trials/session (100/class), classes left(1)/right(2).
- Raw 1000 Hz, 64 ch = 59 EEG + 1 ECG(`ECG`) + 4 EOG(`HEOR/HEOL/VEOU/VEOL`)
  [`eeg.json` EOG/ECG counts are swapped]. Reref Pz → drop → 58 EEG. 4 s @ 250 Hz = 1000.
- Data unit is µV (BDF dim `?V`; MNE does not rescale — do NOT ×1e6).
- evt.bdf triggers are in the TAL channel; parse via `src/preprocessing/neuracle_events.py`
  (MNE misses them). 100×left + 100×right.
- Paper-style preprocessing implemented & validated on sub-001/ses-01 vs the `.mat`
  (labels match exactly, corr ~0.994, std 11.28 vs 11.26).
- Slurm: partitions `gpu2node`(default)/`gpu3node`, `gpu:8`, modules `cuda/11.8`,`anaconda3`,
  env `mi_torch` (torch CPU-only — fix before GPU training).

## Main model: CAP-EEGNet

EEGNet encoder + Adapter + Classification head + Prototype head + Confidence head.
See `docs/MODEL_PLAN.md`. Keep as skeleton until preprocessing is finalized.
