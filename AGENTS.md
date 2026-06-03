# AGENTS.md — Read me first

This is the persistent context file for AI agents (and humans) working in this repo.
It is the project's "soul memory". Keep it accurate; it is the first thing to read.

## What this project is

MI-EEG decoding on the Shanghai University **WBCIC-SHU** dataset. The arc:
raw BDF -> Python preprocessing -> baseline reproduction -> 41/10 cross-subject
pretraining -> target Session-1 fine-tuning -> Session-2/3 online test-then-update,
with explicit **sample confidence** prediction throughout.

Core idea / 宗旨: `docs/references/ChatGPT-EEG-MI-pretraining.md` (senior's chat record).
Master brief: `docs/PROJECT_BRIEF.md`. Memory/journal: `docs/PROGRESS.md`.

## Non-negotiable constraints (memorize these)

1. **Dataset is READ-ONLY**: `/share/workspace2/moto_imagination/WBCIC_SHU`.
   Never modify/write/delete anything outside `/share/home/yuan/SYX`.
2. **Never run training or full preprocessing on the Slurm login node.** Use
   `sbatch` (scripts in `scripts/slurm/`) or an `srun`/`salloc` compute session.
   Only <~30 s sanity checks are allowed on the login node.
3. **Env**: conda `mi_torch` (py3.10, torch 2.6 — currently CPU-only, see
   `docs/ENVIRONMENT.md`). Verify CUDA before long GPU runs.
4. **Tensor convention**: saved processed data is `[trials, channels, time]`,
   target `[200, 58, 1000]`; labels {1,2} -> {0,1}.
5. **Split by subject, never by trial.** No target leakage into source training.
6. **Online learning is test-then-update** (predict+record, THEN update). Never
   train on a trial before evaluating it.
7. **Comment your code** (project requirement). Docstrings on every module/script.
8. **Don't invent** paths / channel order / event ids — inspect first
   (`scripts/check_raw_bdf.py`).

## Where things live

| Path | What |
| --- | --- |
| `.cursor/rules/*.mdc` | fixed rules (auto-applied by Cursor) |
| `docs/` | briefs, dataset/preproc/experiment/model specs, server runbook, env |
| `docs/PROGRESS.md` | dated progress + decision log (UPDATE THIS as you work) |
| `configs/*.yaml` | all tunable parameters |
| `scripts/` | entrypoints (run from project root); `scripts/slurm/` = sbatch |
| `src/` | library code (data, preprocessing, models, training, evaluation, online, utils) |
| `data/` | processed data (gitignored); raw is read from the dataset dir |
| `outputs/`, `logs/`, `checkpoints/` | run artifacts (gitignored) |

## Verified key facts (so you don't re-derive them)

- 2C dataset: 51 subject folders on disk (tsv lists 52; enumerate from disk),
  3 sessions each, 200 trials/session (100/class), classes left(1)/right(2).
- Raw: `sourcedata/2C dataset/sub-XXX/ses-YY/eeg/{data.bdf,evt.bdf}`, 1000 Hz,
  64 ch = 59 EEG + 1 ECG(`ECG`) + 4 EOG(`HEOR/HEOL/VEOU/VEOL`) [VERIFIED; note the
  `eeg.json` EOG/ECG counts are swapped]. Re-ref to Pz, drop Pz -> 58 EEG.
  4 s @ 250 Hz = 1000 samples. ⚠ evt.bdf trigger parsing is an open question (Task 2).
- Paper `.mat` (derivatives): `data [58,1000,200]`, `labels [1,200]` in {1,2}.
- Slurm: partitions `gpu2node`(default)/`gpu3node`, `gpu:8`, modules `cuda/11.8`,`anaconda3`.

## Working rhythm

inspect one session -> preprocess one session -> confirm `[200,58,1000]` ->
preprocess all (Slurm CPU) -> EEGNet baseline -> 41/10 split -> prototype+confidence
-> Session-1 fine-tune -> Session-2/3 online. Don't jump ahead to a giant model.
