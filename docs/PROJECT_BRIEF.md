# Project Brief: SHU MI-EEG Cross-subject Pretraining and Online Adaptation

> 核心思想来源：`docs/references/ChatGPT-EEG-MI-pretraining.md`（学长聊天记录）。
> 本文件是总说明书；固定约束见 `.cursor/rules/`。

## Goal

A fully **Python/PyTorch** pipeline for the Shanghai University **WBCIC-SHU 2C**
motor-imagery EEG dataset: from raw BDF → Python preprocessing → **41/10
cross-subject pretraining of CAP-EEGNet** → target-subject fine-tuning →
online test-then-update — with explicit **sample confidence** throughout.

## Current decision (important)

- The current main workflow does **NOT** prioritize reproducing EEGNet/DeepConvNet/
  FBCNet baselines. Baselines are **optional**, not the priority.
- The main workflow starts from raw BDF preprocessing and proceeds directly to our
  own model (**CAP-EEGNet**) and experiments.
- MATLAB (`preprocessed.m`) and the Neuracle toolbox are a **reference recipe only** —
  never a runtime dependency. The paper `.mat` in `derivatives/` is a label/event
  **cross-check truth only**, not the training data entry point.
- Current stage = **raw-data inspector / Python preprocessing**. Do NOT yet build the
  full model/training/online loops, preprocess all 51×3, or submit GPU jobs.

## Storage model (see docs/PATHS_AND_STORAGE.md)

- Raw data live **outside** this repo. The repo never stores raw BDF.
- All paths come from `configs/paths.yaml` (or env `SHU_2C_ROOT`) — never hard-coded.
- `scripts/build_manifest.py` scans the external raw root → `manifests/shu_2c_raw_manifest.csv`.
- Python preprocessing writes `[200,58,1000]` tensors to the configured processed dir
  (default `outputs/processed_paper_style/`).

## Dataset (2C)

51 subjects (enumerate from disk), 3 sessions each, 200 trials/session (100/class),
classes left(1)/right(2). Raw 1000 Hz, 64 ch = 59 EEG + 1 ECG + 4 EOG. Processed
per session: `[trials, channels, time] = [200, 58, 1000]`. Detail: `docs/DATASET_SHU.md`.

## Main model: CAP-EEGNet (Confidence-aware Prototype EEGNet)

EEGNet encoder + Adapter + Classification head + Prototype head + Confidence head.
Detail/roadmap: `docs/MODEL_PLAN.md`.

## Main experiments (Python/PyTorch)

1. Build manifest from the external raw path.
2. Preprocess raw BDF → `[200,58,1000]`.
3. Create 41/10 **subject-wise** splits (persisted to `splits/`).
4. Train CAP-EEGNet on the 41 source subjects.
5. Zero-shot evaluate on the 10 target subjects.
6. Fine-tune on target Session 1.
7. Online test-then-update on target Session 2 and 3.
8. Ablations: prototype, confidence, adapter, online update.

Detail: `docs/EXPERIMENT_PROTOCOL.md`.

## Hard invariants

- Split by subject, never by trial. No target leakage into source training.
- Online learning is test-then-update (predict+record, THEN update).
- Never hard-code data paths; never write into the external raw dir.
- Never run heavy/GPU work on the Slurm login node.

## Working rhythm (do NOT skip ahead)

```
rules + path config + manifest (done)
  -> Python preprocess one session, confirm [200,58,1000] (done: sub-001/ses-01)
  -> finalize preprocessing across more sessions (via Slurm CPU) when ready
  -> 41/10 subject-wise splits
  -> CAP-EEGNet (encoder -> +prototype -> +confidence -> +adapter)
  -> cross-subject pretraining -> zero-shot
  -> Session-1 fine-tuning
  -> Session-2/3 online test-then-update
  -> ablations
```
