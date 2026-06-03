# Project Brief: SHU MI-EEG Cross-subject Pretraining and Online Adaptation

> 本项目的核心思想来源于学长的聊天记录
> `docs/references/ChatGPT-EEG-MI-pretraining.md`，请把它当作"项目宗旨"。
> 本文件是给 Cursor 和开发者看的总说明书，比 `.cursor/rules` 更详细。

## Goal

Build a Python pipeline for the Shanghai University **WBCIC-SHU** motor-imagery
(MI) EEG dataset. Start from raw BDF data, do preprocessing in Python, reproduce
baseline decoding models, then study **cross-subject pretraining**,
**target-subject fine-tuning**, and **online adaptation** — with explicit
**sample confidence prediction** throughout.

The eventual system (the senior's vision) is:

> **Confidence-aware Online Adaptive Multi-Subagent Pretraining Framework for
> Cross-subject MI EEG Decoding**
> 置信度感知的在线自适应多子模块运动想象 EEG 预训练框架

## Environment & server (verified)

- Dataset root (READ-ONLY): `/share/workspace2/moto_imagination/WBCIC_SHU`.
- Project root: `/share/home/yuan/SYX/eeg-mi-online`. Never write outside `/share/home/yuan/SYX`.
- Conda env: `mi_torch` (python 3.10, torch 2.6, mne 1.10). NOTE: torch is currently
  a **CPU-only** build — see `docs/ENVIRONMENT.md`.
- Slurm cluster, partitions `gpu2node`/`gpu3node`. Never train on the login node.
  See `docs/SERVER_RUNBOOK.md`.

## Dataset (focus: 2C dataset)

- 51 subjects on disk (tsv lists 52, README says 53 — enumerate from disk).
- 3 sessions per subject.
- 2 classes: left-hand grasping MI (trigger 1) vs right-hand grasping MI (trigger 2).
- 200 trials per session, 100 per class.
- Raw files: `data.bdf` + `evt.bdf`; raw 1000 Hz, 64 ch (59 EEG + 1 EOG + 4 ECG).
- Final processed shape per session: `[trials, channels, time] = [200, 58, 1000]`
  (58 EEG after dropping ECG/EOG and Pz; 1000 = 4 s @ 250 Hz).

Full detail: `docs/DATASET_SHU.md`.

## Main stages

### Stage 1: Raw data preprocessing
- Read raw BDF (MNE), inspect ECG/EOG quality.
- Paper-style preprocessing (no ICA first).
- Optional ECG/EOG-assisted artifact cleaning (second variant).
- Save processed trials to `data/processed_paper_style/`.
- Spec: `docs/PREPROCESSING_SPEC.md`.

### Stage 2: Baseline reproduction
- Train EEGNet, DeepConvNet, FBCNet.
- Purpose: verify preprocessing correctness and establish baseline numbers.

### Stage 3: 41/10 cross-subject pretraining
- 41 source subjects vs 10 target subjects, split by subject.
- Train on all source sessions; zero-shot test on all target sessions.
- Repeated random subject-wise splits; report mean +/- std.

### Stage 4: Target-subject fine-tuning
- For each target subject: fine-tune on Session 1, test on Session 2 + 3.
- Compare zero-shot / classifier-only / adapter / prototype / full-model.

### Stage 5: Online learning
- Initialize from the pretrained model (optionally after Session 1 fine-tuning).
- Online test-then-update on Session 2 then Session 3.
- Per trial: predict -> output confidence -> record -> update lightweight modules.

Full detail: `docs/EXPERIMENT_PROTOCOL.md`.

## First main model

First version (keep it minimal, expand later):
- EEGNet-style encoder backbone.
- Classification head.
- Prototype memory (global / subject / session prototypes).
- Confidence head (multi-source, not just softmax max).
- Adapter for fine-tuning / online adaptation.

Design and roadmap: `docs/MODEL_PLAN.md`.

## Research logic (paper framing)

- Paper 1 candidate: cross-subject MI-EEG pretraining with prototype + confidence learning.
- Paper 2 candidate: confidence-guided online adaptation for multi-session MI-EEG decoding.

## Working rhythm (do NOT skip ahead)

```
rules + docs scaffold (this commit)
  -> scripts/check_raw_bdf.py runs on one subject/session
  -> scripts/preprocess_raw.py runs on one subject/session
  -> confirm X == [200, 58, 1000]
  -> scripts/preprocess_all.py over 51 x 3 (via Slurm CPU job)
  -> EEGNet baseline (single session, then subject-wise)
  -> 41/10 split pretraining
  -> prototype + confidence
  -> Session 1 fine-tuning
  -> Session 2/3 online learning
```

The single most important constraint: **let Cursor understand the project
boundaries first; otherwise it will jump straight to building a giant model.**
