# PROGRESS / Memory Log

The project's running memory. **Append a dated entry whenever you finish a
meaningful step or make a design decision.** Newest entries on top. Keep it factual.

Format per entry: date, what was done, decisions made, open questions, next step.

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
  Not changed automatically — needs user decision (`docs/ENVIRONMENT.md`).
- ⚠️ **Event triggers**: `evt.bdf` annotations via MNE gave only `{"7":1,"8":1}`
  (channel "Empty Event Data"), NOT 200 MI markers. Must find the real trigger
  source before from-raw preprocessing (alternate evt parse / Neuracle reader /
  fall back to labels in the paper `.mat`). This is the #1 blocker for Task 2.
- Confirm cue timing for the 4 s epoch window + baseline interval (paper detail TBD).
- Check whether the flat ECG/VEOL is session-specific or dataset-wide.

**Next step**
- Run `scripts/check_raw_bdf.py` on `sub-001/ses-01` (login-node-safe: reads one
  file, prints + dumps JSON), read the report, then implement `preprocess_raw.py`
  for a single session and confirm the `[200, 58, 1000]` shape.
