# Paths and Storage

## Key Principle

The raw SHU MI-EEG data are stored **outside** this project directory.

This repository contains:
- code, configs, rules, docs, manifests, splits, logs, outputs, checkpoints.

This repository does NOT contain:
- raw BDF files;
- large processed datasets unless explicitly configured.

## Required Path Config

Edit `configs/paths.yaml` and set `raw_data.shu_2c_root` to the external server path
of the SHU 2C dataset. On this server it is already filled with the verified location:

```yaml
raw_data:
  shu_2c_root: "/share/workspace2/moto_imagination/WBCIC_SHU"
  raw_subdir: "sourcedata/2C dataset"
  derivatives_subdir: "derivatives/2C dataset_processeddata"
processed_data:
  # paper_style_root: legacy single-session sanity check only (X.npy/y.npy).
  paper_style_root: "outputs/processed_paper_style"
  # eog_ecg_clean_root: FORMAL output of the current stage (one .npz per session).
  # External, outside the repo — never under sourcedata/ or derivatives/.
  eog_ecg_clean_root: "/share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean"
manifests:
  raw_manifest: "manifests/shu_2c_raw_manifest.csv"
  # Formal processed manifest sits NEXT TO the npz outputs (records npz_path).
  processed_manifest: "/share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean/processed_manifest.csv"
splits:
  dir: "splits"
```

> Config keys are `*_root` (the old `*_dir` keys are still accepted by the loader for
> back-compat). Relative paths resolve against the project root; absolute paths (e.g.
> the external `processed/` subtree above) are used as-is.

Rules:
- Never hard-code raw paths in Python. Load via `src/utils/paths.load_paths()`.
- Env var `SHU_2C_ROOT` overrides `raw_data.shu_2c_root`.
- Relative paths resolve against the project root; absolute paths (scratch) are used as-is.
- If the raw path is missing / a placeholder / nonexistent, the loader raises a clear
  error asking you to fill `configs/paths.yaml`. Do not guess paths.
- Never write into the external raw data directory.

## Manifest (the bridge between external raw data and this project)

After setting the raw path:

```bash
python scripts/build_manifest.py --config configs/paths.yaml
```

This creates `manifests/shu_2c_raw_manifest.csv` with columns:
`subject_id, session_id, data_bdf_path, evt_bdf_path, data_bdf_exists,
evt_bdf_exists, data_bdf_size_bytes`.

Verified on this server: 51 subjects, 153 sessions, 0 missing BDFs.

## Processed Data

The **formal** output (`mode: eog_ecg_clean`) is ONE `.npz` per session written under
`eog_ecg_clean_root` (an external `processed/` subtree, NOT under the repo):

```
<eog_ecg_clean_root>/sub-XXX/ses-YY/
  sub-XXX_ses-YY_task-motorimagery_eeg.npz   # X[200,58,1000] f32 µV, y[200] i64,
                                             # subject_id, session_id, sfreq, channel_names
  meta.json
  preprocess_report.json
  manifest_row.json
<eog_ecg_clean_root>/processed_manifest.csv    # one row per session (records npz_path)
<eog_ecg_clean_root>/preprocess_summary.csv    # per-session status / shape / aux info
```

`scripts/preprocess_all.py` aggregates `processed_manifest.csv` + `preprocess_summary.csv`
**beside the npz tree** (NOT under `manifests/`). Debug `X.npy`/`y.npy` are written only
when `output.save_debug_npy: true`. The legacy `paper_style` mode still writes
`{X.npy, y.npy, meta.json}` under `paper_style_root` for single-session sanity checks.
For large-scale runs, point `processed_data.*_root` at a roomy scratch/workspace dir.

The paper-processed `.mat` in `derivatives/` is used ONLY as a label/event
cross-check (`validate_against_mat: true`), never as the training data entry point.

## Splits

Subject-wise 41/10 splits are persisted to `splits/<run_id>_seed<k>.json` for
reproducibility (see `src/data/splits.py`).

## What is git-ignored

Generated/large artifacts: `outputs/*`, `logs/*`, `checkpoints/*`,
`manifests/*.csv`, `splits/*.json`, and binary data (`*.bdf,*.mat,*.npy,*.pt,...`).
Directory structure is kept via `.gitkeep`.
