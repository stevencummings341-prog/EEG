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
  paper_style_dir: "outputs/processed_paper_style"
  eog_ecg_clean_dir: "outputs/processed_eog_ecg_clean"
manifests:
  raw_manifest: "manifests/shu_2c_raw_manifest.csv"
  processed_manifest: "manifests/shu_2c_processed_manifest.csv"
splits:
  dir: "splits"
```

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

Python preprocessing writes to the configured processed dir (default
`outputs/processed_paper_style/sub-XXX/ses-YY/{X.npy, y.npy, meta.json}`), and a
`manifest_row.json` per session. `scripts/preprocess_all.py` will aggregate into
`manifests/shu_2c_processed_manifest.csv`. For large-scale runs, point
`processed_data.*` at a scratch directory.

The paper-processed `.mat` in `derivatives/` is used ONLY as a label/event
cross-check (see `preprocess_raw.py --config ... validate_against_mat: true`), never
as the training data entry point.

## Splits

Subject-wise 41/10 splits are persisted to `splits/<run_id>_seed<k>.json` for
reproducibility (see `src/data/splits.py`).

## What is git-ignored

Generated/large artifacts: `outputs/*`, `logs/*`, `checkpoints/*`,
`manifests/*.csv`, `splits/*.json`, and binary data (`*.bdf,*.mat,*.npy,*.pt,...`).
Directory structure is kept via `.gitkeep`.
