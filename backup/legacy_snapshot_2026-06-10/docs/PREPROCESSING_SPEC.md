# Preprocessing Spec

Two preprocessing pipelines. The **FORMAL** output of the current stage is the
**EOG/ECG-clean** pipeline (`mode: eog_ecg_clean`). The **paper-style** pipeline
(no ICA) is retained only as a single-session sanity check. Both produce
`[trials, channels, time] = [200, 58, 1000]` float32 (µV) and `y` in {0, 1}.

Which pipeline runs is chosen by `mode` in `configs/preprocess.yaml`.

## Outputs

Output dirs come from `configs/paths.yaml` (`processed_data.*_root`).

- **FORMAL — eog_ecg_clean** (`eog_ecg_clean_root`, an EXTERNAL dir outside the repo,
  e.g. `/share/workspace2/moto_imagination/WBCIC_SHU/processed/eog_ecg_clean`):
  ONE `.npz` per session, **not** `X.npy`/`y.npy`:
  - `sub-XXX/ses-YY/sub-XXX_ses-YY_task-motorimagery_eeg.npz`
    (keys: `X` [200,58,1000] float32 µV, `y` [200] int64 {0,1}, `subject_id`,
    `session_id`, `sfreq` (=250), `channel_names` (58))
  - `meta.json`, `preprocess_report.json`, `manifest_row.json`
  - debug `X.npy`/`y.npy` ONLY if `output.save_debug_npy: true` (default OFF)
  - aggregate `processed_manifest.csv` + `preprocess_summary.csv` are written
    **beside the npz tree** (one row per session), NOT under `manifests/`.
- **legacy — paper_style** (`paper_style_root`, default `outputs/processed_paper_style`):
  `sub-XXX/ses-YY/{X.npy, y.npy, meta.json}` — sanity-check only, not the formal output.

> STATUS (2026-06-04): both pipelines IMPLEMENTED.
> - paper-style (`src/preprocessing/shu_preprocess.py`): validated on sub-001/ses-01
>   vs the paper `.mat` (labels match exactly, corr ~0.994, std 11.28 vs 11.26).
>   Faithfully follows the authors' recipe `code/pre-processed/preprocessed.m`.
> - eog_ecg_clean (`src/preprocessing/eog_ecg_clean.py` + `src/preprocessing/pipeline.py`):
>   dry-run on sub-001/002/003 (9 sessions) = 9/9 `ok` (all `[200,58,1000]`, 100/100
>   labels, no NaN/Inf). Full 51×3 run still pending user go-ahead.
> - Both FAIL LOUDLY / record a `failed` status if they cannot produce `[200,58,1000]`
>   — they never silently pad/crop. `preprocess_all.py` isolates per-session failures
>   (records `status`+`error_message`) so one bad session never aborts the full run.
>
> UNIT NOTE: the BDF physical dimension is the garbled `?V` (meant µV); MNE does not
> rescale it, so `get_data()` returns µV-magnitude values. We store µV as-is (do NOT
> multiply by 1e6). Verified: our std 11.28 ≈ paper 11.26.

## Channel roles (VERIFIED on sub-001/ses-01)

- 64 raw channels = **59 EEG + 1 ECG (`ECG`) + 4 EOG (`HEOR/HEOL/VEOU/VEOL`)**.
  (⚠ `eeg.json` says 1 EOG + 4 ECG — swapped; trust the names.)
- Detect channel roles **by name**: EEG from `task-motorimagery_channels.tsv`,
  `ECG` by name, EOG by H/V-EOG prefix (`HEO*`/`VEO*`).
- Some aux channels may be flat (ECG, VEOL were flat in the first 10 s on
  sub-001/ses-01) — check validity over the full recording before the eog_ecg_clean
  pipeline uses them for ICA (the validity gate handles this; invalid ones are skipped).

## Paper-style pipeline (legacy sanity-check, no ICA)

1. **Load** raw `data.bdf` with `mne.io.read_raw_bdf(..., preload=True)`.
2. **Channel types**: set the 59 EEG channels to `eeg`, the EOG channel to `eog`,
   the 4 ECG channels to `ecg` (by name).
3. **Drop aux**: drop the ECG + EOG channels -> 59 EEG channels remain.
4. **Re-reference** to `Pz` (`raw.set_eeg_reference(ref_channels=['Pz'])`), then
   drop `Pz` -> 58 EEG channels.
5. **Band-pass** filter 0.5-40 Hz.
6. **Notch** filter at 50 Hz (PowerLineFrequency).
7. **Events**: parse `evt.bdf` TAL via `src/preprocessing/neuracle_events.py`
   (MNE's reader misses them). Keep MI codes {1, 2}; onset_seconds * sfreq -> sample.
8. **Epoch** `[0, 4)` s relative to each trigger (tmin=0, tmax=4-1/sfreq -> 4000
   samples @1000Hz); expect 200 epochs.
9. **Baseline correction**: whole-epoch demean (MNE `baseline=(None, None)`, matching
   EEGLAB `pop_rmbase`).
10. **Resample** to 250 Hz (1000 Hz -> 250 Hz) so each trial has 1000 samples.
11. **Save** `X = [200, 58, 1000]` float32, `y = [200]` int64 in {0, 1}, `meta.json`.

## EOG/ECG-clean pipeline (FORMAL — `mode: eog_ecg_clean`)

Implemented in `src/preprocessing/eog_ecg_clean.py` (cleaning) +
`src/preprocessing/pipeline.py` (save + manifest row). All knobs under
`aux_cleaning.*` in `configs/preprocess.yaml`.

1. Load all 64 channels; identify EEG / EOG / ECG **by name** (EEG = the rest).
2. **Validate aux channels**: each EOG/ECG channel must be non-flat / non-zero /
   no NaN/Inf (`std >= min_std_uv`, `ptp >= min_ptp_uv`) to be used. Invalid ones
   are recorded and skipped.
3. Set channel types (`eog`/`ecg`/`eeg`) so `picks="eeg"` is exactly the EEG set.
4. **ICA cleaning** (only if enabled, method=`ica`, and >=1 valid aux channel):
   - fit ICA on a **1 Hz high-passed copy**, EEG only (`n_components`=0.99, FastICA,
     `decim`, fixed `random_state`); refit with an integer `n_components_fallback`
     if the float ratio collapses.
   - detect ocular ICs via `find_bads_eog` and cardiac ICs via `find_bads_ecg`,
     both by **absolute correlation** (`|corr|` > threshold, default 0.5);
   - exclude the hit components and `apply` back to the EEG (aux channels untouched);
   - record excluded IC ids, top component scores, channels used, and any warnings.
   - **Any ICA/detection failure degrades to no-aux-clean** (recorded, never crashes).
5. **Paper-style second half**: drop aux (→59 EEG) → re-reference `Pz` and drop it
   (→58 EEG) → 0.5-40 Hz band-pass + 50 Hz notch → parse `evt.bdf` TAL (1→0, 2→1) →
   epoch `[0,4)` s (whole-epoch demean) → resample 250 Hz → `[200,58,1000]`.
6. **Quality checks + status**: shape/dtype/channel/time/sfreq, 100/100 label balance,
   no NaN/Inf, trigger count == 200, and (when the paper `.mat` exists)
   `labels_multiset_match`. Save `.npz` + `meta.json` + `preprocess_report.json`
   (+ `manifest_row.json`); build a processed-manifest row.

## Mandatory logging (`meta.json` / `preprocess_report.json` + `preprocess_summary.csv`)

`preprocess_summary.csv` and `processed_manifest.csv` are written **beside the npz
tree** (under `eog_ecg_clean_root`), one row per session. For every session, record:
`subject_id`, `session_id`, `mode`, `raw_sfreq`, `target_sfreq`,
`orig_n_channels`, `final_n_channels`, `n_events`, `n_trials`, `label_counts`
(per class), `output_shape`, `mne_version`, the aux-cleaning block
(`aux_cleaning_used`, `valid_eog_channels`, `valid_ecg_channels`,
`ica_excluded_components`, component scores, warnings), the `.mat` cross-check
(`labels_match_mat`, `labels_multiset_match`, `n_labels_agree`), and a `status`
field (`ok` / `failed` + `fail_reasons` / `error_message`).

## Sanity checks

- Assert final shape exactly `[200, 58, 1000]`; raise on mismatch.
- Assert labels are a subset of {0, 1} and class counts are 100/100 (warn if not).
- Cross-check at least a few sessions against the paper `.mat` in `derivatives/`
  (channel count, trial count, label distribution; numerical closeness is a bonus,
  not required since pipelines differ).
- No `NaN`/`Inf` in `X`.

## Config

All preprocessing knobs live in `configs/preprocess.yaml` (`mode`, `output.*`,
`aux_cleaning.*` incl. ICA params + thresholds, filter bands, notch, epoch window,
baseline, target sfreq, reference channel, `expect_shape`). All filesystem paths
(raw root, `processed_data.*_root`, manifests) live in `configs/paths.yaml`.
