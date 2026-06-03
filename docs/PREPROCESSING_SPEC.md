# Preprocessing Spec

Two preprocessing variants. Implement **paper-style** first (no ICA). The
**EOG/ECG-clean** variant comes later. Both produce `[trials, channels, time] =
[200, 58, 1000]` float32 and `y` in {0, 1}.

Outputs:
- paper-style -> `data/processed_paper_style/sub-XXX/ses-YY/{X.npy, y.npy, meta.json}`
- eog/ecg-clean -> `data/processed_eog_ecg_clean/sub-XXX/ses-YY/{X.npy, y.npy, meta.json}`

> Some details (exact epoch window relative to cue, baseline interval, filter
> order) are not fully pinned down yet. `check_raw_bdf.py` and a single-session
> dry run must confirm them before running all 51x3. Until confirmed, the script
> must FAIL LOUDLY if it cannot produce `[200, 58, 1000]`, never silently pad/crop.

## Channel roles (VERIFIED on sub-001/ses-01)

- 64 raw channels = **59 EEG + 1 ECG (`ECG`) + 4 EOG (`HEOR/HEOL/VEOU/VEOL`)**.
  (⚠ `eeg.json` says 1 EOG + 4 ECG — swapped; trust the names.)
- Detect channel roles **by name**: EEG from `task-motorimagery_channels.tsv`,
  `ECG` by name, EOG by H/V-EOG prefix (`HEO*`/`VEO*`).
- Some aux channels may be flat (ECG, VEOL were flat in the first 10 s on
  sub-001/ses-01) — check validity over the full recording, especially for variant 2.

## Paper-style pipeline

1. **Load** raw `data.bdf` with `mne.io.read_raw_bdf(..., preload=True)`.
2. **Channel types**: set the 59 EEG channels to `eeg`, the EOG channel to `eog`,
   the 4 ECG channels to `ecg` (by name).
3. **Drop aux**: drop the ECG + EOG channels -> 59 EEG channels remain.
4. **Re-reference** to `Pz` (`raw.set_eeg_reference(ref_channels=['Pz'])`), then
   drop `Pz` -> 58 EEG channels.
5. **Band-pass** filter 0.5-40 Hz.
6. **Notch** filter at 50 Hz (PowerLineFrequency).
7. **Events**: read from `evt.bdf` / annotations; map triggers, keep MI {1, 2}.
   ⚠ OPEN: the naive `read_raw_bdf(evt.bdf).annotations` gave only `{"7","8"}`,
   not 200 markers — resolve the real trigger source first (see DATASET_SHU.md), or
   use the labels from the paper `.mat`.
8. **Epoch** the 4 s MI segment (window relative to cue — confirm exact onset/offset
   with `check_raw_bdf.py`); expect 200 epochs.
9. **Baseline correction**.
10. **Resample** to 250 Hz (1000 Hz -> 250 Hz) so each trial has 1000 samples.
11. **Save** `X = [200, 58, 1000]` float32, `y = [200]` int64 in {0, 1}, `meta.json`.

## EOG/ECG-clean pipeline (variant 2)

1. Load all 64 channels.
2. Set channel types eeg / eog / ecg correctly.
3. Check whether the ECG/EOG channels carry valid (non-flat, non-zero) signal.
4. If valid, use ECG/EOG for artifact detection/removal (e.g. regression / SSP /
   ICA-with-EOG-ECG as decided later). If invalid, log and skip cleaning.
5. Drop ECG/EOG channels after cleaning.
6. Continue with Pz re-reference, band-pass, notch, epoching, baseline, resample.

## Mandatory logging (`meta.json` + `outputs/preprocess_summary.csv`)

For every session, record:
`subject_id`, `session_id`, `variant`, `raw_sfreq`, `target_sfreq`,
`orig_n_channels`, `final_n_channels`, `n_events`, `n_trials`, `label_counts`
(per class), `output_shape`, `n_rejected_trials`, `rejection_reasons`,
`mne_version`, `random_seed`, and a `status` field (`ok` / `failed` + reason).

## Sanity checks

- Assert final shape exactly `[200, 58, 1000]`; raise on mismatch.
- Assert labels are a subset of {0, 1} and class counts are 100/100 (warn if not).
- Cross-check at least a few sessions against the paper `.mat` in `derivatives/`
  (channel count, trial count, label distribution; numerical closeness is a bonus,
  not required since pipelines differ).
- No `NaN`/`Inf` in `X`.

## Config

All knobs live in `configs/preprocess.yaml` (filter bands, notch, epoch window,
baseline interval, target sfreq, reference channel, variant flags, paths).
