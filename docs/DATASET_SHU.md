# Dataset: WBCIC-SHU Motor Imagery EEG

This documents the facts about the dataset **as verified on this server**. When a
fact here disagrees with an external description, trust this file + an inspection
script, not memory.

## Source

- 2019 World Robot Conference Contest — BCI Robot Contest MI.
- Institution: Shanghai University (Banghua Yang, Fenqi Rong).
- BIDS-formatted. License: ODC-BY (see dataset `LICENSE`).
- Root on this server (READ-ONLY): `/share/workspace2/moto_imagination/WBCIC_SHU`.

## Layout

```
WBCIC_SHU/
├── README.md
├── dataset_description.json
├── participants.json
├── participants_2C.tsv          # 2-class participants (lists sub-001 .. sub-052)
├── participants_3C.tsv          # 3-class participants
├── task-motorimagery_channels.tsv      # 59 EEG channel names + types
├── task-motorimagery_eeg.json          # sampling rate, channel counts, etc.
├── task-motorimagery_events.json       # event value -> class mapping
├── task-motorimagery_electrodes.tsv
├── task-motorimagery_coordsystem.json
├── sourcedata/
│   └── 2C dataset/                     # NOTE: literal space in the name
│       └── sub-XXX/ses-YY/eeg/{data.bdf, evt.bdf}
├── derivatives/
│   └── 2C dataset_processeddata/
│       └── sub-XXX/ses-YY/eeg/sub-XXX_ses-YY_task-motorimagery_eeg.mat
└── code/                               # authors' reference code (MNE 0.22 / torch 1.10)
    ├── Deep_learning/  (FBCNet, EEGNet, DeepConvNet)
    ├── Machine_learning/  (CSP, FBCSP)
    └── transform.py
```

- `sub-XXX`: zero-padded to 3 digits (`sub-001` ... `sub-051`).
- `ses-YY`: zero-padded to 2 digits (`ses-01`, `ses-02`, `ses-03`).
- **51** subject folders exist under `sourcedata/2C dataset/`. `participants_2C.tsv`
  lists 52 ids and the README mentions 53 two-class participants. **Always
  enumerate subjects from the filesystem; never hardcode 51/52/53.**

## Recording parameters (`task-motorimagery_eeg.json`)

| Field | Value |
| --- | --- |
| TaskName | Motor imagery |
| SamplingFrequency | 1000 Hz (raw) |
| Manufacturer | neuracle |
| EEGChannelCount | 59 |
| EOGChannelCount | 1 (⚠ see note) |
| ECGChannelCount | 4 (⚠ see note) |
| Total channels | 64 |

> ⚠ **Metadata vs reality**: the JSON above says 1 EOG + 4 ECG, but the actual BDF
> channel names (verified) are 1 ECG + 4 EOG. The JSON EOG/ECG counts are swapped.
> Trust the channel names below.
| PowerLineFrequency | 50 Hz |
| EEGReference | REF |
| EEGPlacementScheme | 10-20 |

## Channels (VERIFIED via `scripts/check_raw_bdf.py` on sub-001/ses-01)

Actual 64-channel order in the BDF:

- **59 EEG** (`task-motorimagery_channels.tsv`):
  Fpz, Fp1, Fp2, AF3, AF4, AF7, AF8, Fz, F1..F8, FCz, FC1..FC6, FT7, FT8, Cz,
  C1..C6, T7, T8, CP1..CP6, TP7, TP8, **Pz**, P3..P8, POz, PO3..PO8, Oz, O1, O2.
- **1 ECG**: `ECG`.
- **4 EOG**: `HEOR`, `HEOL` (horizontal), `VEOU`, `VEOL` (vertical).

Notes:
- This is **1 ECG + 4 EOG**, the opposite of the `eeg.json` counts (see warning above).
- Aux channels can be flat/unused: on sub-001/ses-01 the first 10 s of `ECG` and
  `VEOL` were all-zero, while `HEOR/HEOL/VEOU` were active. Always validate aux
  channels over the full recording before relying on them.
- `Pz` is used as the re-reference and then dropped, giving **58 EEG** channels in
  processed data.

## Events / triggers — OPEN QUESTION

- Reading `evt.bdf` with `mne.io.read_raw_bdf(...).annotations` returned only
  `{"7": 1, "8": 1}` (channel labelled "Empty Event Data") on sub-001/ses-01 — NOT
  the 200 MI trial markers. The real per-trial triggers (values 1/2) must be located
  another way before from-raw preprocessing (alternate evt parse, Neuracle-specific
  reader, or use the labels already stored in the paper `.mat`).
- Session duration is 2250 s (~37.5 min) for 200 trials (~11.25 s/trial), consistent
  with cue + 4 s MI + rest.

## Events (`task-motorimagery_events.json`)

| value | trial_type | used in 2C? | internal label |
| --- | --- | --- | --- |
| 1 | left-hand | yes | 0 |
| 2 | right-hand | yes | 1 |
| 3 | foot | only 3C | n/a |

For the 2C dataset only values {1, 2} appear.

## Trials / sessions

- 3 sessions per subject; 200 trials per session; 100 per class.

## Paper-processed `.mat` (derivatives) — verified

- One `.mat` per subject/session: keys `data` and `labels`.
- `data` shape `[channels, time, trials] = [58, 1000, 200]`, dtype float32.
- `labels` shape `[1, 200]`, values {1, 2}.
- Authors' `transform.py` does `data.transpose(2, 0, 1)` -> `[200, 58, 1000]`
  and `np.ravel(labels)`. This matches our target convention
  `[trials, channels, time]` (we additionally remap labels {1,2} -> {0,1}).
- These `.mat` files are useful as a ground-truth cross-check when validating our
  own from-raw preprocessing.

## Demographics (`participants_2C.tsv`)

Columns: `participant_id`, `gender` (M/F), `age` (years), `handedness` (all R in 2C).
