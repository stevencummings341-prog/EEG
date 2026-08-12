---
title: "EEG Foundation Models (S4 / DINO-DualCD)"
tags:
  - "#pipeline/5_dl_model"
  - "#modality/eeg"
created: "2026-08-04"
updated: "2026-08-04"
status: "active"
---

# EEG Foundation Models — S4 / DINO-DualCD family

## 1. Purpose

The 5 end-to-end models of the new mainline (advisor's re-scope, 2026-08-04): drop the
online/TTA line for now and train these architectures end to end on WBCIC-SHU and
SHU 2022 **separately**. Ported from the advisor-supplied `models_eeg_foundation/`
package and wrapped so they obey this project's model contract.

Parameter counts below are **measured** with this project's model YAMLs at `n_times=1000`,
`n_classes=2` (script: build each variant and sum `p.numel()`).

| registry name | backbone | pooling | DualCD | params @58ch (WBCIC) | params @32ch (SHU) | feature_dim | default lr |
|:---|:---|:---|:---:|---:|---:|---:|---:|
| `s4erp` | S4 | flatten | no | 1,370,882 | 944,898 | 62336 | 1e-3 |
| `dualcd_s4_pos` | S4 | attention | yes | 3,168,772 | 2,316,804 | 128 | 1e-4 |
| `dualcd_s4_timepatch` | S4 | temporal bin | yes | 4,480,386 | 3,628,418 | 1536 | 1e-4 |
| `dualcd_s4_flatten` | S4 | flatten | yes | 66,861,186 | 66,009,218 | 62336 | 1e-4 |
| `dualcd_transformer` | Transformer | flatten | yes | 67,900,034 | 67,048,066 | 62336 | 1e-4 |

`feature_dim` at `n_times=1000` follows `patch_num = (1000 - 28) // 2 + 1 = 487` from the
ShallowNet stem (conv kernel 25, avg-pool 4 stride 2). The flatten variants are dominated
by their DINO projection heads (`62336 x 512`), which is why they cost ~66M parameters
while the S4 backbone itself is tiny.

**Against the advisor's table** (which quotes 906K / 1.99M / 3.30M / 65.8M / 66.8M): the two
flatten variants and the Transformer match to within 0.4% at 32ch/1000/2C, so that table was
computed at our trial length. `dualcd_s4_pos` and `dualcd_s4_timepatch` come out 10-16% higher
here (2.32M vs 1.99M, 3.63M vs 3.30M) — those two rows of the table look like they were
measured under a different configuration (the package's `comparison.md` lists 1.85M / 3.27M for
an ERP setup with C=21, T=170). Nothing is wrong with the port; the numbers above are simply
the measured truth for our two datasets, and `dualcd_s4_timepatch` additionally depends on how
many temporal bins the config asks for.

## 2. What Belongs Here

| file | role |
|:---|:---|
| `s4_layers.py` | S4 layer/block/encoder: HiPPO-LegS init + FFT convolution, pure PyTorch. |
| `pooling.py` | `FlattenPooling` / `AttentionPooling` / `TemporalBinnedPooling`. |
| `encoders.py` | `ShallowNetEmbedding` CNN stem + S4 / Transformer encoder stacks. |
| `losses.py` | `DINOLoss`, `IBOTHead`, `DKoleoLoss`, `ProjectionHead`, `PrototypeBank`, `OrthogonalMask`, intra/inter-class perturbation. |
| `models.py` | The 5 model variants + `MultiViewGenerator`. |
| `adapter.py` | **Project-facing wrapper**: `EEGFoundationConfig` / `EEGFoundationClassifier` / `build_eeg_foundation` / `normalize_trials`. |

Experiments must never import `models.py` directly — go through
`code.models.registry.build_model("<name>", n_channels=..., n_times=..., n_classes=...,
sfreq=..., params=...)`, so the `[B, C, T]` input convention and the
`{logits, features, confidence}` output contract hold everywhere.

## 3. How the wrapper bridges the two dialects

| topic | source package | this project | handled by |
|:---|:---|:---|:---|
| input layout | `(B, T, C)` | `[B, C, T]` (or `[B, 1, C, T]`) | `adapter._as_time_last` |
| config fields | `num_channels/num_classes/seq_len/sampling_rate` | `n_channels/n_times/n_classes/sfreq` | `EEGFoundationConfig` properties |
| output | `S4ERP` -> dict, DualCD -> bare logits tensor | `{logits, features, confidence}` | `EEGFoundationClassifier.forward` |
| DualCD training | `compute_loss` + `update_ema` + `update_prototypes` | trainer-agnostic hooks | `uses_custom_loss` / `training_step` / `after_optimizer_step` |
| normalization | done by hand in the training script | fit-free per trial, applied at load time | `normalize_trials` (called by `cross_subject_protocols`) |

`features` is the pooled representation **before** the causal/spurious mask — the same
vector `encode()` returns — so it is directly comparable to the penultimate features of
EEGNet / DeepConvNet / FBCNet used in Phase 2c.

## 4. Porting notes (every deviation from the advisor's package)

The model math is unchanged. The differences, all deliberate:

1. **`losses.py`**: imported `Tuple` (the file annotated `OrthogonalMask.forward -> Tuple[...]`
   without importing it; harmless under `from __future__ import annotations`, fatal for any
   tool that resolves type hints).
2. **`models.py` `MultiViewGenerator`**: the two band-pass student views were hard-coded to
   4-12 Hz and 12-30 Hz, while the package's own README told users to set
   `model.multi_view.low_freq` — an attribute nothing read. They are now real constructor
   arguments (`low_band` / `high_band`, defaults unchanged) that `generate()` actually uses,
   threaded from the model config via `view_low_band` / `view_high_band`. **Our MI configs
   set mu/beta (8-13 / 13-30 Hz)**, which is the adaptation the package's docs recommend for
   motor imagery.
3. **`models.py`**: the fixed noise level of the `global_b` teacher view is now
   `noise_std` (default 0.1, unchanged).
4. **Unused imports** removed (`Any`, `Optional` in `encoders.py`; `Any`, `Optional` in
   `models.py`) and provenance headers added.
5. **`train_template.py` was NOT ported.** Its job is done by
   `code/training/e2e_trainer.py` (resumable, best+last checkpoints only) plus
   `code/experiments/cross_subject_protocols.py` (subject-level splits, leakage guards,
   metrics). The template's `.npy` loading, 70/15/15 random split, and single-directory
   output do not fit this project's manifest-driven, subject-grouped protocol.
6. **`README.md` / `USAGE_GUIDE.md` / `comparison.md` were not copied verbatim**; their
   substance is folded into this file, the model YAMLs, and the route plan. The originals
   stay in the advisor's package until the user deletes it.

### Config-level adaptations to motor imagery (not code changes)

* **`dualcd_s4_timepatch` bins must be set.** The package default is the ERP window
  0-750 ms; our trials are 4 s (`[0, 4)` s after the MI cue @ 250 Hz), so the default
  would dump ~90% of the 487 patches into the last bin. Our configs use
  `[0, 500, 1000, 1500, 2000, 3000, 4000]` ms (6 bins, denser early where ERD develops).
* **Learning rate** defaults to the package's recommendation: 1e-3 for `s4erp`, 1e-4 for the
  DualCD variants (`adapter.DEFAULT_LR`, used when the experiment config leaves `train.lr`
  null).

## 5. Update Rules

* New variant -> add to `adapter.VARIANTS` + a `code/configs/models/<name>.yaml` + this table
  + `0_docs/FILE_CATALOG.md`.
* Changing anything inside `models.py` / `losses.py` / `s4_layers.py` / `pooling.py` /
  `encoders.py` means diverging from the advisor's package: record it in §4 above.
* Contract tests: `tests/foundation/test_eeg_foundation_contract.py` (19 cases, CPU).

## 6. Related Files

- `AGENTS.md`: authoritative project memory.
- `FOUNDATION_E2E_ROUTE_PLAN.md`: the new mainline route + protocol status.
- `code/training/e2e_trainer.py`: resumable trainer that drives the DualCD hooks.
- `code/experiments/cross_subject_protocols.py`: the cross-subject protocol.
- `code/configs/experiments/{foundation_cross_subject,shu_foundation_cross_subject}.yaml`.
