---
title: "Pretrained Model Integration Contract"
tags:
  - "#method/test_time_adaptation"
  - "#contract"
created: "2026-07-12"
updated: "2026-07-12"
status: "authoritative_contract"
---

# Pretrained Model Integration Contract

**Canonical location:** `3_online_adaptation/PRETRAINED_MODEL_INTEGRATION_CONTRACT.md`

This document is the **authoritative handoff contract** between the senior / model
owner and the Phase 3 TTA backend (`code/tta/`). It defines what must be delivered
for a real pretrained checkpoint to be wired through a thin `ModelAdapter`.

## Honesty / current status (do not overclaim)

| Phrase | Meaning (2026-07-12, Pretrained-Model Readiness Round) |
|:---|:---|
| **Round-1 scaffold complete** | `code/tta/` + `phase3_tta` runner + embedding-replay smoke path exist. |
| **mock live-model path validated** | `ModelInferenceSource` is a real checkpoint→forward→`FeatureBundle` path, proven by test-only mock adapters under `tests/tta/support/` (Profiles A/B/C). **Not** a senior/research pretrained model. |
| **WBCIC full A0 complete** | All 4320 canonical Phase 2c cells: no_tta replay `|Δ|=0` (see `4_experiments/wbci_shu/tta/reports/FULL_A0_REPLAY_VALIDATION_REPORT.md`). |
| **SHU replay smoke passed** | eegnet/seed0, 2 cells, no_tta only; paths under `outputs/experiments/shu/`; `|Δ|<1e-6`. Not a scientific SHU conclusion. |
| **real pretrained model not yet integrated** | No senior / foundation checkpoint is registered as a production adapter. |
| **formal Oracle decision not run** | Minimal Oracle diagnostic may exist in smoke; Phase 3B formal decision gate has **not** been run. |
| **full T3A not run** | Only `t3a_minimal` smoke variant; not a full ablation / sweep. |

**Hard honesty rules**

- Do **not** claim compatibility with arbitrary models.
- Do **not** call mock / baseline example adapters a pretrained research model.
- `embedding_only`, `eegnet` / `deepconvnet` / `fbcnet` / `baseline_torch` registry names are **fixtures or project baselines**, not senior-model integration.
- This contract does **not** assert that T3A works or that cross-session drop is fixed.

---

## 5.1 Required deliverables from senior / model owner

Deliver all of the following with the checkpoint package (or an equivalent signed
sidecar README). Missing items block preflight.

| # | Deliverable | Notes |
|:---|:---|:---|
| 1 | **Model architecture entry** | How to construct the network (class / factory / repo path / `build_*` call). Enough to instantiate without guessing. |
| 2 | **Checkpoint file(s)** | Exact path(s) or download URI; one primary weights file preferred. |
| 3 | **Checkpoint format** | e.g. `torch.save` dict with `state_dict` / `model_state_dict`, safetensors, etc. Document nesting keys. |
| 4 | **`state_dict` key map** | Exact key names (or a documented remapping). Call out prefix strips (`module.`, encoder vs head). |
| 5 | **Model config** | Hyperparameters needed to build the net (channels, depth, hidden dims, dropout, …). Prefer a YAML/JSON sidecar. |
| 6 | **Label mapping** | Integer class id → semantic label (e.g. `0=left`, `1=right`). Must match Phase 2c / manifest convention for the target dataset. |
| 7 | **Expected input shape** | Prefer `[B, C, T]` or `[C, T]` with explicit dims. |
| 8 | **Channel names / order** | Ordered list matching training montage (WBCIC 58ch vs SHU 32ch are **not** interchangeable). |
| 9 | **`sfreq`** | Sampling rate used at train time (Hz). |
| 10 | **Time length** | Samples or seconds per trial window (`n_times` / duration). |
| 11 | **Signal unit / scale** | e.g. µV, Volt, already z-scored; any clip / gain. |
| 12 | **Preprocessing recipe** | Band-pass, notch, referencing, trial crop, artifact policy — enough to reproduce train-time inputs. |
| 13 | **Normalization** | Per-trial / per-channel / BN-stats / none; running stats if required. |
| 14 | **Feature extraction point** | Which tensor is the TTA feature `z` (layer name / hook / return key). Must be frozen-encoder output usable for prototype TTA. |
| 15 | **Feature dim `D`** | Integer `embedding_dim` / `feature_dim`. |
| 16 | **Logits format** | Shape `[B, n_classes]`, pre-softmax scores (not calibrated temperatures unless documented). |
| 17 | **`n_classes`** | Integer; MI mainline is 2 unless explicitly otherwise. |
| 18 | **Checkpoint ↔ data mapping** | Which **dataset / subject / session / seed** (and protocol) this checkpoint was trained on. Required for cell_id alignment with Phase 2c. |
| 19 | **PyTorch / dependency versions** | `torch`, CUDA if any, and other packages that affect load/forward. |

Without (14)–(15) and a way to obtain target features (live forward or precomputed
npz), **minimal T3A cannot run**. Without pred/logits/probs on the target stream,
**`no_tta` cannot run**.

---

## 5.2 Optional capabilities

These improve init / diagnostics / future catalog methods. Adapters may leave them
unimplemented and raise `UnsupportedCapabilityError` (subclass of
`UnsupportedAdapterFeature`). Production callers must fail-fast, not silent-default.

| Capability | Typical use |
|:---|:---|
| **Classifier weights** `get_classifier_weights()` → `[C, D]` | Paper-style T3A `clf_weights` init (**catalog candidate**; Round-1 `t3a_minimal` does **not** implement this init yet). |
| **Source prototypes** `get_source_prototypes()` | Precomputed class means; else compute from `FeatureBundle.source_features` + `source_labels`. |
| **Probabilities** `predict_proba()` | Softmax / calibrated probs; can satisfy `no_tta` via `target_probs`. |
| **Calibration** | Temperature / Platt params if logits are not raw. |
| **Preprocess object** | Serializable transform matching train-time pipeline. |
| **Tokenizer / patcher** | For foundation / token models only; document patch size and time alignment. |
| **Mask / attention mask** | Variable tokens or bad-channel masks. |
| **AMP / dtype policy** | fp16/bf16 rules if live GPU forward is used. |
| **Variable-length trials** | Padding / packing contract if `T` is not fixed. |
| **Intermediate layers** | Extra hooks for analysis; not required for Round-1 methods. |

---

## 5.3 Adapter capability matrix (actual Round-1 semantics)

Source of truth: `code/tta/adapters/base.py`, `methods/no_tta.py`,
`methods/t3a_minimal.py`, `feature_sources/*`.

Methods consume a **`FeatureBundle`** (replay or future live fill). They must **not**
import concrete model classes. Live models enter only via a registered `ModelAdapter`
(+ future `ModelInferenceSource`).

| Capability / field | Required for `no_tta` | Required for `t3a_minimal` | Optional / notes |
|:---|:---|:---|:---|
| `target_pred` **or** `target_logits` **or** `target_probs` | **Yes** (any one; priority pred → logits → probs) | No | Live path: `forward_logits` / `predict_proba` can populate these. |
| `target_features` `[Nt, D]` | No | **Yes** | Live path: `forward_features`. |
| `source_features` + `source_labels` | No | **Yes** when `initialization=src_proto` (Round-1 smoke default) | Not needed for `zeros` / `target_support_only` init. |
| `forward_features` (adapter) | No on embedding-replay path | Required once live inference replaces npz | `EmbeddingOnlyAdapter` correctly refuses forward. |
| `forward_logits` (adapter) | Needed for live `no_tta` if preds not precomputed | No for T3A core | |
| `predict_proba` | Alternate for `no_tta` | No | |
| `get_classifier_weights` | No | **Optional** (future `clf_weights` init; **not** in Round-1 `t3a_minimal`) | |
| `get_source_prototypes` | No | **Optional** (else mean from source features/labels) | |
| `get_feature_dim` / `embedding_dim` | Strongly recommended | Strongly recommended | Fail-fast shape checks. |
| `load_checkpoint` | For any live weights path | Same | |
| `validate_input_shape` | Strongly recommended for live | Same | |
| `get_model_metadata` | **Strongly recommended** | **Strongly recommended** | channels, sfreq, n_times, preprocess tags, dataset mapping. |
| Checkpoint ↔ cell metadata | **Strongly recommended** | **Strongly recommended** | Aligns `cell_id` with Phase 2c metrics. |

**Path notes**

- **Embedding replay (default runner path):** `EmbeddingReplaySource` fills the bundle from Phase 2c npz; methods do not need live forward. Used by smoke / full A0.
- **Live inference (mock-validated):** `ModelInferenceSource` loads checkpoint via adapter, runs `eval()` + `torch.inference_mode()`, batches forward, validates shapes, returns `FeatureBundle`. Proven with test fixtures only; wire a real senior adapter + config before claiming pretrained integration.
- **Capability reporting:** adapters expose `capabilities()` → `AdapterCapabilities`; use `require_capability(...)` for fail-fast.
- **Example adapters** (`BaselineTorchAdapter` / registry aliases) are **not** a senior pretrained integration.
- **Label-free boundary:** `run_label_free` strips `target_y_true` before `method.run` (interface-level).

---

## 5.4 Integration steps

1. **Add adapter** — Implement `code/tta/adapters/<name>.py` subclassing `ModelAdapter`; implement only capabilities the checkpoint provides; leave others raising `UnsupportedCapabilityError`; override `capabilities()`.
2. **Register** — `register_adapter("<name>", Factory)` in the adapters registry (or package init path used by `build_adapter`).
3. **Config checkpoint** — Point experiment YAML at `model_adapter: <name>` plus checkpoint / preprocess / shape fields (`session_tta` / configs).
4. **Preflight** — Run the checklist in §5.5. A dedicated `preflight_only` mode is **planned / to be wired via config** (today modes: `smoke` | `dry_run` | opt-in `full_a0_replay` via `phase3_tta_full_a0.yaml`).
5. **Live inference smoke** — Tiny frozen forward via `ModelInferenceSource` over 1–few trials → features + logits; confirm no NaN/Inf.
6. **Baseline / replay consistency** — Where comparable, `no_tta` acc must match Phase 2c `acc_target` per `cell_id` within project tolerance (A0 gate; WBCIC full A0 already complete on embedding replay).
7. **Minimal T3A** — `t3a_minimal` smoke only; report “pipeline runnable”, not “T3A effective”.
8. **Then Phase 3B Oracle** — Formal Oracle decision **after** pretrained path is credible; do not expand full T3A before the Oracle gate.

Do **not** rewrite TTA methods / evaluators to hard-code a senior architecture.

---

## 5.5 Preflight checklist

Use before any scientific claim on a new adapter.

- [ ] Checkpoint file readable at documented path
- [ ] `state_dict` loads into constructed architecture (key mismatches = fail)
- [ ] Input shape accepted (`validate_input_shape` or equivalent)
- [ ] `forward_features` → `[B, D]` (or `[D]` for single trial) with documented `D`
- [ ] `forward_logits` → `[B, C]` with `C == n_classes`
- [ ] No NaN / Inf in features or logits on a deterministic probe batch
- [ ] Label mapping matches dataset / Phase 2c convention
- [ ] Preprocessing + normalization match train-time recipe
- [ ] Deterministic batch: fixed seed → identical outputs (or documented nondeterminism)
- [ ] Capabilities clearly declared via `capabilities()` (unsupported raises `UnsupportedCapabilityError`)
- [ ] Metadata complete: channels/order, sfreq, n_times, unit, dataset/subject/session/seed mapping

**`preflight_only`:** treat as **planned / to be wired via config**. Checked-in modes today:
`smoke` (default), `dry_run` (no I/O), `full_a0_replay` (opt-in YAML only). Do not invent
runner flags in docs that are not in code.

---

## Registered adapters (as of Round-1 scaffold)

| Registry name | Role | Senior pretrained? |
|:---|:---|:---|
| `embedding_only` | Replay metadata / no live forward | No |
| `baseline_torch` | Example wrapper around project `build_model` | No (fixture / baseline) |
| `eegnet` / `deepconvnet` / `fbcnet` | Aliases of `BaselineTorchAdapter` | No (project baselines) |
| *(future)* `<senior_name>` | Real senior / foundation adapter | **Only when delivered + registered** |

Template (documentation only, not instantiable): `PretrainedModelAdapterTemplate` in
`code/tta/adapters/base.py`.

---

## Related pointers

- Design zone: `3_online_adaptation/PHASE3_TTA_DESIGN.md`, `PHASE3_ROUTE_PLAN.md` (repo root)
- Backend package: `code/tta/` (see `code/tta/README.md`)
- Experiment READMEs: `4_experiments/wbci_shu/tta/README.md`, `4_experiments/shu/tta/README.md`
- Short readiness pointer: `4_experiments/wbci_shu/tta/reports/PRETRAINED_READYNESS_NOTES.md`
