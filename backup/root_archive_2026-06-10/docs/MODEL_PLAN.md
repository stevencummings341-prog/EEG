# Model Plan

> See also `docs/ROADMAP.md` (the authoritative staged plan + minimal-vs-full table).

Final goal (NOT a plain EEGNet classifier): **Confidence-aware Online Adaptive
Multi-Subagent Pretraining Framework for Cross-subject MI EEG Decoding**. Main model =
**CAP-EEGNet** (Python/PyTorch), the concrete instance of the senior's vision
(`docs/references/ChatGPT-EEG-MI-pretraining.md`, sec.13–16): a multi-subagent,
confidence-aware, prototype-based, online-adaptive cross-subject MI decoder.

Baselines (EEGNet/DeepConvNet/FBCNet) are OPTIONAL and not the current priority.

## minimal vs full CAP-EEGNet (must distinguish)

- **Stage 0 minimal (implemented now)** = EEGNet encoder + linear classification head.
  `forward` returns `{logits, features, proto_dist=None, confidence=None}`. **Purpose:
  validate the data + training pipeline only. It is a baseline / lower bound, NOT the
  paper's final method.** Never report it as the final model.
- **full CAP-EEGNet (the paper method)** = everything below. Enabling any full component
  flag (`use_subagents/use_confidence/use_prototype/use_adapter/use_domain_align/
  use_online_update/use_dataset_router`) currently raises a clear NotImplementedError
  ("Reserved for full CAP-EEGNet … NOT implemented in the minimal sanity model").

## full CAP-EEGNet components

```
X [batch, 1, 58, 1000]
  -> Neural Subagents (deep neural experts: temporal-spectral, spatial-topology,
                       entropy-complexity, connectivity)  -> per-view latents + reliability
  -> (Dataset-aware Router, optional v2)  -> per-subagent weights
  -> Reliability-aware Fusion             -> fused representation
  -> Main Encoder (EEGNet-style)          -> features z
  -> Classification Head                  -> logits / p(y|x)
  -> Prototype Memory (global/subject/session) -> distances, margin
  -> Confidence Head (multi-source)       -> sample confidence c
  -> Domain Alignment Head                -> subject/session-invariant representation
  -> Online Update Module (test-then-update; lightweight, backbone frozen)
```

- **Neural Subagents**: deep, end-to-end **differentiable** experts — NOT handcrafted
  CSP/DE/transfer-entropy features concatenated together. Each view (time-frequency,
  spatial-topology, entropy/complexity, dynamic connectivity) is its own small network
  emitting a latent + a reliability score; fused by reliability-aware weighting.
- **Main Encoder**: temporal conv → depthwise spatial conv → separable conv
  (Lawhern et al. 2018), adapted to 58 ch / 250 Hz / 1000 samples. (= the Stage 0 encoder.)
- **Classification Head**: linear → 2 classes.
- **Prototype Memory**: maintain `p_c^global`, `p_{s,c}^subject`, `p_{s,k,c}^session`;
  output distance-to-own vs nearest-wrong and a margin. Online prototype is a convex
  combination, momentum-updated only on high-confidence trials.
- **Confidence Head**: fuse predictive entropy + prototype margin + augmentation
  consistency + OOD/calibration. **NOT just `max softmax`** (EEG models are confidently
  wrong); train with a calibration loss.
- **Adapter**: small bottleneck/LoRA module, the main thing adapted during target
  fine-tuning and online updates (keep the backbone frozen).
- **Domain Alignment Head**: subject/session-invariant representation
  (adversarial/CORAL/MMD/prototype-align): min H(Y|Z) and max H(S|Z).
- **Online Update Module**: test-then-update, confidence-gated, lightweight-only.
- **Dataset-aware Router (optional v2)**: predict per-subagent weights from dataset
  meta-features/statistics/probe scores.

## Losses

Pretraining: `L = L_cls + λ1·L_proto + λ2·L_conf (+ λ3·L_cons + λ4·L_domain)`.
Online:      `L_online = c_i·L_pseudo + λ1·L_proto + λ2·L_distill + λ3·L_calib`,
where `c_i` is the predicted confidence (gates + weights the update).

## Build order / implementation sequence (aligned to docs/ROADMAP.md stages)

1. ✅ (Stage 0) Path config + manifest + Python preprocessing → `[200,58,1000]` validated.
2. ✅ (Stage 0) `src/data/`: `SHUTrialDataset` + subject-wise `splits.py` (5 seeds) +
   smoke tests.
3. ✅ (Stage 0) **minimal** EEGNet encoder + classification head — sanity training passed
   (loss ↓, no shape errors). This is the baseline, NOT the final model.
4. 🚧 (Stage 1) cross-subject training loop + zero-shot eval over the 5 seeds (mean±std).
5. 🚧 (Stage 2) **Confidence head** (multi-source, not softmax-max) + **Prototype memory**
   (global/subject/session) + **Adapter** + **Domain alignment** → this is when "full
   CAP-EEGNet" actually exists. Implementation order within Stage 2:
   (a) prototype memory + margin, (b) multi-source confidence + calibration loss,
   (c) adapter (frozen backbone), (d) domain alignment.
6. 🚧 (Stage 3) target Session-1 fine-tuning (prefer adapter+prototype+calibration).
7. 🚧 (Stage 4) **Online update module**: test-then-update, confidence-gated, lightweight
   only (prototype/adapter/calibration/BN), backbone frozen; stability safeguards.
8. 🚧 (Stage 5) ablations + interpretability.
9. 🚧 (optional v2) Neural subagents expansion + dataset-aware router.

Note: items 4–9 are currently documented skeletons that raise NotImplementedError. Do not
silently ship the minimal model as the full method.

## `src/` module map

| Package | Responsibility |
| --- | --- |
| `src/data` | `manifest.py`, `splits.py` (subject-wise), `shu_dataset.py` |
| `src/preprocessing` | `neuracle_events.py` (TAL parser), `eog_ecg_clean.py` (formal ICA clean) + `pipeline.py` (save/.mat-check), `shu_preprocess.py` (legacy paper-style) |
| `src/models` | `cap_eegnet.py` (main), optional baselines (`eegnet.py`, ...) |
| `src/training` | training loops, losses, optimizers, seeding |
| `src/evaluation` | metrics (acc/bacc/f1/auc), calibration (ece/nll/brier), curves |
| `src/online` | test-then-update loop, lightweight online updates |
| `src/utils` | `paths.py` (config-driven), config/io/logging/seed |

## Confidence metrics

Accuracy, Balanced Accuracy, Macro-F1, AUC, plus ECE, NLL, Brier, and a
risk-coverage curve once the confidence head exists.
