# Model Plan

Main model of this project: **CAP-EEGNet — Confidence-aware Prototype EEGNet**
(Python/PyTorch). It is the concrete first instance of the senior's vision
(`docs/references/ChatGPT-EEG-MI-pretraining.md`): a confidence-aware, prototype-based,
online-adaptive cross-subject MI decoder.

Baselines (EEGNet/DeepConvNet/FBCNet) are OPTIONAL and not the current priority.

## CAP-EEGNet components

```
X [batch, 1, 58, 1000]
  -> EEGNet Encoder (backbone)            -> features z
  -> Adapter (lightweight; for fine-tune / online; backbone frozen by default)
  -> Classification Head                  -> logits / p(y|x)
  -> Prototype Head (global/subject/session prototypes) -> distances, margin
  -> Confidence Head (multi-source)       -> sample confidence c
```

- **EEGNet Encoder**: temporal conv → depthwise spatial conv → separable conv
  (Lawhern et al. 2018), adapted to 58 ch / 250 Hz / 1000 samples.
- **Adapter**: small bottleneck module, the main thing adapted during target
  fine-tuning and online updates (keep the backbone frozen).
- **Classification Head**: linear → 2 classes.
- **Prototype Head**: maintain `p_c^global`, `p_{s,c}^subject`, `p_{s,k,c}^session`;
  output distance-to-own vs nearest-wrong and a margin. Online prototype is a convex
  combination, momentum-updated only on high-confidence trials.
- **Confidence Head**: fuse predictive entropy + prototype margin + augmentation
  consistency + calibration. NOT just `max softmax` (EEG models are confidently wrong).

## Losses

Pretraining: `L = L_cls + λ1·L_proto + λ2·L_conf (+ λ3·L_cons + λ4·L_domain)`.
Online:      `L_online = c_i·L_pseudo + λ1·L_proto + λ2·L_distill + λ3·L_calib`,
where `c_i` is the predicted confidence (gates + weights the update).

## Build order (current stage: data/paths — keep model as skeleton)

1. (done) Path config + manifest + Python preprocessing → `[200,58,1000]` validated.
2. `src/data/`: `SHUTrialDataset` + subject-wise `splits.py` (read from processed dir).
3. EEGNet encoder + classification head (plain) — sanity vs paper-style data.
4. Add Prototype head + Confidence head → CAP-EEGNet.
5. Add Adapter for fine-tuning / online.
6. (optional later) extra experts / dataset-aware router (the chat record's v2).

## `src/` module map

| Package | Responsibility |
| --- | --- |
| `src/data` | `manifest.py`, `splits.py` (subject-wise), `shu_dataset.py` |
| `src/preprocessing` | `neuracle_events.py` (TAL parser), `shu_preprocess.py` (paper-style) |
| `src/models` | `cap_eegnet.py` (main), optional baselines (`eegnet.py`, ...) |
| `src/training` | training loops, losses, optimizers, seeding |
| `src/evaluation` | metrics (acc/bacc/f1/auc), calibration (ece/nll/brier), curves |
| `src/online` | test-then-update loop, lightweight online updates |
| `src/utils` | `paths.py` (config-driven), config/io/logging/seed |

## Confidence metrics

Accuracy, Balanced Accuracy, Macro-F1, AUC, plus ECE, NLL, Brier, and a
risk-coverage curve once the confidence head exists.
