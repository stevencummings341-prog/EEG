# Model Plan

Condensed from the senior's chat record (`docs/references/ChatGPT-EEG-MI-pretraining.md`).
The framework name: **Confidence-aware Online Adaptive Multi-Subagent Pretraining
Framework for Cross-subject MI EEG Decoding**.

The key reframing in the chat record: the "feature toolkits / subagents" are NOT
classic hand-crafted feature extractors — they are **differentiable neural
sub-modules (neural experts)**, fused by reliability/confidence, with explicit
sample-confidence prediction and online adaptation built in from the start.

## Target architecture (long-term)

```
EEG Trial [channels, time]
  ↓
Neural Subagent Toolkit (differentiable experts)
  ├── Temporal-Spectral Agent   (learnable filterbank + temporal CNN / SSM)
  ├── Spatial-Topology Agent     (GNN / spatial attention / channel transformer)
  ├── Entropy-Complexity Agent   (latent complexity / masked modeling)
  ├── Connectivity Agent         (dynamic adjacency / graph attention)
  ├── Prototype Agent            (global / subject / session prototypes)
  ├── Confidence Agent           (uncertainty + calibration heads)
  ├── Domain Agent               (subject/session adversarial / invariance)
  └── Online Adaptation Agent    (adapter / LoRA / BN / prototype memory)
  ↓
Dataset-aware Router  (weights each subagent from dataset-level meta-features)
  ↓
Reliability-aware Fusion  (F = Σ_k w_k · f_k, with per-trial reliability)
  ↓
Main Encoder
  ↓
MI Classification + Confidence Prediction + Online Update
```

## Build order (do not build it all at once)

### v0 — Baselines (Stage 2)
- EEGNet, DeepConvNet, FBCNet. Plain classifiers, no confidence/prototype yet.
- Goal: validate preprocessing and get baseline numbers.
- Reference (read-only, old API): `/share/workspace2/.../code/Deep_learning/`.

### v1 — Minimal confidence+prototype model (Stage 3-4)
Per the chat record's "minimal version", first 5 agents are enough to form a clear
method contribution:
- EEGNet-style **encoder backbone**.
- **Classification head**.
- **Prototype memory/head** (start with global class prototypes; add subject/session).
- **Confidence head** (multi-source, see below).
- **Adapter** for fine-tuning + online adaptation.

### v2 — Add experts & routing (later)
- Riemannian / CSP-FBCSP, Connectivity / Transfer-Entropy, Domain alignment,
  Dataset-aware neural router.

## Confidence prediction (important)

Do NOT use `max softmax` as the confidence. EEG models are often confidently wrong.
Fuse multiple sources:
- softmax confidence + predictive entropy
- prototype margin (distance to own vs nearest wrong prototype)
- augmentation consistency
- session stability / OOD distance
- calibration confidence

`c_i = f_θ(z_i, H(Y|z_i), margin_i, stability_i)`.

Train with classification + confidence-calibration + prototype-margin + (optional)
consistency losses. Evaluate confidence with ECE, NLL, Brier, risk-coverage.

## Losses

Pretraining:
```
L = L_cls + λ1·L_mask + λ2·L_proto + λ3·L_conf + λ4·L_cons + λ5·L_domain
```

Online:
```
L_online = c_i·L_pseudo + λ1·L_proto + λ2·L_distill + λ3·L_calib
```
where `c_i` is the predicted sample confidence (gates and weights the update).

## Prototypes

Maintain `p_c^global`, `p_{s,c}^subject`, `p_{s,k,c}^session`. Online prototype is a
convex combination, updated only on high-confidence trials with momentum:
`p_c ← m·p_c + (1-m)·z_i`.

## Online stability safeguards

confidence threshold · EMA teacher · replay memory · prototype momentum ·
entropy regularization · feature distillation · class-balanced memory · LR decay.

## `src/` module map (skeletons exist; fill in over time)

| Package | Responsibility |
| --- | --- |
| `src/data` | datasets, dataloaders, subject-wise splits |
| `src/preprocessing` | raw BDF -> processed tensors (both variants) |
| `src/models` | encoders (eegnet, deepconvnet, fbcnet), heads, prototypes, adapters |
| `src/training` | training loops, losses, optimizers, seeding |
| `src/evaluation` | metrics (acc/bacc/f1/auc), calibration (ece/nll/brier), curves |
| `src/online` | test-then-update loop, prototype/adapter/calibration updates |
| `src/utils` | paths, config loading, logging, io, reproducibility |
