# Experiment Protocol

Derived from the senior's chat record (`docs/references/ChatGPT-EEG-MI-pretraining.md`,
sections 14-16). Dataset: 2C, 51 subjects (enumerate from disk), 3 sessions each,
200 trials/session. All experiments are Python/PyTorch; the model is **CAP-EEGNet**.

## Global rules

- Data entry = our Python-preprocessed `.npy` (the configured `processed_*` dir);
  NOT the paper `.mat` (that is a label cross-check only).
- **Split by subject, never by trial.** Target subjects must not appear in training.
- Persist every split to JSON in the configured splits dir (`splits/<run_id>_seed<k>.json`).
- Fixed seeds; repeated runs report **mean +/- std**.
- Metrics (all experiments): Accuracy, Balanced Accuracy, Macro-F1, AUC.
- Confidence metrics (when a confidence head exists): ECE, NLL, Brier score,
  confidence-accuracy curve, risk-coverage curve.
- Current stage builds these as configs + skeletons only; no GPU runs yet.

## Experiment 1 — CAP-EEGNet cross-subject zero-shot

- Randomly choose 41 source subjects; the other 10 are targets.
- Train on **all 3 sessions** of the 41 source subjects.
- Zero-shot test on **all 3 sessions** of the 10 target subjects.
- Repeat with several seeds (start with 5, scale to 10); report mean +/- std.
- Question answered: *can the pretrained model generalize to unseen subjects?*

## Experiment 2 — Target-subject fine-tuning (Stage 4)

- Start from the model pretrained on 41 source subjects.
- Per target subject: fine-tune on **Session 1**, test on **Session 2 + Session 3**.
- Compare adaptation strategies:
  - zero-shot (no fine-tuning)
  - classifier-only fine-tuning
  - adapter fine-tuning
  - prototype update
  - full-model fine-tuning
- Also study data budget: full Session 1 / first 20% / few-shot 5,10,20 per class /
  unsupervised (no labels).
- Expectation from the senior: adapter + prototype + confidence calibration should
  beat full-model fine-tuning for stability.

## Experiment 3 — Online adaptation (Stage 5)

Prequential ("test-then-train") evaluation. Per target subject:

```
Session 1: calibration / initial fine-tuning (offline)
Session 2: online test-then-update
Session 3: online test-then-update (continues from Session 2 state)
```

For each trial, strictly in this order:
1. Input the EEG trial.
2. Predict the MI class.
3. Output a confidence score.
4. Record prediction, confidence, correctness.
5. ONLY THEN update the model.

Allowed online-updated modules: prototype memory, adapter, confidence calibration
head, BN statistics (causal), classifier head (only if labels available). The full
backbone is frozen by default.

Two label regimes:
- **Supervised online**: true label known after each trial -> supervised update.
- **Unsupervised online**: use pseudo-labels, gated by confidence
  (`if confidence > threshold: update else: skip / only update norm stats`).

Report: pre-online vs post-online performance, performance-vs-trial curve, and
sensitivity to the confidence threshold.

Stability safeguards (from chat record sec. 11): confidence threshold, EMA teacher,
replay memory, prototype momentum, entropy regularization, feature distillation,
class-balanced memory, learning-rate decay.

## Experiment 4 — Ablations

At minimum:
- without confidence head
- without prototype
- without online update
- full-backbone online update (show it is worse / less stable)
- without domain alignment
- without dataset-aware router
- softmax-confidence only (vs multi-source confidence)
- no confidence threshold in online updates

## Reporting layout

```
outputs/<run_id>/
├── config.yaml         # resolved config
├── split.json          # subject split used
├── metrics.json        # summary metrics
├── per_trial.csv       # online: per-trial pred/conf/correct
├── curves/             # risk-coverage, perf-vs-trial, etc.
└── logs/
checkpoints/<run_id>/
```
