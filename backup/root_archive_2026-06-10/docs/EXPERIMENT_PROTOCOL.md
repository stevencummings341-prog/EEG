# Experiment Protocol

Dataset: 2C, 51 subjects (enumerate from disk), 3 sessions each, 200 trials/session.
All experiments are Python/PyTorch. Data entry = our `eog_ecg_clean` `.npz`,
**`status=ok` only** (148 ok / 5 failed).

## CURRENT MAINLINE (2026-06-07) — three baseline architectures

The plan runs **the three baselines (EEGNet + DeepConvNet + FBCNet)** on the 148
`status=ok` sessions:

1. **Within-session 10-fold CV** (Stratified K-fold per ok session) — no-drift upper bound.
   Start by checking **EEGNet** against the **WBCIC-SHU paper EEGNet baseline**
   (S1 81.77% / S2 86.63% / S3 88.90%; mean ≈ 85.3% within-session 10-fold CV).
2. **Summarize** all three models.
3. **Cross-session** (train ses-i → test ses-j, directed pairs, same subject) — drift.
4. Outputs: `results_within_session.csv`, `results_cross_session.csv`,
   `summary_by_model_protocol.csv`, figures, and `SESSION_MODEL_COMPARE_REPORT.md`
   (`scripts/summarize_session_results.py`). Metrics: Acc/BalAcc/MacroF1/AUC + NLL/Brier/ECE,
   cross-session drop, relative drop. Seed = 0.

Plus the standalone **session-drift diagnostics** (`docs/SESSION_DRIFT_ANALYSIS.md`,
direction A) which is already complete (144 pairs / 50 subjects).

> See `docs/BASELINE_PROTOCOL.md` for the exact, leakage-free within/cross definitions and
> run commands.

### Optional / FUTURE work (intentionally NOT run now)

Kept in code/docs but explicitly out of the current mainline:

- **CAP-EEGNet (v1/v2)** and its components: agent/toolkit/neural-subagents, prototype
  memory, multi-source confidence. The trainer/protocol is model-agnostic, so CAP-EEGNet
  still works when re-enabled; `configs/session_model_compare.yaml` currently has
  `models: ["eegnet","deepconvnet","fbcnet"]` (add `cap_eegnet` to include it).
- **Other protocols/experiments**: LOSO, 41/10 cross-subject pretraining + zero-shot,
  target Session-1 fine-tuning, online test-then-update, ablations. The 41/10 splits exist
  (`splits/cap_eegnet_4110_seed*.json`) and the four experiments below are retained as
  future work.

---

## Global rules

- Data entry = our Python-preprocessed `.npz`, one per session under the configured
  `eog_ecg_clean_root`, **only `status==ok` sessions**
  (`SHUTrialDataset.from_manifest(processed_manifest.csv, statuses=('ok',))`);
  the paper `.mat` is a label cross-check only and is **NEVER** a training/eval entry.
- **Split by subject, never by trial/session.** Target subjects must not appear in training.
- **Repeated subject-wise split**: 5 seeds **2026–2030** (scale to 10 later).
  **Each split trains ONE independent model**; report **mean ± std** across seeds.
  Splits persisted to `splits/<run_id>_seed<k>.json` (`run_id=cap_eegnet_4110`).
- Target subjects require all 3 sessions `ok` (needed for Exp 2/3). Source may contain
  subjects with failed sessions but training uses their `ok` sessions only; failed
  sessions are in `excluded_sessions`.
- Metrics (all experiments): Accuracy, Balanced Accuracy, Macro-F1, AUC.
- Confidence metrics (when the confidence head exists): ECE, NLL, Brier score,
  confidence-accuracy curve, risk-coverage curve. **Confidence is multi-source, never just
  max-softmax.**
- Splits are DONE; training/eval code is still skeletons. No GPU runs until the CPU-only
  torch is fixed AND the user confirms.

> ⏸ **The four experiments below are FUTURE WORK** (not part of the current
> cross-session DG mainline). Retained for when the project returns to the full
> CAP-EEGNet pretraining / fine-tuning / online roadmap.

## Experiment 1 (FUTURE) — CAP-EEGNet cross-subject zero-shot (Stage 1)

- **Input**: 41 source subjects' all `ok` sessions (train); 10 target subjects' all
  sessions (zero-shot test). One model per seed (2026–2030).
- **Process**: train on source; evaluate target with NO target data in training.
- **Output**: Accuracy / Balanced Accuracy / Macro-F1 / AUC, reported as **mean ± std**
  over the 5 seeds; per-seed metrics + the split used saved under `outputs/<run_id>/`.
- **Forbidden**: any target trial in source training; trial/session-wise splitting;
  using derivatives `.mat`; reporting a single seed as the result.
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
  beat full-model fine-tuning for stability. **Default to adapter+prototype+calibration,
  NOT full-model fine-tune.**
- **Forbidden**: fine-tuning or testing on the wrong sessions (S1=tune, S2+S3=test only);
  leaking S2/S3 into the fine-tune set; defaulting to full-backbone fine-tuning.

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
sensitivity to the confidence threshold. Output `outputs/<run_id>/per_trial.csv`
(trial, pred, confidence, correct, updated?).

**Forbidden**: updating on a whole session then testing the same session; updating BEFORE
predicting/recording a trial; updating the full backbone by default; unguarded
pseudo-label updates in the unsupervised regime.

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
