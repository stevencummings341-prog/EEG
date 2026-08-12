---
title: "ATCNet — official implementation, ported to PyTorch"
tags:
  - "#pipeline/5_dl_model"
  - "#modality/eeg"
created: "2026-08-09"
updated: "2026-08-09"
status: "active"
---

# ATCNet (Altaheri et al., IEEE TII 2023)

Why it is here: in the DSGNet paper (IEEE JBHI 2026,
[doi:10.1109/jbhi.2026.3689121](https://doi.org/10.1109/jbhi.2026.3689121)) ATCNet is the
**strongest baseline on SHUv5 three-class LOSO** (Acc 0.6834 vs DSGNet 0.6856). Running it
inside our own pipeline calibrates the pipeline: if our ATCNet lands near 0.68, the
protocol/recipe is comparable and the foundation-model numbers can be read as model
differences; if it lands far below, the gap is ours.

## 1. Provenance

Everything comes from the **authors' official repository**
[Altaheri/EEG-ATCNet](https://github.com/Altaheri/EEG-ATCNet) (Apache-2.0).

| Path | What |
|:---|:---|
| `_official_keras/models.py` | upstream verbatim (contains `ATCNet_`, `Conv_block_`, `TCN_block_`) |
| `_official_keras/attention_models.py` | upstream verbatim (`attention_block`, `mha_block`) |
| `_official_keras/UPSTREAM_README.md`, `_official_keras/LICENSE` | upstream verbatim |
| `atcnet_torch.py` | **our 1:1 PyTorch transcription** of `ATCNet_` (the upstream files are Keras and cannot run here) |
| `adapter.py` | project contract wrapper (`{logits, features, confidence}`) |

The upstream code is TensorFlow/Keras and this project is PyTorch, so the model had to be
transcribed rather than imported. `_official_keras/` is kept next to the port so any line
can be checked against its source.

## 2. Fidelity evidence

With the upstream BCI-IV-2a configuration (22 channels, 1125 samples, 4 classes) the port
has **113,732 parameters — exactly the number published in the official README's results
table**. Every layer, kernel size, pooling size, activation, dropout position, residual
path, Keras initializer (`glorot_uniform` for Conv2D/Dense, `he_uniform` for the TCN
Conv1D), `max_norm(0.6)` kernel constraint and `L2` penalty (conv 0.009, dense 0.5) is
transcribed. On our data (58 ch, 1000 samples, 3 classes): 114,719 params, `Tc=17`,
`Tw=13`, `F2=32`, 5 windows.

## 3. Deviations from upstream (exhaustive)

1. **Data layout** — Keras channels-last `(T, C, 1)` vs PyTorch `[B, 1, T, C]`. Re-indexing
   only.
2. **`same` padding split** — TF puts the extra pad of an even kernel at the end, torch's
   `padding="same"` puts it at the start, so the port pads explicitly with the TF split.
3. **Softmax vs logits** — upstream ends in `softmax` and trains with
   `CategoricalCrossentropy(from_logits=False)`; the port returns the pre-softmax scores
   and the trainer applies `cross_entropy`. Same objective.
4. **`attention` is restricted to `'mha'`/`None`** — `se`/`cbam`/`mhla` exist in
   `attention_models.py` but `ATCNet_` does not use them; passing them raises instead of
   silently doing something else.
5. **`L2` placement** — Keras folds `kernel_regularizer` into the loss automatically; here
   it is added by `ATCNetClassifier.training_step` via the trainer's `uses_custom_loss`
   hook (`code/training/e2e_trainer.py`). Turn off with `use_official_l2: false` (not
   upstream behaviour).
6. **Keras `MultiHeadAttention` is reimplemented** (`_KerasMHA`) because it decouples
   `key_dim` (8) from the embedding size (32); `torch.nn.MultiheadAttention` forces
   `head_dim = embed_dim / num_heads` and cannot express it.

Hyperparameters are the upstream defaults (`code/configs/models/atcnet.yaml`); nothing tuned.

## 4. Training recipe is ours, not the paper's

Official recipe (`main_TrainValTest.py`): Adam `lr=0.001`, batch 64, 500 epochs,
`ReduceLROnPlateau`. The DSGNet paper: Adam `lr=1e-4`, batch 128, 500 epochs. Our run uses
**this project's** cross-subject recipe (AdamW, cosine, batch 64, max 100 epochs,
patience 25, monitor macro-F1) so ATCNet and our five foundation models are compared under
identical conditions. The **split protocol** matches the paper exactly; the **optimizer
recipe** does not. State this in any report.
