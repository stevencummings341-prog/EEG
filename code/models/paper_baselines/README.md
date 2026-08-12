---
title: "Published baselines from the DSGNet paper (official code only)"
tags:
  - "#pipeline/5_dl_model"
  - "#modality/eeg"
created: "2026-08-10"
updated: "2026-08-10"
status: "active"
---

# Published baselines — Table II of the DSGNet paper

Purpose: reproduce the DSGNet paper's own baselines *inside our pipeline*, under the same
split protocol and the same training recipe, so the comparison against our models is
apples-to-apples and the paper's numbers become a sanity check on the pipeline.

Hard rule for this directory: **only the original authors' official code.** No
reimplementations, no model zoos, no ports by third parties. If a paper's official code is
not published or is incomplete, the model is left out and the reason recorded below.

## 1. Included

| Model | Paper ref | Official repo | Language | How we run it |
|:---|:---|:---|:---|:---|
| EEGNet | [18] Lawhern 2018 | [vlawhern/arl-eegmodels](https://github.com/vlawhern/arl-eegmodels) | Keras | 1:1 port → `eegnet_official_torch.py` |
| EEGNeX | [20] Chen 2024 | [chenxiachan/EEGNeX](https://github.com/chenxiachan/EEGNeX) | Keras | 1:1 port → `eegnex_official_torch.py` |
| EEG-Deformer | [23] Ding 2024 | [yi-ding-cs/EEG-Deformer](https://github.com/yi-ding-cs/EEG-Deformer) | **PyTorch** | imported and run **unmodified** |
| ATCNet | [24] Altaheri 2023 | [Altaheri/EEG-ATCNet](https://github.com/Altaheri/EEG-ATCNet) | Keras | 1:1 port → `code/models/atcnet/` |

Official sources are vendored verbatim under `_official/` so every ported line can be checked
against its origin. The Keras files there are **reference only** and are never imported
(TensorFlow is not installed in `mi_torch_cu118`).

## 2. Excluded, with reasons

| Model | Paper ref | Why not |
|:---|:---|:---|
| EEG-Inception | [27] Zhang 2021 | No official release. The only PyTorch version is braindecode's, which states in its own docstring: *"This implementation is not guaranteed to be correct, has not been checked by original authors, only reimplemented based on the paper."* |
| MDGEEG | [35] Song 2024 | Repository is an empty placeholder ("code will be published"). |
| EEG-DG | [38] Zhong 2024 | Released code is **incomplete and inconsistent**: the entry point `main_BCI_IV_2a_2source.py` does `from Shallow_Inception_Network_2source import DG_Network`, but that file is not in the repository (only `Modified_EEGNetwork.py` is). Its `DG_Network.forward(data1, data2)` is also hard-wired to exactly **two** source domains, while our protocol has 8 training subjects — adapting it would mean rewriting the model, not reproducing it. Additionally `Dist_Loss` detaches to NumPy, so it contributes no gradient. |
| DSGNet | — | Official repo ships only an architecture preview (*"full code will be released upon acceptance"*). |

## 3. Fidelity evidence

* **EEGNet**: parameter count derived layer by layer from the official Keras definitions —
  conv1 512 + bn1 16 + depthwise 352 + bn2 32 + separable 512 + bn3 32 + dense 2244 =
  **3,700** at 22 ch / 1125 samples / 4 classes, exactly what this port reports.
  ⚠️ Do not compare against the "2,548" in ATCNet's README results table: that is ATCNet's
  *own* EEGNet variant (different pooling), not `arl-eegmodels`.
* **EEG-Deformer**: no port, upstream code is executed as released, so there is nothing to
  verify beyond the wrapper.
* **ATCNet**: **113,732** params at 22 ch / 1125 samples / 4 classes — the exact number in
  the official README's results table. See `code/models/atcnet/README.md`.
* On our data (58 ch, 1000 samples @250 Hz, 3 classes): EEGNet 3,523 · EEGNeX 59,275 ·
  EEG-Deformer 1,612,307 · ATCNet 114,719.

## 4. Deviations from upstream (exhaustive)

Applies to the two Keras ports (`eegnet_official`, `eegnex`):

1. **Data layout** — Keras channels-last / channels-first vs PyTorch `[B, 1, C, T]`.
   Re-indexing only.
2. **`same` padding split** — TF puts the extra pad of an even or dilated kernel *after* the
   signal; torch's `padding="same"` puts it *before*. `keras_compat.pad_same_time` pads
   explicitly with the TF split so time alignment matches.
3. **`same` average pooling** — TF divides by the count of real (unpadded) elements;
   reproduced with `ceil_mode=True, count_include_pad=False`.
4. **Softmax vs logits** — upstream ends in `softmax` + categorical cross-entropy; the ports
   return pre-softmax scores and the trainer applies `cross_entropy`. Same objective.
5. **`max_norm` axis** — Keras `max_norm` defaults to `axis=0`, which for a Keras conv kernel
   `(kh, kw, in, out)` is a quirky per-row norm. We apply the standard per-output-filter
   renorm (`dim=0` on torch's `(out, in, kh, kw)`), the interpretation every PyTorch port of
   these models uses.
6. **Initializers** — explicitly set to Keras defaults (`glorot_uniform`, zero bias) instead
   of torch's Kaiming-uniform default.

For `eeg_deformer` the only deviation is the wrapper: a forward pre-hook on `mlp_head[0]`
captures its input so we can report `features`. The forward maths is untouched.

## 5. Training recipe is the experiment's, not each paper's

Structural hyperparameters are upstream defaults (`code/configs/models/*.yaml`), but the
optimizer recipe comes from the experiment config so that every model in a run is trained
identically. For the DSGNet comparison that recipe is the DSGNet paper's own
(Adam `lr=1e-4`, batch 128, 500 epochs) — see
`code/configs/experiments/paper_baseline_3c_821.yaml`.
