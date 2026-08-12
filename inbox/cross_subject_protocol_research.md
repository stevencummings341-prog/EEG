---
title: "Cross-subject protocol research: SHU 2022 + WBCIC-SHU 2025"
tags:
  - "#modality/eeg"
created: "2026-08-04"
updated: "2026-08-04"
status: "active"
---

# Cross-subject protocol research: SHU 2022 + WBCIC-SHU 2025

> **Scope and epistemic rules for this document.** Every number below is traced to a named
> source with a URL. Where a source could not be reached (paywall, early access) or where a
> protocol detail is simply not stated in the paper, the text says **"not found / unverified"**
> explicitly rather than guessing. Each claim carries a confidence tag:
> **[high]** = read directly in the primary source; **[medium]** = read in a secondary or
> preprint source, or inferred from an explicit statement; **[low]** = single weak source, or
> my own inference. Statements that are my own judgement rather than a paper's claim are
> marked **(assessment)**.
>
> Literature search performed 2026-08-04. Citation counts from the OpenAlex API on that date.

---

## 1. Core findings (bullet summary)

- **Neither dataset has an official cross-subject split, and neither dataset paper runs a
  zero-shot cross-subject benchmark.** Both authors' own benchmarks are subject-specific
  (within-session and cross-session). WBCIC-SHU adds nothing cross-subject at all; SHU 2022's
  "cross-session adaptation" (CSA) condition is the closest thing, and it is a
  leave-one-subject-out *source* pool followed by supervised fine-tuning on target-subject
  labels — not calibration-free. **[high]**
- **Correction to the task brief:** the SHU 2022 paper's title is *"A large EEG dataset for
  studying **cross-session variability** in motor imagery brain-computer interface"*
  (Sci Data 9, 531), not "A large-scale EEG dataset for motor imagery BCI". The framing
  matters: the authors designed and marketed the dataset for **cross-session**, and the
  literature has followed them. **[high]**
- **WBCIC-SHU 2025 is real, well documented, and lightly used.** It has 18 citations
  (OpenAlex, 2026-08-04). I read the titles/abstracts of all 18. Most are reviews, or papers
  that cite it as "a large dataset exists" while actually experimenting on BCI IV-2a. **Only
  two** were verified to run experiments on it, and **only one** of those does cross-subject.
  So the answer to "is WBCIC-SHU literature scarce?" is: **yes, and cross-subject literature
  on it is essentially a single paper.** **[high]**
- **That one paper is the single most useful reference for this task: EDAPT**
  (J Neural Eng, 2026). It uses WBCIC-SHU (as MOABB's `Yang2025`) under a genuine
  subject-grouped split — **2-fold CV over subjects, 50% train subjects / 50% test subjects** —
  and reports zero-shot cross-subject accuracy of **0.81 (EEGNet), 0.85 (DeepConvNet),
  0.82 (ShallowConvNet), 0.71 (ATCNet)**. **[high]**
- **The surprising implication of those numbers:** on WBCIC-SHU, zero-shot *cross-subject*
  accuracy (0.81–0.85) sits roughly level with the dataset authors' *within-session*
  accuracy (EEGNet 85.32%). On this dataset, crossing subjects appears to cost much less than
  the field's usual intuition suggests. **(assessment, high confidence in the two numbers,
  medium confidence that they are directly comparable given different preprocessing.)**
- **SHU 2022 will behave very differently, and there is a floor effect.** The authors'
  own cross-session accuracy is **53.7%** against a stated chance level of 51.4–53.7%, i.e.
  not significantly above chance (they report p > 0.05). A cross-subject number on SHU 2022
  should be expected near chance unless adaptation is used. **[high]**
- **I could not verify a single published zero-shot LOSO cross-subject accuracy for SHU 2022
  from a primary source.** The best candidate is DSGNet (IEEE JBHI 2026, early access), which
  explicitly states LOSO on "SHU Version 5", but the per-dataset numbers are behind a paywall.
  This is a genuine gap, and it is worth knowing before promising a comparison to literature.
  **[high that the gap exists; the DSGNet numbers are not found / unverified]**
- **Two widely-cited SHU 2022 numbers are not what they appear to be**, and both would be
  wrong to use as a cross-subject sanity band. See §2.4 — one is a randomly-shuffled
  within-subject split published under the title "cross session", the other is a
  single-source-subject transfer that trains on target-user data.
- **MOABB now ships both datasets and a formal cross-subject protocol enum.** `Yang2025` is a
  registered MOABB dataset, and `CrossSubjectEvaluation` + `CrossSubjectMode` encode exactly
  the design decisions this report is about (0% / 20% / 50% / 100% target data, labeled vs
  unlabeled, transductive vs inductive). Adopting those names would make the student's
  protocol legible to reviewers at zero cost. **[high]**

---

## 2. Dataset A: SHU 2022

### 2.1 Primary source

| Field | Value |
|:---|:---|
| Title | A large EEG dataset for studying cross-session variability in motor imagery brain-computer interface |
| Authors | Jun Ma, Banghua Yang (corresponding), Wenzheng Qiu, Yunzhe Li, Shouwei Gao, Xinxing Xia |
| Year / venue | 2022, *Scientific Data* 9:531 (published 2022-09-01) |
| DOI | <https://doi.org/10.1038/s41597-022-01647-1> |
| Full text | <https://pmc.ncbi.nlm.nih.gov/articles/PMC9436944/> (PMCID PMC9436944, PMID 36050394) |
| Data | figshare <https://doi.org/10.6084/m9.figshare.19228725> |
| Citations | 84 (OpenAlex, 2026-08-04) |
| Ethics | Shanghai Second Rehabilitation Hospital Ethics Committee, ECSHSRH 2018-0101 |

Confidence for this whole table: **[high]**.

### 2.2 Dataset facts (all [high], from the PMC full text)

- 25 healthy subjects, age 20–24, 12 female, all naive to MI-BCI. IDs `sub-001`…`sub-025`.
- 5 sessions per subject, recorded 2–3 days apart.
- 32 Ag/AgCl electrodes (10–10 system), 250 Hz, 24-bit, impedance kept below 20 kΩ.
- **90 to 100 trials per session** — nominally 100, but bad trials were removed, so sessions
  have variable trial counts. The paper's Table 2 says "90 to 100" while the abstract says
  "100 trials"; treat 100 as nominal only.
- Trial: 4 s of MI, 1000 samples. Labels `1` = left hand, `2` = right hand (grasping).
- Nominal total 12,500 trials (5 × 25 × 100); actual total is lower.
- Author preprocessing already applied to the released data: bad segments removed (EEGLAB
  auto-flag at >100 µV, then confirmed by two experienced reviewers), baseline removed,
  **0.5–40 Hz FIR band-pass**.
- Released as `.edf` (continuous, preprocessed) and `.mat` (epoched trial data, shape
  `(100, 32, 1000)` = trials × channels × samples), EEG-BIDS organised.
- **Access gotcha:** the figshare `.zip` files are password-protected; the password must be
  requested from Prof. Yang B.H. (`yangbanghua@shu.edu.cn`). **[high]**
- **Timing inconsistency to be aware of:** the paper says each trial lasted 7.5 s, while the
  figshare description says 0–2 s rest / 2–4 s prompt / 4–8 s MI (i.e. 8 s). The released 4 s
  window is the MI period in both accounts, so this does not affect the epochs you get, but
  do not cite a trial duration without checking which source you mean. **[high]**

### 2.3 How the dataset authors set up their own experiments

This is the authoritative protocol reference for this dataset. All **[high]**.

**Shared deep-learning recipe** (EEGNet, deepConvNet, FBCNet): batch size 16, learning rate
0.001, max 1500 epochs, `NLLLoss`, Adam, early stopping when validation accuracy stops
improving. Preprocessed data fed directly, no extra normalization described.

**(a) Within-session (WS).** Within each session independently, all trials randomly split
**8:1:1** into train/validation/test; reported result is the mean of **10-fold
cross-validation**; 125 sessions evaluated. Classical baselines: CSP+SVM (3–35 Hz FIR,
0–4 s window) and FBCSP+SVM (0.5–4 s window). Best: **FBCNet 68.8% ± 0.146**. Note the
split is random over trials within a session, so WS is the optimistic ceiling.

**(b) Cross-session (CS), subject-specific.** Session 1 = training set; sessions 2–5 of the
**same subject** = test set. CSP/FBCSP were dropped as too weak. Average **53.7%**. EEGNet
was the best of the three DL models. 20 sessions exceeded 60% with EEGNet; the single best
result was sub-13 session 3 at 90%. **ANOVA found no significant difference between CS and
chance level (p > 0.05).**

**(c) Cross-session adaptation (CSA) — the only cross-subject-flavoured protocol the authors
run.** Mechanism: adaptive transfer learning on a deepConvNet backbone.
- One subject is the **target domain**; **all remaining subjects are the source domain**
  (i.e. a leave-one-subject-out source pool).
- The source domain is split into train/validation by holding out **3 randomly selected
  subjects as the validation set**, the rest train. *This is a held-out-subjects validation
  scheme, which is the leakage-safe choice.*
- Source base model learning rate 0.01.
- The **target** subject's data is split 90% train / 10% test via 10-fold CV; the base model
  is re-trained on the target training data at learning rate 0.0005.
- The fraction of the target training set actually used for adaptation is swept over
  **0%, 30%, 50%, 70%, 90%, 100%**.
- Results: at 50% → **70.52%**; at 100% → **78.86%** (the headline 78.9%). Significance
  versus WS-CSP: p < 0.05 above 50% ratio, p < 0.001 above 90%.

> **The important reading of CSA for this project.** The 0% point of that sweep *is* a
> zero-shot cross-subject (calibration-free) condition — LOSO source pool, no target data.
> The paper plots it in Fig. 8 but **the numeric value at 0% is not stated in the article
> text**; it would have to be read off the figure or from the Supplementary Tables /
> `results.csv` shipped with the dataset. **Not found / unverified as a citable number.**
> **[high that the value is absent from the text]**

**Stated chance levels** (used for the significance dashed lines): 51.4% at p = 0.001 for
N = 12500; 51.6% at p = 0.05 for N = 2500; 53.7% at p = 0.001 for N = 2500. Method follows
Combrisson & Jerbi. Useful: it tells you the dataset's own authors consider ~51–54% the
"nothing is happening" band. **[high]**

**Author-stated intended uses** (Usage Notes): within-session, cross-session, and
**cross-subject** ("data of multiple subjects were transferred to train a better model").
So cross-subject is explicitly sanctioned by the authors — they just never benchmarked it
without target data. **[high]**

### 2.4 Papers that use SHU 2022 — and two numbers you should not trust

| Paper | Venue / URL | Evaluation type on SHU | Number |
|:---|:---|:---|:---|
| Ma et al. 2022 (dataset paper) | Sci Data 9:531 · <https://doi.org/10.1038/s41597-022-01647-1> | within-session; cross-session; cross-session adaptation (LOSO source + target fine-tune) | WS 68.8% (FBCNet); CS 53.7%; CSA 78.9% |
| **ResGCN + H-RFE** (Chen et al.) | Sci Rep 14 (2024) · <https://doi.org/10.1038/s41598-024-73536-z> | **titled "cross session" but is within-subject, sessions pooled, trials randomly shuffled** — see below | 90.03% (10 subjects) — **do not use** |
| **ST-GENN** | J Neurosci Methods (2025) · <https://pubmed.ncbi.nlm.nih.gov/40350042/> · <https://www.sciencedirect.com/science/article/abs/pii/S0165027025001244> | cross-subject, but single "golden subject" source **and target-user data is used to train the Generator** | 67.2% — **not zero-shot** |
| **BARN-DA** | Brain Sci 16(4):363 (2026) · <https://doi.org/10.3390/brainsci16040363> | **cross-session only on SHU** (their Table 2 literally labels SHU "Cross Session") | 61.76% cross-session |
| **DSGNet** (Lou, X. et al.) | IEEE JBHI 2026 early access · <https://doi.org/10.1109/jbhi.2026.3689121> · <https://bura.brunel.ac.uk/handle/2438/33344> | **LOSO cross-subject** on OpenBMI + BCI IV-2a + **SHU Version 5** + BCI IV-2b | **not found / unverified** (paywalled) |
| EEG-MFTNet | arXiv:2604.05843 · <https://arxiv.org/html/2604.05843v1> | subject-dependent cross-session (ses-1 train, 20% held out for val, ses 2–5 test, per-subject model) | 58.9% avg (per session 58.4 / 57.2 / 61.0 / 58.8) |
| Spatio-temporal DL evaluation | ICEECIT 2024 · <https://doi.org/10.1109/iceecit63698.2024.10859806> | **protocol not stated in abstract** | EEGNex 75.67%, ATCNet 76.05% — **protocol unverified** |
| BTTA-DG | ICLR 2026 submission, seen only via a paper-note site · <https://en.papernotes.org/ICLR2026/self_supervised/bayesian_test-time_adaptation_via_dirichlet_feature_projection_and_gmm-driven_in/> | claims cross-subject LOSO on SHU MI + 3 MOABB datasets, 10 runs per method | SHU number **not shown**; **[low]** confidence, secondary source |

**Why the 90.03% ResGCN number is not a cross-session or cross-subject result.** Direct quotes
from the paper: *"We utilize the data from each sampling time point as input samples, **shuffling
the entire EEG data collected during the motor imagery period of each subject** to create
training data"* and *"Each experiment employed 10-fold cross-validation, with 90% of the dataset
used for training and the remaining 10% for testing."* It also uses **only the first 10
subjects**. So: single-subject models, all five sessions pooled, trials randomly shuffled before
splitting. Nothing is held out by session or by subject. **[high]** on the quotes;
**(assessment, high)** that the resulting 90.03% cannot be compared to any cross-session or
cross-subject number. It is nonetheless cited as a cross-session SOTA, including by the search
engines I used, so it is worth being explicit about.

**Why ST-GENN's 67.2% is not zero-shot.** The paper states: *"A golden subject is selected as
the source domain, while **concurrently utilizing data from target users to train the
Generator**"* and *"Unlike traditional transfer learning methods that rely on large datasets
containing data from multiple subjects, the proposed method only requires training with data
from one subject."* So it is (i) single-source-subject, not a 24-subject pool, and (ii)
target-data-using. A Chinese-language news summary of this paper claims it used LOSO; the
primary text does not support that, and I trust the primary. **[high]** on the quotes,
**[medium]** on my reading that this makes the number non-comparable to LOSO.

**Why BARN-DA matters even though it is not cross-subject on SHU.** BARN-DA is a 2026
domain-adaptation paper that *does* run LOSO cross-subject — on BCI IV-2a, 2b and III 4a — and
*chose not to* on SHU, using SHU only for cross-session. **(assessment)** That is a meaningful
signal about the community's read of SHU's difficulty. Its cross-session SHU table also gives a
clean per-subject baseline set worth reusing: ConvNet, EEGNet, ATCNet, CTNet, SST-DPN, BARN-DA
61.76% average. Hyperparameters: Adam, batch 64, lr 0.002, **300 epochs for cross-session and
50 epochs for cross-subject**. **[high]**

### 2.5 SHU 2022 gotchas

- **Variable trial counts per session (90–100) mean class balance is not guaranteed.** Bad-trial
  removal was applied per trial, not per class, so a session can be imbalanced. Count labels
  per session rather than assuming 50/50. **[high]** on the mechanism; the actual per-session
  imbalance is **not found / unverified** (needs a pass over the `.mat` files).
- **Labels are `{1,2}`, not `{0,1}`.** Must be remapped. **[high]**
- **The released 4 s epoch is already the MI window.** You cannot choose a "0.5–3.5 s after cue"
  crop relative to the cue from the `.mat` trial data, because the cue period is not in the
  epoch. If you want a different window relative to cue onset you must go back to the `.edf`
  continuous data + `.tsv` events. **[high]**
- **Band-pass is already 0.5–40 Hz with baseline removed.** Any further filtering is on top of
  that; you cannot recover 0–0.5 Hz. **[high]**
- **Floor effect.** Cross-session is at chance (53.7%, p > 0.05). Expect cross-subject to be at
  or below that without adaptation, and design your statistics accordingly — with a near-chance
  floor, a 1–2pp "improvement" is noise. **(assessment, high)** This matches this project's own
  Phase 1/2b SHU findings (cross-session 0.538 EEGNet; best alignment gain +1.42pp).
- **No commonly excluded "bad subjects" convention exists** for SHU. **[high]** — I found no
  paper excluding specific subjects. Individual differences are large though: EEG-MFTNet reports
  subjects 6 and 20 consistently >90% while 1, 18, 19, 25 stay below 55%; the dataset paper
  highlights subjects 6, 13, 20, 21 as separable. **[high]**
- **No recommended channel subset** is published for SHU. **[high]** — the ResGCN paper's channel
  selection is per-subject and adaptive, not a fixed recommendation.
- **No official train/test split, no competition rules.** **[high]**

---

## 3. Dataset B: WBCIC-SHU 2025

### 3.1 Primary source — the dataset does exist and is properly published

| Field | Value |
|:---|:---|
| Title | A multi-day and high-quality EEG dataset for motor imagery brain-computer interface |
| Authors | Banghua Yang*, Fenqi Rong* (equal first), Yunlong Xie, Du Li, Jiayang Zhang, Fu Li, Guangming Shi, Xiaorong Gao |
| Year / venue | 2025, *Scientific Data* 12:488 (received 2023-05-22, accepted 2025-03-11, published 2025-03-23) |
| DOI | <https://doi.org/10.1038/s41597-025-04826-y> |
| Data | Figshare+ <https://doi.org/10.25452/figshare.plus.22671172> (65.6 GB single ZIP, CC-BY 4.0) |
| Mirror | NEMAR / OpenNeuro <https://doi.org/10.82901/nemar.nm000246> (`nm000246`) |
| Loader | MOABB `Yang2025` · <https://moabb.neurotechx.com/docs/generated/moabb.datasets.Yang2025.html> |
| Citations | 18 (OpenAlex, 2026-08-04) |
| Provenance | 2019 World Robot Conference Contest — BCI Robot Contest MI (WBCIC-MI) |
| Ethics | Tsinghua University Medical Ethics Committee, 20190002 |

Confidence: **[high]**. Note "WBCIC" expands to **World Robot Conference Contest – BCI
Robot Contest**, not "Wearable BCI Challenge" as the task brief guessed. **[high]**

### 3.2 Dataset facts (all [high])

- 62 healthy right-handed participants, ages 17–30, 18 female, all BCI-naive.
  **2C paradigm = 51 subjects** (left vs right hand grasping); **3C = 11 subjects**
  (+ foot-hooking). MOABB confirms 2C = subjects 1–51, 3C = subjects 52–62.
- 3 sessions per subject, on different days. Each session ≈ 35–48 min: eyes-open 60 s,
  eyes-closed 60 s, then **5 MI blocks**, with a flexible ≥60 s break between blocks.
- 2C: **40 trials per block × 5 blocks = 200 trials per session, balanced 100 / 100 per class.**
  3C: 60 per block = 300 per session, 100 per class. **So 2C is class-balanced by design.**
- Trial structure: **1.5 s visual+auditory cue → 4.0 s MI → 2.0 s rest = 7.5 s.** Participants
  were told to mentally repeat the imagined action 2–4 times during the MI period.
- Raw: 64 channels — **1–59 EEG, 60 ECG, 61–64 EOG** — Neuracle NeuSen W wireless amplifier,
  **1000 Hz**, impedance target <5 kΩ, 10–20 / 10-05 montage. ECG and EOG electrodes were
  *not used* in these experiments.
- Author preprocessing chain, in order: drop ECG+EOG (→ 59 ch) → **re-reference to Pz**
  (→ **58 ch**, Pz consumed as the reference; the paper states this was the best of the
  methods they tested) → FIR band-pass **0.5–40 Hz** plus **50 Hz** line filtering (with
  cutoffs below 2 Hz doubled) → extract the **4 s epoch after the cue ends** → remove epoch
  baseline → **downsample to 250 Hz**.
- Processed `.mat` per session: `data` of shape `[58 × 1000 × 200]`
  (channels × samples × trials), `labels` in `{1,2}` (2C) or `{1,2,3}` (3C).
- **153 independent 2C sessions** and 33 independent 3C sessions were benchmarked
  (51 × 3 = 153). *This is consistent with this project's own QC result of 148 usable /
  5 failed WBCIC sessions.*

### 3.3 The authors' own benchmark is within-session only — and has a leakage flaw

**Protocol** (all **[high]**): 10-fold cross-validation, per session. Deep models share
batch size 16, lr 0.001, `NLLLoss`, Adam. A **two-stage training strategy** (borrowed from
FBCNet) with an **80 / 10 / 10 train / validation / test** split:
- *Stage 1:* train on the 80%, monitor validation accuracy, stop if it does not improve for
  **200 consecutive epochs**, restore the parameters at peak validation accuracy.
  Max 1500 epochs.
- *Stage 2:* starting from the restored model, train on train+validation combined
  (the 90%), and **"terminated when the test set loss fell below the loss achieved during
  the first stage."** Max 600 epochs.

> ⚠ **Gotcha (assessment, high confidence).** Stage 2's stopping criterion is a function of the
> **test set** loss. That is test-set leakage into model selection, so the headline 85.32%
> should be read as an optimistic within-session ceiling, not a clean generalization estimate.
> If the student reproduces the official benchmark as a reference point, this must be stated.
> The quoted sentence is verbatim from the paper. **[high]** on the quote.

**2C results across 153 sessions** (input = the 58-channel preprocessed data): **[high]**

| Algorithm | 2C accuracy | 3C accuracy (33 sessions) |
|:---|---:|---:|
| CSP + SVM | 61.12% | not reported |
| FBCSP + SVM | 67.46% | 58.40% |
| **EEGNet** | **85.32%** | 75.34% |
| deepConvNet | 84.47% | **76.90%** |
| FBCNet | 78.40% | 74.77% |

**Session effect — this is the single most important design consideration for this dataset.**
With EEGNet within-session, accuracy **rises** monotonically on average across recording days:
2C session 1 **81.77%** → session 3 **88.90%**; 3C 71.91% → 83.27%. The authors attribute this
to subjects *learning* the MI task. Most of the 51 subjects improve from session 1 to session 3;
S3, S6, S20, S24, S26 are consistently high; **S9 and S16 get worse in session 3**. **[high]**

> **(assessment, high)** This means "session" on WBCIC-SHU is not a pure nuisance/drift variable
> the way it is on SHU 2022 — it is confounded with skill acquisition. Any cross-subject result
> will shift depending on which sessions you pool, and pooling all three sessions of every
> subject mixes three different skill levels into one training distribution.

**Authors' own cross-dataset comparison** (same preprocessing and algorithms applied to all):
their 2C 85.31% vs **BCI IV-2a 79.03%** (4-class) vs **OpenBMI 74.70%** (2-class). **[high]**

### 3.4 Who actually uses WBCIC-SHU — the honest answer is "almost nobody, yet"

18 citations total. I checked all 18 titles/abstracts. Verified users:

**(a) EDAPT — the only verified cross-subject work on this dataset. [high]**

> Sabino, A. et al. (mackelab). *EDAPT: towards calibration-free BCIs with continual online
> adaptation.* J Neural Eng (2026). <https://doi.org/10.1088/1741-2552/ae5689> ·
> preprint <https://arxiv.org/html/2508.10474v1> · code <https://github.com/mackelab/EDAPT>
>
> *(Author list beyond the mackelab group attribution: **not found / unverified** — the IOP and
> arXiv landing pages I retrieved did not expose the full author list in a form I could read
> reliably. Cite by title until confirmed.)*

Its protocol on `Yang2025` is directly reusable, and every element below is **[high]**:

- **Subject split: 2-fold cross-validation over subjects — train on 50% of subjects, evaluate
  on the remaining 50%.** Test subjects are then evaluated individually. Not LOSO. With 51
  subjects that is roughly 25/26 per fold.
- **Sessions: all sessions of a subject are concatenated into one recording.** So sessions are
  pooled, not held separate.
- **They use the RAW 59-channel 1000 Hz data**, not the 58-channel 250 Hz derivative
  (their dataset table lists Yang2025 as 59 ch / 1000 Hz).
- **Preprocessing: 2–47 Hz band-pass + average re-referencing.** Uniform across all nine
  datasets in the paper.
- **Optional per-subject covariance alignment (Euclidean Alignment) applied independently to
  each subject *before* aggregating the training subjects.** This is the leakage-safe form:
  statistics are per-subject, never global.
- **Pretraining:** 100 epochs, Adam, lr 1×10⁻⁴, batch 64, weight decay 0, cross-entropy.
- **Online adaptation** (their contribution, and directly relevant to this project's Phase 3/4):
  per trial — (1) optional unsupervised alignment, (2) predict, (3) supervised finetune on the
  true label. Unsupervised part = EMA-updated reference covariance (β = 0.9) whitening
  `X̃ = C_ref^{-1/2} X`, plus **AdaBN computed de novo from the single incoming trial**
  (explicitly the "BN-1" formulation). Supervised part = **20-trial warm-up with frozen
  weights**, then a **sliding window buffer of the 50 most recent trials**, finetuned
  **3 epochs** per incoming trial at lr 1×10⁻⁴. Two variants: **full finetuning** vs
  **decision-only finetuning** (freeze the conv feature extractor, update only the final FC
  layer, to limit catastrophic forgetting).
- **Metrics:** zero-shot accuracy (pretrained model on all trials of a new subject, no
  adaptation) and "final overall accuracy" as the main reported metric. The journal abstract
  describes the aggregate as *balanced* accuracy; the ablation table is labelled accuracy.
  **[medium]** on which of the two Table 1 uses.

**EDAPT results on Yang2025 (WBCIC-SHU 2C), mean ± std over test subjects: [high]**

| Model | PRE-ZS (zero-shot cross-subject) | CFT-only (no pretraining) | PRE+UDA | PRE+CFT | PRE+UDA+CFT |
|:---|---:|---:|---:|---:|---:|
| EEGNetv4 | **0.81 ± 0.12** | 0.78 ± 0.12 | 0.80 ± 0.12 | 0.84 ± 0.12 | 0.85 ± 0.12 |
| ShallowConvNet | **0.82 ± 0.12** | 0.73 ± 0.12 | 0.78 ± 0.11 | 0.82 ± 0.12 | 0.85 ± 0.11 |
| DeepConvNet | **0.85 ± 0.12** | 0.74 ± 0.12 | 0.83 ± 0.12 | 0.86 ± 0.11 | 0.88 ± 0.12 |
| ATCNet | **0.71 ± 0.12** | 0.59 ± 0.09 | 0.75 ± 0.11 | 0.78 ± 0.12 | 0.79 ± 0.12 |

Three findings from EDAPT worth carrying into a protocol discussion, all **[high]**:
1. **The PRE-ZS column is the cross-subject reference band for WBCIC-SHU: ~0.71–0.85
   depending on architecture.**
2. **Population pretraining is what matters most.** CFT-only (train from scratch per test
   subject) is *worse* than zero-shot pretrained on every architecture here.
3. **UDA alone is unreliable; supervised finetuning is the dependable gain.** PRE+UDA gave
   inconsistent effects (it helped ATCNet on Yang2025, and was flat or negative elsewhere),
   whereas PRE+CFT improved consistently and significantly. Their scaling analysis further
   shows accuracy rises monotonically with the *total* pretraining data budget (5→40 subjects,
   50→600 trials/subject) and is largely insensitive to how that budget is split between
   more subjects vs more trials per subject.

**(b) ATCNet-CIAM — uses WBCIC-SHU, but subject-dependent only. [medium — arXiv preprint,
not peer-reviewed]**

> *ATCNet-CIAM for Multi-Session Motor Imagery EEG Signal Classification.* arXiv:2607.23522 ·
> <https://arxiv.org/html/2607.23522>

It defines three **subject-dependent** protocols on WBCIC-MI, which are useful names to borrow:
- **"standard"**: sessions 1+2 train / session 3 test → 2C **85.39 ± 12.93**, κ 0.708 ± 0.259
- **"within-session"**: stratified 80/20 within session 1 → 2C **89.46 ± 10.47**, κ 0.789 ± 0.209
- **"cross-session"**: session 1 train / session 2 test, same subject → per-class diagonal drops
  to 0.81 for both 2C classes (the aggregate accuracy for this row was not legible in the HTML
  table I retrieved — **not found / unverified**)

It explicitly states: *"cross-subject generalization is an orthogonal direction that remains
important future work."* Its preprocessing practice is a good template regardless:
per-channel z-score with **statistics computed exclusively from the training data**;
**training-only** augmentation (random temporal shift up to ±50 samples, and within-class Mixup
with Beta(0.2, 0.2)); label smoothing 0.05; Adam with warmup, gradient clipping,
ReduceLROnPlateau; early stopping on validation accuracy; metrics accuracy + Cohen's κ +
macro-F1 + confusion matrices. It reads the 58-ch 4 s BIDS derivatives directly. **[high]** on
these details.

**(c) Papers that cite WBCIC-SHU but do not use it.** Worth knowing so you do not chase them:
the ICHORA 2026 paper *"A Robust EEGNet-Based Framework with Advanced Preprocessing for
Cross-Subject and Cross-Session Motor-Imagery BCI"*
(<https://doi.org/10.1109/ichora69329.2026.11537167>) sounds perfect but experiments on
**BCI IV-2a only** (LOSO: EEGNet-1D 83.35 ± 11.98, EEGNet-2D 84.04 ± 11.45). LPGGNet
(Brain Sci 15:1257, <https://doi.org/10.3390/brainsci15121257>) uses BCI IV-2a plus a private
lab dataset. The rest of the 18 are reviews, unrelated-paradigm papers, or other dataset
papers citing it as prior art. **[high]**

### 3.5 WBCIC-SHU gotchas

- **Subject-count inconsistency across official sources.** The paper says 51 subjects did 2C
  and 11 did 3C (= 62 total). The Figshare+ landing page says *"52 subjects completed the
  two-class MI experiment, while 11 subjects participated in the three-class"* (= 63, which
  also contradicts its own "sixty-two healthy participants" in the same paragraph). MOABB
  encodes 2C = subjects 1–51. **Trust the paper and MOABB: 51 subjects in 2C.** **[high]** on
  the discrepancy existing.
- **Pz is the reference and is therefore gone from the 58-channel data.** If a montage or a
  channel-subset selection assumes Pz exists, it will fail. **[high]**
- **The 58 vs 59 channel question depends on which release you load.** 58 = the preprocessed
  `.mat` derivative (Pz re-referenced). 59 = the raw BDF EEG channels. EDAPT used 59; the
  dataset paper's own benchmark used 58; this project uses 58. Not interchangeable. **[high]**
- **The session-1 → session-3 learning effect** (§3.3) is the biggest confound. **(assessment)**
- **No officially excluded subjects**, but S9 and S16 regress in session 3 and would be the
  natural candidates if anyone ever proposes an exclusion rule. **[high]**
- **No recommended channel subset.** **[high]** — for a motor-cortex subset there is no
  WBCIC-specific precedent; the closest convention is Kwon et al.'s 20 motor channels on
  OpenBMI (§4.2).
- **No official train/test split.** Although the data comes from a 2019 competition, the data
  paper does not publish the competition's train/test partition or rules. **[high]** —
  I searched for competition rules and found none attached to the release.
- **Download friction:** a single 65.6 GB ZIP on Figshare+; the dataset page itself suggests
  falling back to "Version 3" for batched download. A NEMAR mirror and the MOABB loader are
  easier entry points. **[high]**
- **Class balance is not a problem here** (100/100 per session by design) — a genuine advantage
  over SHU 2022. **[high]**

---

## 4. Comparable-dataset conventions

### 4.1 The de facto standards, in one place

| Question | Dominant convention | Source |
|:---|:---|:---|
| Cross-subject split | **LOSO** (leave-one-subject-out), N folds for N subjects | Kwon 2019; BARN-DA; DSGNet; Junqueira 2024 — all **[high]** |
| Sessions of a subject | **pooled** (concatenated) on the training side; the test subject's data is also pooled unless a specific session is designated | MOABB `CrossSubjectEvaluation` docstring: *"trained on all subjects but one, concatenating sessions"* **[high]** |
| Validation set | **held-out source subjects** is the leakage-safe form; **random trials from training subjects** is also common | SHU CSA uses 3 held-out source subjects **[high]**; Kim et al. split source data 8:2 **[high]** |
| Target subject's data at model-selection time | **must never be used** in the zero-shot condition | MOABB `CrossSubjectMode.TRAIN` = 0% target **[high]** |
| Normalization | per-subject or per-session statistics (legitimate); **never** global statistics fitted over train+test | Junqueira 2024 recommends EA as standard **[high]**; ATCNet-CIAM fits z-score on train only **[high]** |
| Metric aggregation | mean ± std **over subjects** | Kwon 74.15 ± 15.83; BARN-DA per-subject tables; EDAPT ± over test subjects — **[high]** |

### 4.2 OpenBMI / Lee2019 (54 subjects) — the canonical large-scale cross-subject protocol

**Dataset:** Lee, M.-H. et al. *EEG dataset and OpenBMI toolbox for three BCI paradigms: an
investigation into BCI illiteracy.* GigaScience 8, giz002 (2019).
<https://doi.org/10.1093/gigascience/giz002>. 54 healthy subjects (ages 24–35, 25 female;
about half naive), 62 EEG + 4 EMG channels, 1000 Hz, BrainAmp, nasion-referenced, grounded at
Fpz, **two sessions on different days**, MI = left vs right hand. **[high]**

**Kwon, O.-Y., Lee, M.-H., Guan, C., Lee, S.-W.** *Subject-Independent Brain–Computer
Interfaces Based on Deep Convolutional Neural Networks.* IEEE TNNLS 31(10):3839–3852 (2019/2020).
<https://doi.org/10.1109/TNNLS.2019.2946869>. This is the protocol everyone cites. All **[high]**:

- **LOSO-CV.** For target subject *i*: train on **all 53 other subjects across both sessions =
  21,200 trials**. Because they expand each trial into 20 frequency bands, the CNN saw
  424,000 input samples (20 bands × 53 subjects × 200 trials × 2 sessions).
- **Test set = only the target subject's session-2 *online* data.** The same test data is used
  for both the subject-dependent and subject-independent comparisons, which is what makes the
  two directly comparable.
- **Channel subset: 20 motor-cortex electrodes** — FC-5/3/1/2/4/6, C-5/3/1/z/2/4/6,
  CP-5/3/1/z/2/4/6. **Downsampled to 100 Hz** (anti-aliased first), justified by only caring
  about <40 Hz.
- 30 candidate frequency bands, 20 selected; all 10 CSP spatial filters used; 20×20 covariance
  matrices zero-padded to 28×28 as CNN input.
- lr 1×10⁻⁵, 50% dropout, batch 100, Adam, 4 layers, 256-unit first FC layer, 3×3 filters,
  72,264,076 parameters.
- **⚠ Model selection gotcha:** early stopping at **20 epochs, fixed globally**, chosen from
  observed convergence — *no per-fold held-out validation subjects are described.*
  **(assessment, high)** This is a weakness of the canonical protocol, not a template to copy.
- **Results (mean ± std over 54 subjects):**

| Subject-**independent** | Accuracy | Subject-**dependent** | Accuracy |
|:---|---:|:---|---:|
| Pooled CSP | 65.65 ± 16.11 | CSP | 68.57 ± 17.57 |
| Fused model | 67.37 ± 16.01 | CSSP | 69.68 ± 18.53 |
| MR-FBCSP | 68.59 ± 15.28 | FBCSP | 70.59 ± 18.56 |
| **Proposed CNN** | **74.15 ± 15.83** | BSSFO | 71.02 ± 18.83 |

  The headline claim is that the subject-independent model beat all subject-dependent ones
  (paired t-tests p < 0.05); the ANOVA across subject-dependent methods and the proposal showed
  no significant difference (F(4,265) = 0.7373, p = 0.5672).
- **Cost:** 12 min training per target subject; 0.15 ms inference per trial.

**A second OpenBMI LOSO variant worth knowing** — *Subject-Independent BCI with Open-Set Subject
Recognition*, arXiv:2301.07894 (<https://doi.org/10.48550/arxiv.2301.07894>): LOSO; **source data
split 8:2 into train/validation**; **evaluation on only the 4th session of the test subject**;
downsampled to 250 Hz; and a **subject-count ablation** (11 / 21 / 31 / 54 subjects, with 5 runs
at 10 subjects, 3 at 20, 2 at 30, 1 at 53). Their reported accuracies rise from ~70–73% with
11 subjects to ~84–85% with 54, which is a clean demonstration that **the size of the source
pool is a first-order variable.** **[high]**

### 4.3 BCI Competition IV-2a / 2b

- **Official splits exist but are session-based, not subject-based.** 2a: session 1 train /
  session 2 test. 2b: sessions 1–2 (or 1–3) train / the rest test. **[high]**
- **For cross-subject there is no official split, so LOSO is the de facto standard.** **[high]**
- **LOSO reference band, 4-class BCI IV-2a** (BARN-DA Table 8, same code and protocol for all
  rows — a rare apples-to-apples table): **[high]**

| Method | LOSO accuracy (mean ± std over 9 subjects) |
|:---|---:|
| ConvNet | 57.50 ± 12.90 |
| SST-DPN | 57.87 ± 10.66 |
| ATCNet | 60.74 ± 12.40 |
| EEGNet | 61.07 ± 9.33 |
| CTNet | 62.00 ± 11.44 |
| BARN-DA | 65.49 ± 11.64 |

- **LOSO, 2-class BCI IV-2b:** BARN-DA 78.78%, next best CTNet 76.50%. **[high]**
- Window conventions in BARN-DA: 2a uses 2–6 s (4 s), 2b uses 3–7 s, BCIC III 4a downsampled to
  100 Hz. ATCNet-CIAM instead uses 1.5–6.0 s for 2a and 3.0–7.5 s for 2b (1125 samples).
  **There is no single agreed window** — state yours explicitly. **[high]**

### 4.4 Euclidean Alignment: the one preprocessing step the literature actively recommends

- **Junqueira, B. et al.** *A systematic evaluation of Euclidean alignment with deep learning for
  EEG decoding.* J Neural Eng (2024). <https://doi.org/10.1088/1741-2552/ad4f18> ·
  arXiv:2401.10746 (<https://arxiv.org/html/2401.10746v4>). LOSO with multiple source subjects.
  **EA improves target-subject decoding by 4.33% (pseudo-online) / 5.55% (offline) and reduces
  convergence time by more than 70%.** It improved the mean accuracy for **all** cross-subject
  models and datasets they evaluated. Their conclusion, quoted: *"we believe that Euclidean
  Alignment should be a standard pre-processing step when training cross-subject models."*
  **[high]**
- **Wu et al.** *Revisiting Euclidean alignment for transfer learning in EEG-based BCIs.*
  J Neural Eng (2025). <https://doi.org/10.1088/1741-2552/addd49>. Practical placement advice:
  put EA **between temporal filtering and the deep-learning module** (their TF-EA-EEGNet beats
  EA-EEGNet and plain EEGNet). Also reports EA lifting a 4-way task from 61.53% → 70.44%
  (+8.91%) in one configuration, and that EA alone contributes far more than Mixup. **[high]**
- Why EA is leakage-safe and why it matters for the protocol design: EA whitens each *recording*
  by its own mean spatial covariance. The statistics are computed **per subject from that
  subject's own trials only**, including the unseen test subject, and no labels are used — so it
  is legitimate even in a strict zero-shot protocol. **(assessment, high)** MOABB formalises the
  stricter variant where only the first 20% of each target session may be used for this
  (`TRAIN_AND_TARGET_UNLABELED_20P`) so that even the unlabeled alignment trials are excluded
  from scoring. **[high]**

### 4.5 MOABB: standardized protocol machinery worth adopting verbatim

**Benchmark paper:** Chevallier, S. et al. *The largest EEG-based BCI reproducibility study for
open science: the MOABB benchmark.* arXiv:2404.15319 (<https://arxiv.org/html/2404.15319>).
30 pipelines (11 raw, 13 Riemannian, 6 deep learning) × 36 datasets (14 MI). **[high]**

Two things from it that bear directly on expectations:
- **The published MOABB result tables are within-session (5-fold), *not* cross-subject.**
  The docs say so explicitly: *"the results are obtained using Within-Session evaluation."*
  So do not use <https://moabb.neurotechx.com/docs/paper_results.html> as a cross-subject band.
  For orientation, Lee2019_MI **within-session** left-vs-right: ACM+TS+SVM 83.05 ± 13.97,
  CSP+LDA 76.88 ± 17.41. **[high]**
- **Left-hand vs right-hand is systematically harder than right-hand vs feet**, consistently
  across raw, Riemannian and deep-learning pipeline families and across datasets. Both of this
  project's datasets are left-vs-right, i.e. the harder of the two binary MI tasks. **[high]**

**The code-level conventions** (`moabb/evaluations/`) are the most directly reusable artifact
in this whole report. All **[high]**:

- `CrossSubjectEvaluation` — *"Evaluate performance of the pipeline trained on all subjects but
  one, **concatenating sessions**."* Default splitter is `LeaveOneGroupOut` (= LOSO); setting
  `n_splits=k` switches it to `GroupKFold(n_splits=k)` — i.e. **k-fold grouped by subject is a
  first-class supported option, not a shortcut.** Scores are recorded per session.
- `CrossSubjectMode` enumerates exactly the "how much target data is allowed" axis:

| Mode | Target fraction | Labeled? | Meaning |
|:---|---:|:---|:---|
| `TRAIN` | 0.0 | – | strict zero-shot / calibration-free |
| `TRAIN_TRIALWISE` | 0.0 | – | zero-shot, one target trial per prediction call (strictly inductive) |
| `TRAIN_AND_TARGET_UNLABELED_20P` | 0.2 | no | first 20% of each target session usable for alignment; remaining 80% scored |
| `TRAIN_AND_TARGET_UNLABELED_50P` | 0.5 | no | as above, 50% |
| `TRAIN_AND_TARGET_UNLABELED_FULL` | 1.0 | no | **transductive** — adapts on the same unlabeled block that is scored |
| `TRAIN_AND_TARGET_LABELED_20P` | 0.2 | yes | supervised calibration |
| `TRAIN_AND_TARGET_LABELED_50P` | 0.5 | yes | supervised calibration |

  A guard in the code enforces `calibration_labeled=True` only for `calibration_size ≤ 0.5`.
- The MOABB tutorial states the reporting rule plainly: *"Recording the mode next to every
  result is essential because the two scores answer different questions and use different
  numbers of scored target trials."* **This is the single best piece of protocol hygiene advice
  in this report.**

---

## 5. Recommended protocol options to discuss with the advisor

**Common decisions that should hold across all options** (so the options differ only in split
geometry, and results stay comparable):

1. **Two separate axes, never conflated.** "Cross-subject" = the test subject's data was never
   seen in training. "Cross-session" = a different recording day. On WBCIC-SHU (3 sessions) and
   SHU 2022 (5 sessions) you can build a **2×2**: subject seen/unseen × session seen/unseen.
   Report the cell, not a single scalar.
2. **The target subject's data never touches validation or model selection** in the zero-shot
   condition. Validation comes from **held-out source subjects**, following the SHU authors'
   own CSA design (3 held-out source subjects) rather than random trials from training subjects.
3. **Normalization is per-recording, never global.** Per-session (or per-subject) z-score and/or
   Euclidean Alignment, each fitted on that recording's own trials only. Never fit
   channel statistics over the pooled train+test set. Report which.
4. **Include an Euclidean Alignment arm.** Junqueira et al. recommend it as a *standard* step
   for cross-subject models (+4.33% and >70% faster convergence); it is label-free and
   leakage-safe; and this project already has no-learning alignment machinery from Phase 2b.
5. **Metrics:** accuracy **and** balanced accuracy **and** Cohen's κ, plus macro-F1 and AUC.
   Balanced accuracy matters for SHU 2022 (variable trial counts, §2.5); κ makes results
   comparable to the WBCIC literature (ATCNet-CIAM reports κ). Aggregate as
   **mean ± std over subjects**, and report the per-subject table.
6. **Seeds: 5**, matching this project's existing Phase 1 standard, and report mean over seeds
   separately from spread over subjects — they are different quantities and collapsing them
   hides which one dominates.
7. **Name the target-data condition using MOABB's vocabulary** (`TRAIN`,
   `TRAIN_AND_TARGET_UNLABELED_20P`, …) even if the implementation is in-house.

**Compute cost, expressed in training runs** (deliberately not in wall-clock hours, which I
cannot estimate without a benchmark on this cluster). Assume 3 backbones — EEGNet,
DeepConvNet, FBCNet, matching this project's Phase 1 set — and 5 seeds:

| Option | Folds (WBCIC / SHU) | Runs (WBCIC) | Runs (SHU) | Total | Relative cost | Train-set size per fold (WBCIC) |
|:---|:---|---:|---:|---:|---:|---:|
| **A — LOSO** | 51 / 25 | 765 | 375 | **1,140** | **1.0×** | ~50 subjects ≈ 30,000 trials |
| **B — 5-fold subject-grouped** | 5 / 5 | 75 | 75 | **150** | **0.13× (≈7.6× cheaper)** | ~41 subjects ≈ 24,600 trials |
| **C — fixed 3-way subject partition** | 1 / 1 | 15 | 15 | **30** | **0.026× (≈38× cheaper)** | 31 subjects ≈ 18,600 trials |

(WBCIC 2C = 51 subjects × 3 sessions × 200 trials = 30,600 trials, of which this project has
148/153 sessions usable. SHU 2022 = 25 subjects × 5 sessions × ~90–100 trials ≈ 12,500.)

---

### Option A — Leave-one-subject-out, strict zero-shot, nested subject-level validation

**Design.** For each of the 51 (WBCIC) / 25 (SHU) subjects: that subject is the test subject
(all of their sessions). From the remaining N−1 subjects, hold out **5 (WBCIC) / 3 (SHU)** as
**validation subjects** — the SHU authors used 3, so matching that is defensible — and train on
the rest. Early stopping and any hyperparameter choice comes from the validation subjects only.
Target data: **0%** (`TRAIN`).

**Pros.** This is what reviewers expect and what DSGNet, BARN-DA, Kwon et al. and Junqueira et
al. all do. Maximum source data per fold (~50 subjects on WBCIC), which matters because both
Kwon-style ablations and EDAPT's scaling analysis show accuracy rising monotonically with source
pool size. Every subject gets a per-subject number, so the std over subjects is honest.

**Cons.** ~1,140 training runs. On WBCIC that is 765 runs, and each run trains on ~30k trials of
58×1000 float data — the dominant cost in the whole plan. Also, 51 folds × 5 seeds produces a
lot of near-duplicate models, i.e. the marginal information per GPU-hour is low.

**When to choose it.** If the cross-subject result is going into the paper as a headline claim,
or if the advisor wants a number comparable to published LOSO work.

---

### Option B — 5-fold subject-grouped (`GroupKFold`), strict zero-shot ← **recommended primary**

**Design.** Partition subjects into 5 disjoint folds (~10 test subjects per fold on WBCIC,
5 on SHU). For each fold: test on those subjects, and carve **20% of the training subjects**
(≈8 on WBCIC, ≈4 on SHU) as validation subjects. Target data: **0%** (`TRAIN`). Stratify the
fold assignment so folds are matched on each subject's within-session accuracy tier (using the
dataset paper's published per-subject figures), otherwise a fold that happens to collect
S9 + S16 will look anomalous.

**Pros.** ~7.6× cheaper than LOSO for 150 runs total, while **still giving every subject exactly
one held-out test score per seed** — so the per-subject table and the mean ± std over subjects
are computed the same way as in Option A, and the two are directly comparable. It is an
explicitly supported MOABB mode (`CrossSubjectEvaluation(n_splits=5)` → `GroupKFold`), so it is
not an ad-hoc shortcut. The freed compute buys the things that actually matter scientifically:
the EA arm, the calibration ladder, more seeds, and the adaptation experiments.

**Cons.** ~41 training subjects per fold instead of ~50, which by EDAPT's own scaling curve
should make results **slightly pessimistic** relative to LOSO. That is a conservative bias,
which is the safe direction, but it must be stated when comparing to published LOSO numbers.

**Why I recommend it as the primary.** The binding constraint here is compute, and the
scientific question this project is actually chasing (Phase 3: does adaptation recover the
cross-session/cross-subject gap?) needs *many conditions*, not one maximally-tight number.
Option B spends the compute on conditions. Then run Option A on **SHU 2022 only** (25 folds =
375 runs) as a verification that the 5-fold and LOSO numbers agree; if they agree on the small
dataset, the 5-fold number on WBCIC is credible.

---

### Option C — Fixed 3-way subject partition + a calibration ladder

**Design.** One fixed partition, e.g. WBCIC **31 train / 10 validation / 10 test** subjects;
SHU **15 / 5 / 5**. Fix it once, write the subject IDs into the config, never touch the test
subjects until the end. Then sweep the target-data axis on the test subjects:
**0% → 20% unlabeled → 50% unlabeled → 20% labeled → 50% labeled**, i.e. MOABB's
`TRAIN` / `..._UNLABELED_20P` / `..._UNLABELED_50P` / `..._LABELED_20P` / `..._LABELED_50P`.
This deliberately mirrors the SHU authors' own CSA adaptation ratios (0/30/50/70/90/100%) and
EDAPT's warm-up-then-finetune design.

**Pros.** Cheapest by far (30 runs for the zero-shot arm), so it is the right vehicle for fast
ablation iteration and for the many-condition adaptation sweep. In concrete trial counts the
ladder is legible to an advisor: on WBCIC a session is 200 trials, so 20% of one session ≈ 40
trials and 50% ≈ 100 trials — directly comparable to EDAPT's 20-trial warm-up plus 50-trial
sliding window. It also connects straight to the existing Phase 3 Oracle/T3A line, where the
question is precisely "how much target information is needed".

**Cons.** A 10-subject test set has high variance and no fold-level replication, so it cannot
carry a headline claim. Not comparable to LOSO literature. Risk of implicit overfitting to the
fixed test set across many iterations if discipline slips.

**When to choose it.** As the **development / ablation harness** alongside Option B, not as the
final reported protocol.

---

### Suggested combination

1. **Option C** as the development harness — build the runner, get the calibration ladder
   working, iterate on EA and normalization choices. 30–60 runs.
2. **Option B** for the reported main table on both datasets. 150 runs.
3. **Option A on SHU 2022 only** as a LOSO-vs-5-fold consistency check. 375 runs, and it is on
   the smaller dataset so each run is cheap.
4. Only escalate to **Option A on WBCIC** (765 runs) if a reviewer or the advisor requires LOSO
   for the headline number.

Total ≈ 555–585 runs versus 1,140 for LOSO-everything — roughly half the compute, with more
scientific conditions covered.

**Implementation note, per this project's own rules** (`AGENTS.md` §6): this needs a new
experiment layer — `code/experiments/<name>.py` (or a new runner in `code/runners.py`) plus
`code/configs/experiments/{phase_cross_subject,shu_phase_cross_subject}.yaml` — driven through
the single human entry point `python code/run.py --config code/configs/experiments/<phase>.yaml`.
The subject-level split must be a config field (fold assignment written out explicitly, not
generated from an unseeded shuffle) so that folds are reproducible across models and seeds.

---

## 6. Open questions for the advisor

1. **Does cross-subject belong in the current research chain at all, or is it a new axis?**
   The project's chain is drift diagnostic → baseline → multi-source → no-learning alignment →
   prototype drift → adaptation, all **cross-session**. Phase 3 (Oracle then T3A) is approved
   and cross-session. Adding cross-subject is a second generalization axis. Is it (a) a
   replacement, (b) a parallel track, or (c) a later extension after the Phase 3 Oracle verdict?
2. **Zero-shot or calibrated?** Kwon et al.'s framing is *calibration-free* (0% target). The SHU
   authors' own headline (78.9%) uses 100% of the target's training trials. EDAPT's whole point
   is that **20 warm-up trials + a 50-trial sliding window** recovers most of the gap. Which of
   these is the claim we want to make? This single decision determines the protocol.
3. **Pool sessions, or restrict to session 1?** On WBCIC-SHU the session-1 → session-3 accuracy
   rise (81.77% → 88.90%) is a *skill acquisition* effect, not just drift. Pooling all three
   sessions mixes three skill levels into the training distribution. EDAPT pooled. ATCNet-CIAM
   used session 1 only for its within-session protocol. Which do we want, and do we want a
   session-stratified cross-subject result (train on others' session 1 → test on target's
   session 1, etc.)?
4. **Full channel set, or a motor-cortex subset?** Kwon et al. used 20 motor channels on
   OpenBMI's 62. This project uses all 58 (WBCIC) / 32 (SHU). A subset would cut compute and
   might improve cross-subject transfer by removing subject-idiosyncratic non-motor channels,
   but there is **no published channel-subset recommendation for either of our datasets**.
   Worth a cheap ablation under Option C?
5. **Should Euclidean Alignment be mandatory or an ablation arm?** Junqueira et al. explicitly
   recommend it as standard for cross-subject. If we make it default, our numbers are not
   comparable to un-aligned baselines; if we make it an arm, it doubles that part of the grid.
6. **How do we handle SHU 2022's floor effect?** The authors' cross-session number (53.7%) is
   not significantly above chance. If cross-subject on SHU lands at 52–55%, we have a
   near-chance result. Do we (a) report it as a legitimate negative/floor result — which is
   scientifically honest and consistent with this project's existing SHU findings — (b) restrict
   SHU to the adaptation conditions where it does move, or (c) drop SHU from the cross-subject
   analysis and keep it as the cross-session dataset only (which is what BARN-DA chose to do)?
7. **We cannot match a published cross-subject number on either dataset. Is that acceptable?**
   For WBCIC-SHU there is exactly one comparable source (EDAPT's PRE-ZS: 0.81 EEGNet /
   0.85 DeepConvNet under a 2-fold subject split with 2–47 Hz filtering on the 59-ch raw data).
   For SHU 2022 there is **none** that I could verify. Options: (a) accept it and cite EDAPT as
   the only anchor; (b) reproduce EDAPT's exact preprocessing on WBCIC as an explicit
   calibration run, which would be a genuinely valuable and citable contribution; (c) obtain
   the DSGNet paper through institutional access to recover a SHU LOSO number.
8. **Do we want the official within-session benchmark reproduced as a reference row?** If so, we
   must decide what to do about the WBCIC official benchmark's stage-2 test-set stopping
   criterion (§3.3) — reproduce it faithfully and flag the leakage, or fix it and report a
   corrected (lower) reference number.
9. **Compute budget?** Options A/B/C differ by ~38× in training runs. A number of available
   GPU-hours turns this from a philosophical choice into an arithmetic one.

---

## 7. Reference list with URLs

**Dataset papers (primary)**

1. Ma, J., Yang, B., Qiu, W., Li, Y., Gao, S., Xia, X. (2022). *A large EEG dataset for studying
   cross-session variability in motor imagery brain-computer interface.* Scientific Data 9:531.
   <https://doi.org/10.1038/s41597-022-01647-1> · full text
   <https://pmc.ncbi.nlm.nih.gov/articles/PMC9436944/> · data
   <https://doi.org/10.6084/m9.figshare.19228725>
2. Yang, B., Rong, F., Xie, Y., Li, D., Zhang, J., Li, F., Shi, G., Gao, X. (2025). *A multi-day
   and high-quality EEG dataset for motor imagery brain-computer interface.* Scientific Data
   12:488. <https://doi.org/10.1038/s41597-025-04826-y> · data
   <https://doi.org/10.25452/figshare.plus.22671172> · mirror
   <https://doi.org/10.82901/nemar.nm000246> · loader
   <https://moabb.neurotechx.com/docs/generated/moabb.datasets.Yang2025.html>
3. Lee, M.-H. et al. (2019). *EEG dataset and OpenBMI toolbox for three BCI paradigms: an
   investigation into BCI illiteracy.* GigaScience 8:giz002.
   <https://doi.org/10.1093/gigascience/giz002>

**Cross-subject protocol references**

4. Kwon, O.-Y., Lee, M.-H., Guan, C., Lee, S.-W. (2019/2020). *Subject-Independent
   Brain–Computer Interfaces Based on Deep Convolutional Neural Networks.* IEEE TNNLS
   31(10):3839–3852. <https://doi.org/10.1109/TNNLS.2019.2946869>
5. *Subject-Independent Brain-Computer Interfaces with Open-Set Subject Recognition* (2023).
   arXiv:2301.07894. <https://doi.org/10.48550/arxiv.2301.07894>
6. Lou, X., Li, X., Meng, X., Li, H., et al. (2026). *Subject-Independent Deep Learning Framework
   for Motor Imagery Electroencephalogram Decoding in Neurorehabilitation* (DSGNet). IEEE JBHI,
   early access. <https://doi.org/10.1109/jbhi.2026.3689121> · open record
   <https://bura.brunel.ac.uk/handle/2438/33344>
   — *LOSO on OpenBMI + BCI IV-2a + SHU Version 5 + BCI IV-2b; per-dataset numbers not
   retrievable (paywalled).*

**Alignment / normalization**

7. Junqueira, B. et al. (2024). *A systematic evaluation of Euclidean alignment with deep
   learning for EEG decoding.* J Neural Eng. <https://doi.org/10.1088/1741-2552/ad4f18> ·
   arXiv <https://arxiv.org/html/2401.10746v4>
8. Wu, D. et al. (2025). *Revisiting Euclidean alignment for transfer learning in EEG-based
   brain–computer interfaces.* J Neural Eng. <https://doi.org/10.1088/1741-2552/addd49>
9. *Latent alignment in deep learning models for EEG decoding.* J Neural Eng (2025).
   <https://iopscience.iop.org/article/10.1088/1741-2552/adb336>

**Benchmarking framework**

10. Chevallier, S. et al. (2024). *The largest EEG-based BCI reproducibility study for open
    science: the MOABB benchmark.* arXiv:2404.15319. <https://arxiv.org/html/2404.15319> ·
    results <https://moabb.neurotechx.com/docs/paper_results.html> (**within-session only**)
11. MOABB evaluation source:
    <https://github.com/NeuroTechX/moabb/blob/develop/moabb/evaluations/evaluations.py> ·
    protocols <https://github.com/NeuroTechX/moabb/blob/develop/moabb/evaluations/protocols.py> ·
    `CrossSubjectMode` docs
    <https://moabb.neurotechx.com/docs/generated/moabb.evaluations.CrossSubjectMode.html> ·
    cross-subject transfer tutorial
    <https://moabb.neurotechx.com/docs/auto_examples/how_to_benchmark/plot_cross_subject_transfer_rpa.html>

**Papers using WBCIC-SHU 2025**

12. *EDAPT: towards calibration-free BCIs with continual online adaptation.* J Neural Eng (2026).
    <https://doi.org/10.1088/1741-2552/ae5689> · preprint
    <https://arxiv.org/html/2508.10474v1> · code <https://github.com/mackelab/EDAPT>
    — **the only verified cross-subject work on WBCIC-SHU.** (Full author list unverified.)
13. *ATCNet-CIAM for Multi-Session Motor Imagery EEG Signal Classification.* arXiv:2607.23522.
    <https://arxiv.org/html/2607.23522> — subject-dependent only; cross-subject left as future
    work. Preprint, not peer reviewed.
14. *A Robust EEGNet-Based Framework with Advanced Preprocessing for Cross-Subject and
    Cross-Session Motor-Imagery BCI.* ICHORA 2026.
    <https://doi.org/10.1109/ichora69329.2026.11537167> — **cites WBCIC-SHU but experiments on
    BCI IV-2a only** (LOSO 83.35 ± 11.98 / 84.04 ± 11.45).

**Papers using SHU 2022**

15. *Joint hybrid recursive feature elimination based channel selection and ResGCN for cross
    session MI recognition.* Scientific Reports 14 (2024).
    <https://doi.org/10.1038/s41598-024-73536-z> ·
    <https://pmc.ncbi.nlm.nih.gov/articles/PMC11464737/>
    — **90.03% is a within-subject randomly-shuffled split on 10 subjects; not a valid
    cross-session or cross-subject number.**
16. *A subject transfer neural network fuses Generator and Euclidean alignment for EEG-based
    motor imagery classification* (ST-GENN). J Neurosci Methods (2025).
    <https://pubmed.ncbi.nlm.nih.gov/40350042/> ·
    <https://www.sciencedirect.com/science/article/abs/pii/S0165027025001244>
    — **67.2% on SHU, single "golden subject" source, target data used to train the Generator;
    not zero-shot.**
17. *A Band-Aware Riemannian Network with Domain Adaptation for Motor Imagery EEG Signal
    Decoding* (BARN-DA). Brain Sciences 16(4):363 (2026).
    <https://doi.org/10.3390/brainsci16040363>
    — **SHU used for cross-session only (61.76%); cross-subject LOSO done on BCI IV-2a
    (65.49%), IV-2b (78.78%), III 4a (78.14%). Its BCI IV-2a LOSO table is a good
    apples-to-apples reference band.**
18. *EEG-MFTNet: An Enhanced EEGNet Architecture with Multi-Scale Temporal Convolutions and
    Transformer Fusion for Cross-Session Motor Imagery Decoding.* arXiv:2604.05843.
    <https://arxiv.org/html/2604.05843v1> — subject-dependent cross-session on SHU, 58.9%.
19. *Evaluation of Spatio-temporal Deep Learning for Hand Movement based on Motor Imagery
    Electroencephalography.* ICEECIT 2024.
    <https://doi.org/10.1109/iceecit63698.2024.10859806> — SHU, EEGNex 75.67% / ATCNet 76.05%;
    **protocol not stated in the abstract, unverified.**
20. *LPGGNet: Learning from Local–Partition–Global Graph Representations for Motor Imagery EEG
    Recognition.* Brain Sciences 15:1257 (2025). <https://doi.org/10.3390/brainsci15121257>
    — cites WBCIC-SHU; experiments on BCI IV-2a + a private lab dataset.

**Model architectures referenced for accuracy bands**

21. Lawhern, V. J. et al. (2018). *EEGNet: a compact convolutional neural network for EEG-based
    brain–computer interfaces.* J Neural Eng 15:056013.
    <https://doi.org/10.1088/1741-2552/aace8c>
22. Schirrmeister, R. T. et al. (2017). *Deep learning with convolutional neural networks for
    EEG decoding and visualization.* Human Brain Mapping 38:5391–5420.
    <https://doi.org/10.1002/hbm.23730>
23. Mane, R. et al. (2021). *FBCNet: A multi-view convolutional neural network for
    brain-computer interface.* arXiv:2104.01233. <https://doi.org/10.48550/arXiv.2104.01233>
24. Combrisson, E., Jerbi, K. (2015). *Exceeding chance level by chance: the caveat of
    theoretical chance levels in brain signal classification and statistical assessment of
    decoding accuracy.* J Neurosci Methods 250:126–136. — the basis for both dataset papers'
    stated chance levels. <https://doi.org/10.1016/j.jneumeth.2015.01.010>

**Unverified / low-confidence, listed for completeness only**

25. *Bayesian Test-Time Adaptation via Dirichlet feature projection and GMM-Driven Inference for
    Motor Imagery EEG Decoding* (BTTA-DG), ICLR 2026 submission, seen only via a third-party
    paper-note site:
    <https://en.papernotes.org/ICLR2026/self_supervised/bayesian_test-time_adaptation_via_dirichlet_feature_projection_and_gmm-driven_in/>
    — claims cross-subject LOSO on SHU MI with 10 runs per method, but the SHU number is not
    shown and I could not reach a primary source. **[low]**
