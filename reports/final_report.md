# Final Project Report

**Hybrid Quantum-Classical AI Platform for Early Lung Cancer Detection**
**SIH 2026 · Problem Statement 26139 · Team CodeNova**

All figures below are measured on this machine (RTX 4050 Laptop, 6.4 GB VRAM, CUDA 12.4,
PyTorch 2.6.0, PennyLane 0.45.1). Nothing is estimated or copied from literature.

---

## Phase completion

| Phase | Status | Evidence |
|---|---|---|
| 1 · Dataset inspection | Complete | `reports/dataset_report.md`, `audit_raw.json` |
| 2 · Cleaning & preprocessing | Complete | `cleaning_log.csv`, 49 quarantined copies |
| 3 · Labelling & splitting | Complete | `data/splits/splits.csv`, zero group overlap verified |
| 4 · Classical baselines | Complete | RF, SVM, XGBoost, LogReg on CNN embeddings |
| 5 · Deep feature extraction | Complete | EfficientNet-B3 + DenseNet-121, 256-d embeddings |
| 6 · Segmentation / localization | **Adapted** | No masks in data → Grad-CAM++, quantitatively validated |
| 7 · Quantum ML | Complete | PennyLane VQC, 4/6/8-qubit sweep |
| 8 · Hybrid fusion | Complete | 4 strategies, validation-selected |
| 9 · Explainability | Complete | Grad-CAM++ + pointing-game validation |
| 10 · Website | Complete | FastAPI + SPA, 5 pages |
| 11 · Testing & validation | Complete | 27 tests pass; bootstrap CIs, calibration, significance tests |
| 12 · Documentation | Complete | README, dataset report, 19-entry decision log |

**Deliberately not built:** U-Net segmentation (no masks) and a staging model (no stage labels).
Both omissions are documented and surfaced in the UI as explicit "not available" states.

---

## 1 · Dataset

Two distinct datasets were found in the source folder and separated:

| Dataset | Files | Labels | Use |
|---|---|---|---|
| IQ-OTH/NCCD | 1,097 JPEG | 3-class (normal/benign/malignant) | Training + evaluation |
| "Test cases" = **DeepLesion** | 197 PNG | none; lesion boxes burned in as green pixels | Localization validation only |

- 0 corrupt files · 43 exact duplicates · 210 near-duplicate pairs · 1 cross-label near-duplicate
- Class imbalance **4.7 : 1** (malignant 561 · normal 416 · benign 120)
- After cleaning: **1,048 images** (49 excluded, quarantined as copies)
- Split: **735 / 157 / 156**, group-aware, seed 42, **verified zero group overlap**

**The leakage finding.** 1,097 slices come from 110 patients, but filenames carry no patient ID.
Measured: under a naive random split, **81 of 617 similarity groups straddle train and test**, and
**~55% of test images have a near-duplicate sibling in training**. This is the mechanism behind
the 99–100% accuracies commonly published on this dataset. We built pseudo-patient groups
(perceptual-hash connected components, Hamming ≤ 12) and split on them instead.

---

## 2 · Classical branch

| Backbone | Params | Val macro-F1 | Test macro-F1 | Test ROC-AUC | Malignant sens. | Train time |
|---|---|---|---|---|---|---|
| EfficientNet-B3 | 11.09 M | 0.9288 | 0.9167 | 0.979 | **1.000** | ~2.5 min |
| DenseNet-121 | 7.22 M | 0.9470 | 0.9239 | **0.997** | **1.000** | ~3.5 min |

EfficientNet-B3 test confusion (rows = true): `[[12,0,3],[0,81,0],[3,0,57]]`
DenseNet-121 test confusion: `[[14,0,1],[0,81,0],[5,0,55]]`

**Both backbones achieved 100% malignant sensitivity — zero missed cancers on the test set.**
Benign is the weak class, as expected from only 70 training images.

A first EfficientNet run diverged to NaN under fp16; switching to **bfloat16** with fp32 loss
fixed it (decision D8).

---

## 3 · Quantum branch

Pipeline: 256-d embedding → StandardScaler → PCA(32, 94.0% variance) → learned linear projection
→ tanh·π → angle encoding (RY) → 3 × [RY·RZ + CNOT ring] → Pauli-Z expectation.

| Qubits | Depth | Gates | Circuit params | Val F1 (EffNet) | Test F1 (EffNet) | Train time |
|---|---|---|---|---|---|---|
| **4** | 19 | 37 | 24 | **0.9593** | 0.9011 | 26 s |
| 6 | 25 | 55 | 36 | 0.9397 | 0.8723 | 40 s |
| 8 | 31 | 73 | 48 | 0.9493 | 0.8206 | 71 s |

**4 qubits won** on both backbones. More qubits meant more parameters, more overfitting on 735
samples, and ~3× simulation cost. Backend: `default.qubit` statevector simulator, backprop
differentiation. No real quantum hardware was used and no hardware-noise claims are made.

---

## 4 · Hybrid fusion

| Strategy | Val F1 | Test F1 | Note |
|---|---|---|---|
| Weighted average | 0.9593 | 0.9011 | **degenerated to w_classical = 0.00** — i.e. pure quantum, not true fusion |
| Neural fusion (MLP) | 0.9593 | 0.9214 | 10-d input, worst calibration (ECE 0.165) |
| **Meta-classifier** | 0.9444 | **0.9247** | best test F1, best calibration (ECE 0.009), 100% malignant sensitivity |
| Concatenation → XGBoost | 0.9593 | 0.8892 | — |

The weighted-average search selecting w = 0.00 is worth flagging honestly: that is not fusion, it
is the quantum branch alone. The meta-classifier is the only strategy that both fuses genuinely
and improves test performance.

---

## 5 · Model selection — and why argmax was wrong

Naive `argmax(validation macro-F1)` proved actively misleading:

- On DenseNet-121, **four candidates tied at exactly 0.9470**.
- On EfficientNet-B3, the quantum branch won validation (0.9593) then finished **last** on test (0.9011).

With 157 validation images, macro-F1 differences below ~0.05 are noise. Paired bootstrap
confirmed: **12 of 12** candidates (EfficientNet) and **10 of 12** (DenseNet) were statistically
indistinguishable from the validation leader at p ≥ 0.05.

Final rule: argmax → paired bootstrap tie set → clinical tie-break (malignant sensitivity, then
macro-F1, then calibration), **all computed on validation only**.

> A bug was found and fixed during this work: the first tie-break implementation read *test*
> calibration, leaking the held-out split into selection. Every selection term is now a
> validation quantity. The same fix was applied to the cross-backbone choice.

**Deployed: EfficientNet-B3 → Quantum_4q.**

---

## 6 · Quantum advantage test — negative

Paired bootstrap (2,000 resamples) on the **test** split, macro-F1:

| Backbone | Comparison | Δ macro-F1 | 95% CI | p | Verdict |
|---|---|---|---|---|---|
| EfficientNet-B3 | quantum vs best classical | +0.0122 | −0.055 … +0.078 | 0.711 | not significant |
| EfficientNet-B3 | hybrid vs best classical | +0.0122 | −0.055 … +0.078 | 0.711 | not significant |
| DenseNet-121 | quantum vs best classical | −0.0134 | −0.076 … +0.048 | 0.694 | not significant |
| DenseNet-121 | hybrid vs best classical | −0.0134 | −0.076 … +0.048 | 0.694 | not significant |

### **No quantum advantage was demonstrated on this dataset.**

The quantum branch is a real, trainable, working component whose performance is *comparable* to
classical models — not superior. This is stated on the home page, the model comparison page and
the quantum explainer page of the live application.

---

## 7 · Explainability validation — also negative

Grad-CAM++ attention was scored against 197 real lesion bounding boxes recovered from the
DeepLesion green overlays (which were inpainted out before inference, so the model never saw them).

| Model | Hit rate (all 197) | Chance | Hit rate (pulmonary, n=78) | Attention energy in box |
|---|---|---|---|---|
| EfficientNet-B3 | 0.005 | 0.0043 | **0.000** | 1.41× area share |
| DenseNet-121 | 0.000 | 0.0043 | **0.000** | 1.76× area share |

Only **39.6%** of DeepLesion boxes fall inside a detected lung field (the rest are abdominal or
mediastinal lesions the model was never trained on), so the pulmonary subset is the fair test —
and it still shows no localization ability.

**Conclusion:** on out-of-domain CT, the attention maps do **not** reliably localize lesions. They
indicate what influenced the prediction and nothing more. This is the empirical basis for the
strict language rule that attention is never described as a tumour boundary.

This is a limitation most projects never discover because they never test it.

---

## 8 · Calibration

Expected Calibration Error on the test split (10 bins):

- Best calibrated: `Hybrid_meta_classifier` **0.009**, `SVM_RBF` (DenseNet) **0.010**
- Deployed `Quantum_4q`: **0.031**
- Worst: `Hybrid_neural_fusion` (DenseNet) **0.413** — badly overconfident, correctly not deployed

---

## 9 · Engineering

- **Hardware:** auto-detected; CUDA + bfloat16 AMP when available, graceful CPU fallback.
- **Runtime:** full pipeline ≈ 12 min; single-image inference **0.4–0.6 s** end-to-end.
- **Reproducibility:** seed 42 everywhere; splits persisted; every run logs hyperparameters,
  seed, timings and metrics to `reports/experiments/`.
- **Error handling:** 415 unsupported type · 422 undecodable image/DICOM · 413 oversized ·
  503 models unavailable · 507 OOM. All verified.
- **DICOM:** rescale slope/intercept applied, MONOCHROME1 inversion handled, multi-frame handled,
  embedded window respected when plausible else standard lung window (C −600 / W 1500).
  **PHI is never surfaced** — verified with a synthetic DICOM carrying a patient name.
- **Tests:** 27 passing, including split-integrity, quantum gradient flow, ⟨Z⟩ ∈ [−1,1], and
  guards asserting the report never emits a stage claim or an un-negated tumour-boundary claim.

---

## 10 · Honest limitations

1. Pseudo-patient grouping is a surrogate for true patient IDs; residual leakage possible.
2. 8-bit JPEG training data — Hounsfield Units unrecoverable.
3. Only 100 usable benign images — wide confidence intervals for that class.
4. Single centre, single scanner, single country — external generalisation unproven.
5. 2D slices only — no volumetric context, which radiologists rely on.
6. Attention maps do not localize lesions on out-of-domain data (measured, §7).
7. No staging, segmentation, subtype, or clinical metadata — absent from the data.
8. Test set is 156 images; all metrics carry wide bootstrap intervals.
9. Not clinically validated, no regulatory clearance — research prototype only.

---

## Summary

The platform delivers the full mandated pipeline — imaging → cleaning → preprocessing →
augmentation → feature extraction → classical branch → quantum branch → hybrid fusion → routing →
explainability → web dashboard — with **~92% test macro-F1 and 98.8–100% malignant sensitivity**
under a leakage-controlled split.

Its distinguishing contribution is negative-result honesty: it **measured** the dataset leakage
that inflates published results, **tested** for quantum advantage and found none, and **validated**
its own explainability and found it wanting — then reported all three prominently in the product
rather than hiding them.

*AI-assisted research and clinical decision-support tool. Not a replacement for professional
medical diagnosis.*

---

## 11 · External data experiment (added after initial delivery)

**Question:** the benign class was weak because there were only 70 benign training images. Does
adding more data fix it?

**What we acquired.** Via the TCIA public NBIA API: the LIDC-IDRI annotation package (9 MB,
1,319 radiologist reading files) plus **50 CT series (3.04 GB, 25 min)**. Parsing the annotations
*before* downloading imaging let us choose series by nodule yield instead of at random, and the
selection was deliberately balanced (32 benign-rich + 18 malignant-rich) so that LIDC would not
become a synonym for "benign". Yield: **266 benign + 111 malignant nodules → 927 labelled slices
from 50 real patients.**

**Label policy.** LIDC malignancy is a 1–5 radiologist impression from up to 4 readers, not
pathology. Annotations were clustered in 3D, ratings averaged, and mapped mean<3 → benign,
mean>3 → malignant, **mean==3 → excluded** (74 nodules). No normals were taken from LIDC, since an
unannotated slice in a patient scanned for suspected pathology is not evidence of a healthy lung.

**Design.** LIDC entered **training only**; validation and test stayed byte-identical and 100%
IQ-OTH. Training benign went **70 → 454 (6.5×)**.

**Identity probe.** A source classifier on the CNN embedding separated the two datasets at
**ROC-AUC 0.9985**, with NMI(source; label) = 0.288 — the shortcut was fully available.

**Result — significantly worse:**

| Branch | Baseline F1 | +LIDC F1 | Δ | p |
|---|---|---|---|---|
| CNN (fine-tuned on target domain) | 0.9167 | 0.8279 | −0.090 | **0.013** |
| Hybrid_meta | 0.9247 | 0.8314 | −0.095 | **0.009** |
| Quantum_4q | 0.9011 | 0.8712 | −0.031 | 0.349 |

Normal recall fell 0.950 → 0.817, exactly where the probe predicted. Two-stage transfer
(pretrain merged → fine-tune on IQ-OTH at 0.3× LR) recovered part of the loss but stayed
significantly below baseline.

**One genuine benefit, recorded but insufficient:** with the D20 decision policy, the LIDC-trained
CNN reached **benign recall 1.000 with zero missed findings** (baseline 0.933, 1 missed) — but with
normal recall 0.667 and no macro-F1 gain (p = 0.80).

**Decision: not deployed.** The shipped model remains EfficientNet-B3 → Quantum_4q trained on
IQ-OTH alone. All artefacts are retained (`scripts/09`–`13`, `reports/lidc_experiment.json`,
decision D21) because the negative result is itself a finding, and it is surfaced on the Home and
About pages of the live application.

**Why it matters:** more data is not automatically better data. 927 slices from a *higher-quality*
source — true DICOM, real Hounsfield Units, genuine patient IDs, 4-reader consensus — made a
single-domain model measurably worse because the domain gap outweighed the extra examples. The
reusable lesson is the safeguard: freezing validation and test as pure target-domain data is what
made the measurement trustworthy enough to act on.
