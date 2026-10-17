# QuantumLung — Complete Project Report

**Hybrid Quantum-Classical AI Platform for Early Lung Cancer Detection**
Smart India Hackathon 2026 · Problem Statement 26139 · Team CodeNova

Location: `C:\Users\alanf\sih-quantum-lung`
Everything below is measured, not estimated. Every figure traces to a JSON artefact in `reports/`.

---

## 1. What this project is

A chest-CT clinical decision-support platform with three parallel model branches — classical
deep learning, a variational quantum circuit, and a hybrid fusion of the two — wrapped in a
hospital-workflow web application.

Its distinguishing feature is not accuracy. It is that **every claim is measured, and the
measurements that came out badly are reported as prominently as the ones that came out well.**
Four separate negative results are surfaced inside the product.

---

## 2. Scale of the build

| Area | Lines | Files |
|---|---|---|
| `src/` — ML core | 2,790 | 22 |
| `api/` — backend | 1,445 | 10 |
| `scripts/` — pipeline | 6,326 | 32 |
| `tests/` | 631 | 2 |
| `web/src/` — frontend | 5,568 | 23 |
| **Total hand-written** | **16,760** | **89** |

- **30 numbered pipeline scripts** (`01`–`30`), each reproducible and idempotent
- **54 tests**, all passing
- **29 documented decisions** in `reports/decisions.md`
- **24 JSON result artefacts** in `reports/`
- **0** TypeScript errors, **0** ESLint problems, **0** browser console errors

Hardware: NVIDIA RTX 4050 Laptop (6.4 GB VRAM), CUDA 12.4, bfloat16.

---

## 3. The datasets

### 3.1 What was found (forensic audit, `scripts/01`–`02`)

The supplied folder contained **two unrelated datasets**, which had to be separated:

| Dataset | Files | Labels | Role |
|---|---|---|---|
| **IQ-OTH/NCCD** | 1,097 JPEG | normal / benign / malignant | training + evaluation |
| **"Test cases"** = **DeepLesion** | 197 PNG | none; lesion boxes burned in as green pixels | localisation validation only |

- 0 corrupt files · 43 exact duplicates · 210 near-duplicate pairs · 1 cross-label near-duplicate
- Class imbalance **4.7 : 1** (malignant 561 · normal 416 · benign 120)
- 8-bit JPEG — **Hounsfield Units are unrecoverable**

### 3.2 The critical finding — data leakage

The README states 1,097 slices come from **110 patients**, but **filenames carry no patient ID**.
Consecutive CT slices from one patient are near-identical.

**Measured:** under a naive random split, **81 of 617 similarity groups straddle train and test**,
and **~55% of test images have a near-duplicate sibling in training.**

That is the mechanism behind the 99–100% accuracies published on this dataset.

**Fix:** pseudo-patient grouping — a 16×16 perceptual-hash similarity graph, connected components
at Hamming distance ≤ 12, giving 617 groups. Splits are group-aware with **verified zero overlap**.
Threshold chosen by sweeping 4/8/12/16/20 and taking the knee.

### 3.3 Cleaning

Flag-and-quarantine, never delete. 49 images excluded (43 exact duplicates, 6 label-ambiguous);
**copies** placed in `data/quarantine/`, originals untouched. Final pool **1,048 images**,
split 735 / 157 / 156.

### 3.4 Externally acquired data (LIDC-IDRI)

Downloaded from TCIA's public API: **50 CT series, 3.04 GB, 25 minutes, 50 real patients.**

Efficiency detail: the 9 MB annotation package was parsed **before** downloading any imaging, so
series could be chosen by nodule yield rather than at random — and deliberately balanced
(32 benign-rich + 18 malignant-rich), because ranking purely by benign yield would have made
every LIDC image benign and handed the model a "looks like LIDC = benign" shortcut.

Yield: **266 benign + 111 malignant nodules → 927 labelled slices → 922 consensus masks.**

---

## 4. The machine-learning pipeline

```
CT image (PNG / JPG / DICOM)
   ↓  decode · DICOM rescale slope-intercept · lung window · green-overlay inpaint
   ↓  bilateral denoise → CLAHE → aspect-preserving letterbox 300×300
   ↓
EfficientNet-B3 / DenseNet-121  →  256-d embedding  →  Grad-CAM++ attention
   ↓                                      ↓
   ↓                          StandardScaler → PCA(32)      [fit on TRAIN only]
   ↓                               ↓                ↓
softmax head            classical heads      learned projection → tanh·π
                        RF·SVM·XGB·LogReg          ↓
                               ↓            angle encoding (RY)
                               ↓            3 × [RY·RZ + CNOT ring]
                               ↓            Pauli-Z ⟨Z⟩ measurement
                               └──────┬──────────┘
                                 hybrid fusion (4 strategies)
                                      ↓
                          adaptive router (validation-selected)
                                      ↓
                      risk · region · measured params · limits
```

### 4.1 Model selection — significance-aware

Naive `argmax(validation macro-F1)` proved actively misleading:
- On DenseNet-121, **four candidates tied at exactly 0.9470**
- On EfficientNet-B3, the quantum branch **won** validation (0.9593) then finished **last** on test

With 157 validation images, differences below ~0.05 are noise. The final rule:
1. argmax on validation
2. **paired bootstrap** (2,000 resamples) to find every candidate not significantly worse (p ≥ 0.05)
3. clinical tie-break within that set — malignant sensitivity, then macro-F1, then calibration

**A bug was found and fixed here:** the first tie-break read *test* calibration, leaking the
held-out split into selection. Every selection term is now a validation quantity.

### 4.2 In-domain results (IQ-OTH held-out test, n = 156)

| Model | Type | Accuracy | F1 (macro) | 95% CI | ROC-AUC | Malig. sens. | ECE |
|---|---|---|---|---|---|---|---|
| Hybrid_meta_classifier | Hybrid | 96.8% | **92.5%** | 0.848–0.981 | 0.979 | **100%** | 0.009 |
| Hybrid_neural_fusion | Hybrid | 96.2% | 92.1% | 0.845–0.976 | 0.983 | 98.8% | 0.165 |
| CNN_DeepLearning | Deep learning | 96.2% | 91.7% | 0.835–0.973 | 0.979 | **100%** | 0.084 |
| **Quantum_4q** *(deployed)* | Quantum | 95.5% | 90.1% | 0.823–0.964 | 0.980 | 98.8% | 0.031 |
| RandomForest | Classical ML | 94.2% | 88.8% | 0.806–0.947 | 0.978 | **100%** | 0.112 |
| SVM_RBF | Classical ML | 93.6% | 87.8% | 0.803–0.943 | 0.976 | **100%** | 0.033 |
| XGBoost | Classical ML | 92.3% | 86.2% | 0.775–0.927 | 0.971 | 96.3% | 0.044 |
| Quantum_8q | Quantum | 89.7% | 82.1% | 0.731–0.893 | 0.962 | 92.6% | 0.100 |

24 candidates evaluated across two backbones. **Malignant sensitivity 98.8–100%** — the clinically
decisive metric, since a missed cancer is the costliest error.

### 4.3 Quantum branch

PCA(32) → learned linear projection → tanh·π → angle encoding (RY) → 3 × [RY·RZ + CNOT ring] →
Pauli-Z expectation. Simulator (`default.qubit`), backprop differentiation.

| Qubits | Depth | Gates | Params | Val F1 | Test F1 | Train time |
|---|---|---|---|---|---|---|
| **4** | 19 | 37 | 24 | **0.9593** | 0.9011 | 26 s |
| 6 | 25 | 55 | 36 | 0.9397 | 0.8723 | 40 s |
| 8 | 31 | 73 | 48 | 0.9493 | 0.8206 | 71 s |

4 qubits won on both backbones. More qubits overfit at 735 training samples.

**Quantum advantage test:** paired bootstrap on the test split gave Δ macro-F1 = +0.012 (p = 0.711)
on EfficientNet-B3 and −0.013 (p = 0.694) on DenseNet-121. **No advantage demonstrated** — stated
on the home page.

### 4.4 Cost-sensitive decision policy

Benign was the weak class (73.3% recall; all 4 errors were lesions called *normal*). Threshold
tuning on the 157-image validation split failed — it holds only 15 benign images. Tuning by
**StratifiedGroupKFold CV over train+val** (85 benign examples) found a working normal-class penalty:

| Metric (test) | Before | After |
|---|---|---|
| Benign recall | 0.733 | **0.933** |
| Malignant recall | 0.988 | 0.988 (unchanged) |
| Normal recall | 0.967 | 0.917 |
| **Missed findings** | 4 | **1** |

---

## 5. Segmentation and explainability

### 5.1 U-Net nodule segmentation

Trained on LIDC radiologist contours with **≥2-of-4 reader consensus** (922 masks). Patient-level
split on real IDs. Nodules occupy a median **0.026% of the image**, so the loss is Dice-based —
plain BCE is minimised by predicting background everywhere.

**The single Dice number was decomposed, because it hides two different things:**

| Question | Answer |
|---|---|
| Detection rate (found it at all) | **55.4%** |
| Dice **when it detects** | **0.705** |
| Median centre error when detected | **0.24 px ≈ 0.17 mm** |
| Predicted/true area ratio | 1.06 |
| False positives | 0.35 blobs/slice |

**When it finds a nodule it outlines it to within a fifth of a millimetre. It finds about half.**

**The misses are blindness, not uncertainty:** median peak probability inside missed nodules is
**0.0003**, with 97% below 0.05. Threshold sweeps from 0.05–0.55 moved detection by under 3 points.

**Three fixes were tried and all failed:** threshold tuning, 2.5D adjacent-slice stacking
(detection identical to four decimals, 84% the same misses), and more slices per nodule (made it
worse — edge slices diluted the signal; rolled back from backup).

### 5.2 Grad-CAM++ — validated and found wanting

Scored against 197 real DeepLesion lesion boxes (green overlays inpainted out first so the model
couldn't cheat):

| Model | Hit rate (all 197) | Chance | Pulmonary subset (n=78) |
|---|---|---|---|
| EfficientNet-B3 | 0.005 | 0.0043 | **0.000** |
| DenseNet-121 | 0.000 | 0.0043 | **0.000** |

**No better than chance.** This is why attention maps are never labelled tumour boundaries
anywhere in the product — a unit test enforces it.

---

## 6. External validation — the decisive result

Tested on **LIDC-IDRI**: 922 slices, 50 patients, different hospitals, scanners, country and
reconstruction pipeline. Verified fair — every row of the training split has `source=IQ-OTH`.

| | ROC-AUC on P(malignant) | 95% CI |
|---|---|---|
| **In-domain** (IQ-OTH held-out, n=96) | **1.000** | 1.00–1.00 |
| **External** (LIDC, n=922) | **0.488** | **0.449–0.528** |
| Change | **−0.512** | |

**0.50 is chance, and the external interval straddles it.**

It labelled **77% of external slices malignant** — sensitivity 0.931 next to specificity 0.066,
the signature of a model answering "malignant" by default. It also called **163 slices (17.7%)
"normal"** when every one contained an annotated nodule.

**Perfect in-domain alongside chance externally is diagnostic:** the model learned dataset
appearance, not lesion morphology.

---

## 7. The domain-shift investigation and fix

### 7.1 Where the dataset signature lives

A linear probe on simple appearance descriptors, grouped CV:

| View | Domain ROC-AUC |
|---|---|
| Whole slice | **1.000** |
| Lung-field masked | 0.939 |
| Random lung-interior crop | **0.937** |

**Cropping alone does not remove it** — the signature is in the lung parenchyma itself.

### 7.2 Harmonisation search

Eight transforms measured on two axes simultaneously (domain separability down, signal up).
Best was lung-mask + quantile histogram matching at **domain AUC 0.857** — helpful, insufficient.
Shipped as `preprocess_harmonised()`.

### 7.3 Domain-adversarial training

Four arms, identical architecture / seed / schedule, one switch differing. DANN uses target
*images* for the discriminator but never target labels (unsupervised adaptation).

| Arm | In-domain | External | 95% CI |
|---|---|---|---|
| baseline | 1.0000 | **0.4605** | 0.422–0.500 |
| harmonised | 0.9890 | 0.3631 | 0.323–0.401 |
| DANN | 0.9973 | 0.4060 | 0.368–0.447 |
| harmonised + DANN | 0.9817 | 0.3530 | 0.313–0.392 |

**Both interventions made it worse.** Reported as measured.

### 7.4 The diagnosis

| | Value |
|---|---|
| Nodule **size alone** predicting LIDC malignancy | **AUC 0.873** |
| The trained model on the same data | 0.457 |
| Spearman(model score, nodule area) | **+0.050 (p = 0.13)** |

A single scalar — lesion size, genuine radiology — reaches 0.873. The network's score is
**statistically uncorrelated with it**.

**The model was never looking at the nodule. It was classifying the picture.**

### 7.5 The fix — classify the nodule

Nodule-centred ROI crops, patient-level split on real LIDC IDs, grouped 5-fold CV over all
50 patients (922 out-of-fold predictions):

| Strategy | AUC | 95% CI |
|---|---|---|
| **10 hand-crafted features** | **0.8617** | 0.836–0.887 |
| MobileNetV3-Small (1.5M params) | 0.7853 | 0.752–0.814 |
| Frozen embedding + features fused | 0.7395 | 0.706–0.773 |
| EfficientNet-B3 (10.7M params) | 0.7263 | 0.692–0.760 |
| Frozen encoder + linear probe | 0.6112 | 0.573–0.651 |
| Heavy regularisation | 0.5548 | 0.516–0.596 |

**Whole-image reference on the same data: 0.457.**

### 7.6 Overfitting — measured as a gap

| Model | Train AUC | Held-out AUC | Gap | Verdict |
|---|---|---|---|---|
| SVM-RBF (C=3) | 0.9486 | 0.8617 | +0.087 | mild |
| **Logistic regression** | 0.9000 | **0.8739** | **+0.026** | **well generalised** |
| SVM C=100 *(control)* | 0.9809 | 0.8136 | +0.167 | moderate |

Logistic regression is better on **both** axes, which reversed the model choice. The C=100 arm is
a positive control confirming the measurement detects overfitting when present.

**Learning curve (15 → 50 patients: 0.824 → 0.874) is still rising** — more annotated patients is
the highest-value next investment.

### 7.7 What worked, what didn't

**Worked:** cutting capacity 7× (+0.059), and abandoning the CNN for interpretable features (+0.076,
non-overlapping CIs).

**Did not work:** heavier regularisation (worst arm — cutout was destroying the lesion), frozen
ImageNet features, naive fusion. **Three of four textbook fixes made things worse.**

---

## 8. Hybrid quantum on ROI features

A 10-dimensional interpretable vector is the natural scale for a small circuit — one feature per
qubit, no lossy PCA.

| Model | AUC (held-out patients) | 95% CI |
|---|---|---|
| **SVM-RBF** | **0.9145** | 0.864–0.956 |
| RandomForest | 0.9083 | 0.861–0.949 |
| LogisticRegression | 0.9059 | 0.843–0.954 |
| Hybrid (SVM + quantum) | 0.8985 | 0.845–0.944 |
| **Quantum (8 qubits)** | **0.8756** | 0.819–0.927 |

**Quantum advantage test:** quantum vs best classical Δ = −0.0385, **p = 0.010** — significantly
**worse**. Hybrid Δ = −0.0159, p = 0.047 — also significantly worse. The circuit genuinely works
(0.876 is respectable); it is simply beaten.

---

## 9. Product build — clinical platform

### 9.1 Stack

**Frontend:** Next.js 16, React 19, TypeScript (strict), Tailwind 4, lucide-react, Recharts.
Custom design system — neutral surfaces, semantic colour only for meaning, one 4px spacing scale,
~20 reusable primitives. Risk is always icon + label + colour, never colour alone.

**Backend:** FastAPI, layered into `routers` / `services` / `db`. SQLAlchemy 2.0 + SQLite
(portable to Postgres unchanged).

**ML:** PyTorch 2.6 + CUDA 12.4, timm 1.0.29, PennyLane 0.45.1, scikit-learn 1.9, XGBoost 3.2,
OpenCV 5.0, pydicom 3.0.2, SHAP 0.51.

### 9.2 Database entities

Organization · User · Patient · Scan · Analysis · Finding · Report · DoctorFeedback · FollowUp ·
ModelVersion · **AuditLog** (append-only).

### 9.3 Pages (11 routes)

| Page | Purpose |
|---|---|
| `/` | Product landing page |
| `/dashboard` | Overview — workload, triage, activity |
| `/dashboard/analyze` | **Radiology workspace** — viewer, AI visualisation, findings |
| `/dashboard/queue` | AI priority queue, filter + sort |
| `/dashboard/patients` · `/[id]` | Patient list · timeline + scan comparison |
| `/dashboard/follow-ups` | Clinician-set review dates |
| `/dashboard/reports` | Preliminary report + review workflow + clinician assessment |
| `/dashboard/analytics` | Model comparison + external validation |
| `/dashboard/domain-shift` | The full investigation as a five-step argument |
| `/dashboard/technology` | Quantum circuit, pipeline, measured findings |
| `/dashboard/settings` | Deployed config, security posture, role model |

### 9.4 Report workflow

`AI generated → Doctor review → Doctor edit → Approve / Reject`. The AI-generated body is
**never overwritten**; clinician changes accumulate separately so the original stays auditable.
Feedback is **stored, never auto-retrained** — an unvalidated feedback loop on a clinical model
is unsafe.

### 9.5 Demo data

14 synthetic patients, 25+ analyses. Identifiers fictional; **every risk score was produced by the
real model on real CT images.** Flagged `is_demo` at the *database* level, not just in the UI.

---

## 10. Safety engineering

### 10.1 Input gate

Discovered when web-sourced images returned confident wrong answers.

**Finding 1 — resolution collapse.** 12 ground-truth *normal* images downscaled to 250×189
flipped to malignant **12/12**. Threshold measured: ≥400 px validated, 256–399 degraded,
<256 unreliable.

**Finding 2 — no reject option.** At a fine 512×512, the classifier labelled pure black "benign"
(0.87), pure white "benign" (0.90), random noise "benign" (0.82) and a gradient "malignant" (0.94).

**A check was built and deleted:** Mahalanobis OOD on the CNN embedding could not separate real CT
(median 19) from pure black (20.3) or noise (21.6). It raised false alarms on real scans while
passing blank ones — worse than nothing. Recorded so nobody rebuilds it.

**Shipped:** a two-part gate (resolution on the *original upload*; plausibility via lung field,
intensity spread, sharpness). Real scans pass; blank/tiny inputs are marked unreliable with a
prominent banner.

**Two bugs in my own code found here:** the quality check used `>= 128px` (far too lenient) *and*
inspected the processed 300×300 frame rather than the upload — so it always passed.

### 10.2 Privacy

Opaque patient identifiers only — no names, DOB or contact details modelled anywhere. DICOM
patient tags never surfaced (verified with a synthetic DICOM carrying a patient name). Append-only
audit log. Five-role model.

**Not claimed:** HIPAA/GDPR certification, medical device approval, clinical validation.

### 10.3 Model readiness

**Level 2 of 5 — Internally validated.** Blocked at Level 3 because the external CI (0.449–0.528)
includes chance. A test asserts the system *cannot* award Level 3 while that is true.

---

## 11. Capabilities deliberately refused

| Requested | Status | Reason |
|---|---|---|
| Cancer staging | **Permanently disabled** | no stage labels exist in any dataset |
| Histological subtype | **Permanently disabled** | no subtype labels |
| Lobe-level localisation | Refused | no lobe labels — would be guesswork |
| Multilingual patient view | Not built | needs professional medical translation; machine-translating clinical text is unsafe |
| Educational AI assistant | Not built | content exists at `/api/education`; conversational surface not built |

---

## 12. The five negative results

These are the project's credibility, and all are visible inside the product.

1. **Data leakage** — ~55% of test images had a near-duplicate in training under a naive split. Measured and removed.
2. **No quantum advantage** — p = 0.69–0.71 in-domain; **significantly worse** (p = 0.010) on ROI features.
3. **Grad-CAM at chance** — hit rate 0.000–0.005 against 197 real lesion boxes.
4. **External data made training worse** — adding 927 LIDC slices degraded macro-F1 significantly (p = 0.009). Not deployed.
5. **External validation failed** — AUC 0.488, CI straddles chance. The model does not generalise.

---

## 13. Bugs found and fixed in my own work

1. **fp16 divergence** — training hit NaN at epoch 3; switched to bfloat16 with fp32 loss.
2. **Test leakage in model selection** — the tie-break read *test* calibration; now validation-only.
3. **Duplicate React key** — "Clinical staging" listed twice.
4. **`window.location` navigation** — caused full page reloads; switched to the Next router.
5. **NaN in JSON responses** — a metrics file broke `/api/experiments`; added a sanitiser.
6. **Resolution check measuring the wrong thing** — inspected the processed frame, not the upload.
7. **Over-aggressive import stripping** — removed imports that were in use; caught by tsc and restored.

---

## 14. How to run it

```bash
cd C:\Users\alanf\sih-quantum-lung

# Full platform (backend + frontend, opens browser)
python scripts/start_platform.py

# Core ML pipeline from scratch (~12 min on the RTX 4050)
python scripts/run_all.py

# Including the external-data stages (~3 GB download, ~25 min)
python scripts/run_all.py --with-external

# Domain-shift investigation
python scripts/22_domain_signature_probe.py
python scripts/23_harmonisation_search.py
python scripts/24_train_domain_robust.py
python scripts/25_inversion_analysis.py
python scripts/26_roi_nodule_classifier.py
python scripts/27_hybrid_quantum_roi.py
python scripts/28_build_final_comparison.py
python scripts/29_overfitting_fixes.py
python scripts/30_overfitting_check.py

# Tests
python -m pytest tests/ -v        # 54 tests
```

| Service | URL |
|---|---|
| Clinical platform | http://localhost:3000 |
| API | http://127.0.0.1:8000 |
| API docs | http://127.0.0.1:8000/docs |

---

## 15. Honest current status

**What works:**
- Full pipeline runs end to end, ~0.2 s per image inference
- In-domain classification: macro-F1 0.90–0.93, malignant sensitivity 98.8–100%
- Nodule segmentation localises to ~0.17 mm when it detects (55% detection)
- ROI feature model: **0.874 held-out, well generalised** across unseen patients
- Professional clinical platform, 11 pages, real persistence and workflow

**What does not work:**
- **The deployed model does not generalise.** External AUC 0.488 — at chance.
- Grad-CAM does not localise out of domain.
- Segmentation misses ~45% of nodules.
- The quantum branch is not better than classical, and on ROI features is significantly worse.

**What is untested:**
- **The ROI model has never been evaluated cross-dataset.** IQ-OTH has no nodule annotations, so
  ROIs cannot be built for it. The 0.874 is cross-*patient*; the 0.488 is cross-*dataset*. These
  are different claims and are never conflated in the tables.

**The single highest-value next step:** annotate nodules in IQ-OTH, or download more LIDC patients
(50 of ~1,000 used; the learning curve is still rising) and test the ROI model cross-dataset. That
is the one experiment that would move readiness from Level 2 to Level 3.

---

## 16. Documentation index

| File | Contents |
|---|---|
| `README.md` | Setup, run instructions, headline results |
| `PROJECT_AUDIT.md` | Phase 1 audit before the platform build |
| `PROJECT_REPORT.md` | This document |
| `reports/dataset_report.md` | Full forensic dataset audit |
| `reports/decisions.md` | **29 documented decisions with reasoning** |
| `reports/final_report.md` | Phase-by-phase ML results |
| `reports/PHASE_SUMMARY.md` | Delivery status per phase |
| `reports/*.json` | 24 machine-readable result artefacts |

---

*AI-assisted research and clinical decision-support tool. Not a replacement for professional
medical diagnosis. This platform has not been clinically validated, holds no regulatory clearance,
and must not be used to make patient-care decisions.*
