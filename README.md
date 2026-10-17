# QuantumLung — AI-Assisted Lung Screening & Radiology Decision Support

**Smart India Hackathon 2026 · Problem Statement 26139 · Team CodeNova**

A chest-CT decision-support platform that runs a **classical deep-learning branch**, a
**variational quantum branch** (PennyLane), and a **hybrid fusion** of the two — then reports what
each concluded, how much they agree, and precisely what the underlying data does and does not
support.

> ⚕️ **AI-assisted research and clinical decision-support tool. Not a replacement for professional
> medical diagnosis.**

---

## What makes this build different

Most work on this dataset reports 99–100% accuracy. We report **~90–94%**, and that is the
headline feature, not a shortfall.

| | |
|---|---|
| **Leakage found and fixed** | The dataset has 1,097 slices from only 110 patients but **no patient IDs**. We measured that a naive random split puts a near-duplicate of **~55% of test images** into training. We built pseudo-patient groups and split on them (verified zero overlap). |
| **Quantum advantage tested, not claimed** | Paired bootstrap on the held-out test split: **p = 0.69–0.71 → not significant**. We say so on the home page. |
| **Explainability validated, and it failed** | Grad-CAM++ scored against 197 real lesion boxes recovered from burned-in DeepLesion overlays: **hit rate ≈ chance**. That is exactly why we never call attention maps tumour boundaries. |
| **External data tested, and rejected** | We downloaded 50 LIDC-IDRI series (3 GB) and added 927 labelled slices, taking benign training images from 70 to **454 (6.5x)**. It made the model **significantly worse** on the unchanged test set (macro-F1 −0.095, p = 0.009). Domain shift beat the extra examples, so we did not deploy it — and we report it. |
| **Real segmentation, once real masks existed** | The platform refused segmentation until the LIDC download supplied radiologist contours. A U-Net now localises nodules **within ~0.17 mm when it detects them (Dice 0.705)** — but detects only **55%**, and we report both numbers, and it transfers (3.333x more firing on malignant than normal). Staging is still refused. |
| **Capabilities honestly refused** | No staging model (no stage labels). No U-Net (no masks). No subtype. The UI renders these as explicit "not available" cards. |

---

## Results (held-out test set, n = 156, group-aware split)

Deployed configuration: **EfficientNet-B3 backbone → Quantum-4q head**.

| Model | Type | Accuracy | F1 (macro) | 95% CI | ROC-AUC | Malignant sens. | ECE |
|---|---|---|---|---|---|---|---|
| Hybrid_meta_classifier | Hybrid | 96.8% | **92.5%** | 0.848–0.981 | 0.979 | **100%** | 0.009 |
| Hybrid_neural_fusion | Hybrid | 96.2% | 92.1% | 0.845–0.976 | 0.983 | 98.8% | 0.165 |
| CNN_DeepLearning | Deep learning | 96.2% | 91.7% | 0.835–0.973 | 0.979 | **100%** | 0.084 |
| **Quantum_4q** *(deployed)* | Quantum | 95.5% | 90.1% | 0.823–0.964 | 0.980 | 98.8% | 0.031 |
| RandomForest | Classical ML | 94.2% | 88.8% | 0.806–0.947 | 0.978 | **100%** | 0.112 |
| SVM_RBF | Classical ML | 93.6% | 87.8% | 0.803–0.943 | 0.976 | **100%** | 0.033 |
| XGBoost | Classical ML | 92.3% | 86.2% | 0.775–0.927 | 0.971 | 96.3% | 0.044 |
| Quantum_8q | Quantum | 89.7% | 82.1% | 0.731–0.893 | 0.962 | 92.6% | 0.100 |

On the DenseNet-121 backbone, SVM_RBF reached **93.5%** macro-F1 / 0.996 ROC-AUC.

**Malignant sensitivity is 98.8–100% across most models** — the clinically decisive metric, since
a missed cancer is the costliest error.

### Benign-class improvement (cost-sensitive decision policy)

Benign was the weak class (73.3% recall; all 4 errors were *lesions called normal* — the dangerous
direction). Threshold tuning on the 157-image validation split failed: it holds only 15 benign
images and cannot resolve a threshold. Tuning instead by **StratifiedGroupKFold CV over train+val**
(85 benign examples) found a normal-class penalty that works:

| Metric (test) | Before | After |
|---|---|---|
| Benign recall | 0.733 | **0.933** |
| Malignant recall | 0.988 | 0.988 *(unchanged)* |
| Normal recall | 0.967 | 0.917 |
| Macro-F1 | 0.9011 | **0.9140** |
| **Missed findings** (lesion called normal) | 4 | **1** |

The trade — ~5 points of normal specificity for 20 points of benign sensitivity — is the correct
direction for screening support. Policy and caveats: `reports/decision_policy.json`;
experiment: `scripts/08_improve_benign.py`.

### External data experiment — a negative result we kept

The decision policy above is a patch; the real constraint was 70 benign training images. So we
acquired more data from **LIDC-IDRI** via the TCIA public API.

The annotations (9 MB) were parsed **before** downloading any imaging, so the 50 series could be
chosen by nodule yield rather than at random — and deliberately balanced (32 benign-rich + 18
malignant-rich), because ranking purely by benign yield would have made every LIDC image benign
and handed the model a "looks like LIDC = benign" shortcut.

| | |
|---|---|
| Downloaded | 50 series · 3.04 GB · 25 min · 50 real patients |
| Extracted | **927 labelled slices** (625 benign, 302 malignant) |
| Excluded | 74 nodules with mean malignancy exactly 3 (indeterminate) |
| Training benign | **70 → 454 (6.5×)** |
| Val / test | **byte-identical, 100% IQ-OTH** — so a shortcut cannot inflate results |

**Result on the unchanged test set:**

| Branch | Baseline F1 | +LIDC F1 | Δ | p |
|---|---|---|---|---|
| CNN (after target-domain fine-tune) | 0.9167 | 0.8279 | **−0.090** | **0.013** |
| Hybrid_meta | 0.9247 | 0.8314 | **−0.095** | **0.009** |
| Quantum_4q | 0.9011 | 0.8712 | −0.031 | 0.349 |

Normal recall fell 0.950 → 0.817 — exactly where the identity probe (source ROC-AUC 0.9985)
predicted, since LIDC contributed no normals. Two-stage transfer (pretrain merged, fine-tune on
IQ-OTH at 0.3× LR) recovered only part of it.

**Decision: not deployed.** The shipped model is still trained on IQ-OTH alone. One genuine gain
was recorded — with the decision policy, the LIDC-trained CNN reached benign recall 1.000 and
**zero missed findings** — but with no macro-F1 improvement (p = 0.80) it did not justify shipping.

*More data is not automatically better data.* Full write-up: `reports/lidc_experiment.json`,
decision **D21** in `reports/decisions.md`.

### Tumour segmentation (U-Net on LIDC contours)

"Where exactly is it?" is answered by a **supervised U-Net**, not by the attention map.

| | |
|---|---|
| Training labels | 922 masks, ≥2-of-4 radiologist consensus contours |
| Split | patient-level, 35/7/8 patients, zero overlap |
| **Detection rate** | **55%** of annotated nodules found |
| **Dice when detected** | **0.705**, centre within **~0.17 mm** |
| Dice / IoU over all slices | 0.392 / 0.334 |
| Cross-domain | fires **3.333×** more on malignant than normal IQ-OTH images |
| False positives | still fires on **16%** of normal-labelled images |

Note the contrast with the classification experiment above: the LIDC-trained **classifier did not
transfer**, but the **segmenter does**. Nodule morphology is a local cue; whole-image
classification is not.

The dashboard shows the mask, per-lesion geometry (area, equivalent diameter, % of lung field,
laterality and zone) and per-lesion confidence — with detections outside the lung field discarded,
and the false-positive rate stated on the panel.

### External validation — the model does not generalise

Tested on **LIDC-IDRI**, a dataset the deployed classifier has **never seen** (different
country, hospitals, scanners and reconstruction pipeline; verified every training row is
`source=IQ-OTH`).

| | ROC-AUC on P(malignant) | 95% CI |
|---|---|---|
| In-domain (IQ-OTH held-out, n=96) | **1.000** | 1.00–1.00 |
| **External (LIDC, n=922, 50 patients)** | **0.488** | 0.45–0.53 |

**0.50 is chance, and the external interval straddles it.** On scans from other hospitals the
model has no measurable ability to separate benign from malignant. It labelled
76% of external slices malignant —
sensitivity 0.93 next to specificity 0.07,
the signature of a model answering "malignant" by default.

Perfect in-domain separation (1.000) alongside chance externally is the signature of learned
**dataset appearance**, not lesion morphology. Every other figure in this README is
**in-domain only**. Full analysis: decision **D26**.

### Domain-shift redesign — diagnosing and fixing the generalisation failure

External validation (D26) showed the whole-image model at chance on unseen hospitals. The
investigation that followed is the most substantial result in the project.

**Step 1 — where the dataset signature lives.** A linear probe on simple appearance
descriptors names the source dataset at **AUC 1.000** from a whole slice,
and still **0.937** from a random lung crop. Cropping alone does not fix it.

**Step 2 — harmonisation.** Eight transforms measured on two axes. Best reached domain AUC
0.857 — helpful, not sufficient. Shipped as `preprocess_harmonised()`.

**Step 3 — domain-adversarial training.** Four arms (baseline / harmonised / DANN /
both), identical seed and schedule. **All four stayed at or below chance externally.**
Reported as measured.

**Step 4 — the diagnosis.** Nodule **size alone** predicts LIDC malignancy at
**AUC 0.873**. The trained model's score correlates with size at
**ρ = +0.050 (p = 0.13)** — statistically uncorrelated.
*The model was never looking at the nodule.*

**Step 5 — the fix.** Classify the annotated nodule instead of the image, patient-level split
on real LIDC IDs:

| Model | AUC | 95% CI |
|---|---|---|
| **ROI features + SVM-RBF** | **0.9145** | 0.864–0.956 |
| ROI features + Quantum (8q) | 0.8756 | 0.819–0.927 |
| ROI CNN (EfficientNet-B3) | 0.6797 | 0.583–0.765 |
| *Whole-image model, same data* | *0.4566* | |

**0.46 → 0.91.** Two secondary findings, both reported: a 10-feature interpretable model
**beats EfficientNet-B3** (35 training patients is too few for a deep net), and the quantum
branch is **significantly worse** than classical (p = 0.010).

**The honest limit:** 0.91 is cross-*patient* within LIDC; 0.46 is cross-*dataset*. They are
different evaluations and the tables label them so. Cross-dataset ROI validation is not yet
possible because IQ-OTH has no nodule annotations.

Full chain: decision **D27**. Live view: **Dashboard → Domain Shift**.

### Overfitting: what fixed it (and what did not)

The ROI CNN carried 10.7M parameters against 632 crops from 35 patients. Six strategies were
compared under **grouped 5-fold CV over all 50 patients** (the earlier 8-patient split was too
small to rank anything):

| Strategy | AUC | 95% CI |
|---|---|---|
| **10 hand-crafted features (SVM)** | **0.8617** | 0.836–0.887 |
| MobileNetV3-Small (1.5M params) | 0.7853 | 0.752–0.814 |
| Frozen embedding + features | 0.7395 | 0.706–0.772 |
| EfficientNet-B3 (baseline) | 0.7263 | 0.692–0.760 |
| Frozen encoder + linear probe | 0.6112 | 0.573–0.651 |
| Heavy regularisation | 0.5548 | 0.516–0.596 |

**Worked:** cutting capacity 7× (+0.059), and dropping the CNN entirely for 10
interpretable features. **Did not work:** heavier regularisation (worst arm), frozen ImageNet
features, naive fusion. See decision **D28**.

### Model readiness

**Level 2 of 5 — Internally validated.** external ROC-AUC 0.488, 95% CI 0.449-0.528 - does not exclude chance (0.50)
Not clinically deployable. Assessed from artefacts by `src/evaluation/readiness.py`.

**Quantum advantage test:** Δmacro-F1 = +0.012 (p = 0.711) on EfficientNet-B3 and −0.013
(p = 0.694) on DenseNet-121. **No advantage demonstrated.**

---

---

## Running the platform

The clinical platform is a Next.js application backed by the FastAPI service.

```bash
python scripts/start_platform.py          # dev, opens the browser
python scripts/start_platform.py --prod   # production build
```

| Service | URL |
|---|---|
| Clinical platform | http://localhost:3000 |
| API | http://127.0.0.1:8000 |
| API docs | http://127.0.0.1:8000/docs |

Seed synthetic demonstration data (real inference, fictional patients):

```bash
python scripts/19_seed_demo_data.py --patients 14
```

### Application

| Page | Purpose |
|---|---|
| `/` | Product landing page |
| `/dashboard` | Overview — workload, triage status, activity |
| `/dashboard/analyze` | **Radiology workspace** — viewer, AI visualisation, findings |
| `/dashboard/queue` | AI priority queue with filtering and sorting |
| `/dashboard/patients` | Patient list · `/patients/[id]` timeline + scan comparison |
| `/dashboard/follow-ups` | Clinician-set review dates |
| `/dashboard/reports` | Preliminary report + review workflow + clinician assessment |
| `/dashboard/analytics` | Model comparison from real experimental results |
| `/dashboard/technology` | Quantum circuit, pipeline, measured findings |
| `/dashboard/settings` | Deployed config, security posture, role model |

### Architecture

```
web/                Next.js 16 · TypeScript · Tailwind 4 · lucide-react
  src/app/          routes (landing + dashboard)
  src/components/   AppShell, design-system primitives
  src/lib/          typed API client, domain types, utilities

api/
  main.py           app wiring, platform endpoints
  routers/          clinical workflow endpoints
  services/         business logic (triage, persistence, reporting)
  db/               SQLAlchemy models + session

src/                ML core — UNCHANGED by the platform build
```

**Database:** SQLite at `data/app/quantumlung.db`. Entities: Organization, User,
Patient, Scan, Analysis, Finding, Report, DoctorFeedback, FollowUp, ModelVersion,
AuditLog. Portable to Postgres unchanged.

### Product guarantees carried into the UI

- Patient records store an **opaque identifier only** — no names, dates of birth or contact details.
- DICOM patient-identifying tags are **never** surfaced in API responses.
- Demo records are flagged **in the database**, not just labelled in the interface.
- **Staging and subtype remain disabled** with a stated reason on every report.
- Grad-CAM is never called segmentation; the U-Net mask is never called a confirmed boundary.
- **Detection rate is always shown next to Dice** — a single Dice number misleads.
- Clinician feedback is **stored, never auto-retrained** into a production model.
- Follow-up intervals are **clinician-set**; the platform proposes none.
- Scan comparison reports **"observed imaging and model differences"**, never progression.
- No HIPAA / medical-device / clinical-validation claims are made anywhere.


---

## Setup

Requires **Python 3.11**. A CUDA GPU is optional but recommended.

```bash
git clone <this repo>  &&  cd sih-quantum-lung

python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/macOS

# GPU (CUDA 12.4) — install torch first:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CPU-only users can skip the line above.

pip install -r requirements.txt
```

Point the pipeline at your data by editing `RAW_SOURCE` in [`src/config.py`](src/config.py).
It currently expects:

```
<RAW_SOURCE>/
├── The IQ-OTHNCCD lung cancer dataset/
│   └── The IQ-OTHNCCD lung cancer dataset/
│       ├── Bengin cases/      (120 jpg)
│       ├── Malignant cases/   (561 jpg)
│       └── Normal cases/      (416 jpg)
└── Test cases/                (197 png, DeepLesion — optional)
```

---

## Run

### Just the web app (models already trained)

```bash
python scripts/serve.py
```

Then open **http://127.0.0.1:8000**.

### Full pipeline from scratch

```bash
python scripts/run_all.py
```

Or step by step:

```bash
python scripts/01_audit_dataset.py        # forensic audit          -> reports/audit_raw.json
python scripts/02_audit_leakage.py        # leakage + lesion boxes  -> reports/deeplesion_boxes.json
python scripts/03_clean_and_split.py      # clean + group-aware split
python scripts/04_train_cnn.py --backbone efficientnet_b3
python scripts/04_train_cnn.py --backbone densenet121 --tag densenet121
python scripts/05_train_branches.py --tag efficientnet_b3   # classical + quantum + fusion
python scripts/05_train_branches.py --tag densenet121
python scripts/06_model_selection.py      # significance-aware selection
python scripts/07_validate_localization.py # Grad-CAM vs real lesion boxes
python scripts/08_improve_benign.py       # benign-class improvement experiments
```

Optional — the external-data experiment (needs ~3 GB download, ~25 min):

```bash
python scripts/09_lidc_parse_annotations.py   # parse LIDC XML, rank series by nodule yield
python scripts/10_lidc_download.py            # download the selected 50 series from TCIA
python scripts/11_lidc_extract_slices.py      # DICOM -> lung-windowed labelled slices
python scripts/12_merge_and_probe.py          # merge into TRAIN only + dataset-identity probe
python scripts/13_compare_lidc_effect.py      # before/after with paired-bootstrap significance
python scripts/14_build_lidc_masks.py         # radiologist contours -> consensus masks
python scripts/15_train_unet.py               # U-Net segmentation + cross-domain transfer probe
python scripts/21_external_validation.py      # external validation on LIDC (D26)
```

Domain-shift investigation and redesign (D27):

```bash
python scripts/22_domain_signature_probe.py   # where the dataset signature lives
python scripts/23_harmonisation_search.py     # which preprocessing removes it
python scripts/24_train_domain_robust.py      # baseline / harmonised / DANN / both
python scripts/25_inversion_analysis.py       # the diagnosis: size confound
python scripts/26_roi_nodule_classifier.py    # nodule-centred ROI pipeline
python scripts/27_hybrid_quantum_roi.py       # classical vs quantum vs hybrid on ROI features
python scripts/28_build_final_comparison.py   # assembles Tables 1-3
```

Total runtime on an RTX 4050 (6 GB): **≈ 12 minutes**.

### Tests

```bash
python -m pytest tests/ -v      # 27 tests
```

---

## Pipeline

```
CT image (PNG / JPG / DICOM)
   ↓  decode · DICOM rescale slope-intercept · lung window · green-overlay inpaint
   ↓  bilateral denoise → CLAHE → aspect-preserving letterbox 300×300
   ↓
EfficientNet-B3 / DenseNet-121  ──→  256-d embedding  ──→  Grad-CAM++ attention
   ↓                                      ↓
   ↓                            StandardScaler → PCA(32)      [fit on TRAIN only]
   ↓                                 ↓                ↓
softmax head              classical heads      learned projection → tanh·π
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

---

## Project structure

```
data/
  quarantine/           49 flagged images (copies; originals untouched)
  external/lidc/        LIDC-IDRI subset: annotations, series zips, extracted slices
  splits/               splits.csv + split_manifest.json
models/                 checkpoints, embeddings, branch models  (+ _backup.pt copies)
src/
  config.py             all constants, seeds, hardware detection
  preprocessing/        DICOM + raster loading, CLAHE, letterbox, lung mask, measured params
  data/                 dataset, augmentation, loaders
  models/               EfficientNet/DenseNet wrapper + focal loss
  quantum/              PennyLane VQC (configurable qubits)
  explainability/       Grad-CAM++, attention region, pointing game
  evaluation/           metrics, bootstrap CIs, calibration
  inference/            end-to-end engine + clinical information module
api/main.py             FastAPI backend
web/                    Next.js clinical platform (primary UI)
frontend/               original single-page app (retained, served at :8000)
scripts/                01–07 pipeline + serve.py + run_all.py
reports/
  dataset_report.md     forensic audit
  decisions.md          19 documented decisions incl. 2 bugs found and fixed
  model_comparison.json all metrics + significance tests
  localization_validation.json
  experiments/          per-run hyperparameters, seeds, timings, metrics
tests/                  27 tests
```

---

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/analyze` | Analyze one CT image (multipart). Returns prediction, all branches, Grad-CAM images, measured params, clinical info, limitations. |
| `GET /api/health` | Engine + hardware status |
| `GET /api/meta` | Classes, deployed config, quantum circuit spec, unsupported capabilities |
| `GET /api/model-comparison` | Full metrics, CIs, significance tests |
| `GET /api/localization-validation` | Grad-CAM pointing-game results |
| `GET /api/education` | Cancer education content |
| `GET /api/dataset-report` | Dataset report markdown |
| `GET /api/experiments` | All experiment records |

Errors are explicit: `415` unsupported type · `422` undecodable image/DICOM · `413` too large ·
`503` models unavailable · `507` out of memory.

---

## Reproducibility

Seed **42** throughout (Python, NumPy, PyTorch, cuDNN deterministic). Splits are written to
`data/splits/splits.csv` and reused by every downstream stage. Every run writes hyperparameters,
seed, dataset version, timings and metrics to `reports/experiments/`.

---

## Known limitations

1. **Pseudo-patient grouping is a surrogate** for true patient IDs; residual leakage possible.
2. **8-bit JPEG training data** — HU densitometry unavailable.
3. **Only 100 usable benign images** — wide confidence intervals for that class.
4. **Single centre, single scanner, single country** — external generalisation unproven.
5. **2D slices only** — no volumetric context.
6. **Attention maps do not localise lesions on out-of-domain data** (measured, §D14).
7. **No staging, segmentation, subtype, or clinical metadata** — absent from the data.
8. **External data did not transfer.** Adding LIDC-IDRI slices degraded performance on the
   IQ-OTH domain (D21); the model is therefore trained on a single centre only, and its
   cross-domain generalisation is measured to be poor.
9. Not clinically validated. No regulatory clearance. Research prototype only.

---

**Team CodeNova** · Smart India Hackathon 2026 · Problem Statement 26139
*AI-assisted research and clinical decision-support tool. Not a replacement for professional medical diagnosis.*
