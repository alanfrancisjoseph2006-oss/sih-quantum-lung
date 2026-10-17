# Dataset Forensic Report

**Project:** Hybrid Quantum-Classical AI Platform for Early Lung Cancer Detection
**Problem Statement:** SIH 2026 — PS-26139 · **Team:** CodeNova
**Source audited:** `C:\Users\alanf\OneDrive\Desktop\sih dataset`
**Audit scripts:** `scripts/01_audit_dataset.py`, `scripts/02_audit_leakage.py`, `scripts/03_clean_and_split.py`
**Status:** raw data was opened read-only; **nothing in the source folder was modified, moved, or deleted.**

---

## 1. Executive summary

The folder contains **two entirely different datasets** that had to be identified separately.
They serve different roles, and conflating them would have been a serious methodological error.

| # | Sub-dataset | Files | Format | Labels | Role in this project |
|---|---|---|---|---|---|
| A | **IQ-OTH/NCCD Lung Cancer Dataset** | 1,097 | JPG, 8-bit | 3-class: normal / benign / malignant | **Supervised training + evaluation** |
| B | **"Test cases"** — identified as **DeepLesion key-slices** | 197 | PNG, 8-bit RGB | **No class labels**; lesion **bounding boxes burned in as green overlay** | **Independent localization validation** (not classification training) |

**Total: 1,294 image files + 1 README `.txt`. Zero corrupt files.**

The single most important finding is a **severe near-duplicate leakage hazard in dataset A**
(Section 5). Handling it correctly is the difference between an honest result and the
inflated ~99% accuracies commonly reported on this dataset.

---

## 2. Sub-dataset A — IQ-OTH/NCCD

### 2.1 Provenance (from the bundled README)

Collected at the Iraq-Oncology Teaching Hospital / National Center for Cancer Diseases
over three months in autumn 2019. Siemens SOMATOM scanner, 120 kV, 1 mm slice thickness,
window width 350–1200 HU, window centre 50–600. **Originally DICOM**, but distributed here
as de-identified JPEGs. README states **1,190 images from 110 cases**
(40 malignant, 15 benign, 55 normal).

### 2.2 What is actually on disk

| Property | Finding |
|---|---|
| Files present | **1,097** (README claims 1,190 → **93 images absent from this distribution**) |
| Format | JPEG, `RGB` mode but **verified pixel-identical across R=G=B** → truly 8-bit grayscale |
| Bit depth | 8-bit. **Hounsfield Units are NOT recoverable** — already windowed and lossily compressed |
| Corrupt / unreadable | **0** |
| Near-blank / empty | **0** |

### 2.3 Class distribution — significant imbalance

| Class | Files | Share | Cases (per README) |
|---|---|---|---|
| Malignant | 561 | 51.1% | 40 |
| Normal | 416 | 37.9% | 55 |
| **Benign** | **120** | **10.9%** | 15 |

**Imbalance ratio 4.7 : 1** (malignant : benign). Benign is the minority class *and* the
hardest class clinically. This drives three design decisions: class-weighted loss,
macro-averaged metrics, and **recall/sensitivity prioritised over accuracy**.

### 2.4 Resolution distribution

| Resolution | Count |
|---|---|
| 512 × 512 | 1,036 |
| 623 × 512 | 31 |
| 801 × 512 | 28 |
| 768 × 768 | 1 |
| 511 × 404 | 1 |

61 images (5.6%) are non-square — these carry scanner side-panels. Flagged and recorded;
retained, since aspect-preserving resize handles them.

---

## 3. Labels — what exists and what does **not**

| Label type | Present? | Consequence |
|---|---|---|
| Cancer / non-cancer | Yes (3-class) | Classification head trained on this |
| Malignant vs benign | Yes | Supported |
| **Cancer stage (I–IV)** | **ABSENT** | **No stage model will be built or displayed.** README mentions patients were "in different stages" but **no per-image stage label exists anywhere in the distribution** |
| **Tumor segmentation masks** | **ABSENT** | **No U-Net trained on dataset A.** Grad-CAM attention is used instead, and is labelled as attention — never as a tumor boundary |
| Bounding boxes (dataset A) | Absent | — |
| Bounding boxes (dataset B) | **197, as burned-in green overlays** | Used for **Grad-CAM localization validation** |
| Patient IDs | **ABSENT in filenames** | **Critical** — see Section 5 |
| Clinical metadata (age/sex/smoking) | Absent | No multimodal fusion possible |
| DICOM headers | Absent (JPEG only) | DICOM *reading* is still implemented in the app for user uploads |

> **Explicit anti-hallucination statement.** This dataset supports **3-class image
> classification only**. It does **not** support clinical staging, subtype prediction,
> or supervised segmentation. Any such claim would be fabrication, and none is made
> anywhere in this platform.

---

## 4. Duplicates

Detected via MD5 (exact) and 16×16 perceptual hash (near-duplicate).

| Check | Result |
|---|---|
| Exact byte-identical duplicates | **43 redundant files** across 43 groups |
| Cross-label exact duplicates | **0** (good — no direct label contradiction) |
| Near-duplicate pairs (pHash ≤ 4) | **210 pairs** |
| **Cross-label near-duplicate pairs** | **1** — `Bengin case (33).jpg` ↔ `Normal case (83).jpg`, Hamming distance **2** |

That last row is a genuine **label-consistency concern**: two near-identical slices carry
opposite labels. Such images are quarantined rather than silently trusted.

---

## 5. THE CRITICAL FINDING — patient-level leakage

The IQ-OTH/NCCD README states the 1,097 slices come from only **110 patients**, i.e.
roughly **10 slices per patient**. Consecutive CT slices from one patient are nearly
identical images.

**But the filenames contain no patient identifier** — they are only
`Malignant case (1..561).jpg`, etc. There is therefore **no supplied way to group slices by
patient**, and a naive random split will scatter slices of the *same patient* across train
and test.

**Measured severity.** Under a naive random 80/20 split:

> **81 of 617 similarity groups straddle train and test, and ~55% of test images have a
> near-duplicate sibling in the training set.**

That is the mechanism by which published IQ-OTH/NCCD results reach 99–100% accuracy: the
model is largely being tested on images it has effectively already seen.

### Mitigation implemented

Since real patient IDs are unavailable, we construct a **pseudo-patient grouping**:
a perceptual-hash similarity graph over all 1,097 images, with **connected components at
Hamming distance ≤ 12** treated as one "pseudo-patient".

- Yields **617 groups** — deliberately more conservative (finer) than the true 110 patients.
- Splits are made **group-aware**: an entire group goes wholly to train, val, *or* test.
- **Verified group overlap between all split pairs = 0.**

This cannot perfectly reconstruct patient identity, and that limitation is stated openly in
the report and on the website. It does, however, remove the dominant leakage path, and
means our reported metrics are **honest and materially lower than the literature's** — by
design.

---

## 6. Sub-dataset B — "Test cases" (DeepLesion)

Filenames follow the exact **DeepLesion** convention
`{patient}_{study}_{series}_{slice}.png` — e.g. `000001_03_01_088.png`.

| Property | Finding |
|---|---|
| Files | 197 PNG, all 512 × 512 |
| Unique patient IDs | **77** (genuine, parsed from filenames) |
| Colour content | **All 197 contain pure-green (0,255,0) pixels** |
| Nature of the green | **Burned-in lesion bounding-box annotations** |
| Extracted boxes | **197** (`reports/deeplesion_boxes.json`) |
| Median box size | 23 × 22 px → median lesion area = **0.20%** of the image |
| Class labels | **None** |

### How this is used — and how it is *not*

- **Used as an independent lesion-localization validation set.** The green boxes give a
  real spatial ground truth, letting us **quantitatively** score whether Grad-CAM attention
  lands on the actual lesion (pointing-game hit-rate) instead of merely asserting that the
  heatmaps "look right".
- **Never used for classification training or accuracy reporting** — it has no class labels.
- **The burned-in boxes are a visual artifact.** Feeding these images to a classifier
  unmodified would let it cheat off the green box. Therefore the green overlay is
  **programmatically inpainted/removed before any model sees the pixels**; the box
  coordinates are retained separately as ground truth.

---

## 7. Dataset suitability verdict

| Intended capability | Supported? | Decision |
|---|---|---|
| Cancer classification (3-class) | **Yes** | Primary task |
| Tumor **segmentation** (U-Net) | **No masks** | **Not trained.** Documented omission; Grad-CAM substituted |
| Tumor **localization** | **Weakly** | Grad-CAM attention only, **validated** against dataset B boxes |
| Cancer **staging** | **No stage labels** | **Not implemented.** UI displays an explicit "cannot be determined" notice |
| Cancer subtype | No subtype labels | Not implemented |
| Patient-level splitting | **No patient IDs** | Pseudo-patient surrogate grouping (Section 5) |
| DICOM handling | Not in dataset | Implemented in the app for **user uploads** only |

---

## 8. Cleaning actions taken

Policy: **flag and quarantine — never delete.** Originals in OneDrive remain untouched;
quarantined items are *copies*.

| Flag | Count | Action |
|---|---|---|
| `EXACT_DUPLICATE` | 43 | Excluded from modelling (first occurrence retained) |
| `LABEL_AMBIGUOUS` (mixed-label similarity group) | 6 | Excluded from modelling |
| `NON_STANDARD_RESOLUTION` | 61 | **Kept**, noted only — resize handles it |
| Corrupt / blank | 0 | — |

**Clean modelling pool: 1,048 images** (malignant 547 · normal 401 · benign 100).
Full per-file rationale: `reports/cleaning_log.csv`. Quarantined copies: `data/quarantine/`.

---

## 9. Final splits (seed = 42, group-aware, stratified)

| Split | n | % | Benign | Malignant | Normal |
|---|---|---|---|---|---|
| Train | 735 | 70.1% | 70 | 384 | 281 |
| Validation | 157 | 15.0% | 15 | 82 | 60 |
| Test | 156 | 14.9% | 15 | 81 | 60 |

**Group overlap: train∩val = 0, train∩test = 0, val∩test = 0.**
Reproducible from `data/splits/splits.csv` + `split_manifest.json`.

---

## 10. Honest limitations

1. **No true patient IDs** — pseudo-patient grouping is a surrogate; residual leakage cannot be fully excluded.
2. **8-bit JPEG, not DICOM** — HU values are unrecoverable; true lung-windowing cannot be applied to dataset A. CLAHE on the pre-windowed image is used instead, and this deviation is documented.
3. **Only 100 usable benign images** — the minority class is small; benign metrics will carry wide confidence intervals.
4. **Single-centre, single-scanner, single-country** — external generalisation is unproven.
5. **93 images missing** relative to the README's stated 1,190.
6. **No staging, no masks, no subtype, no clinical metadata** — capabilities correspondingly absent from the platform.
7. **2D slices only** — no 3D volumetric context, which real radiologists rely on.

---

*Generated during Phase 1. All figures are measured, not estimated.*
