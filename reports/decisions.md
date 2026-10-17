# Engineering & Scientific Decision Log

**Project:** Hybrid Quantum-Classical AI Platform for Early Lung Cancer Detection
**SIH 2026 · PS 26139 · Team CodeNova**

Every non-obvious decision, and every deviation from the original brief, with the reason.

---

## D1 — Project lives outside OneDrive

**Decision:** Code and models in `C:\Users\alanf\sih-quantum-lung`, not inside the OneDrive folder.
**Why:** Model checkpoints are ~45 MB each and change often; OneDrive sync would thrash and could
lock files mid-write. The dataset in OneDrive is **read-only** to this project — never modified,
moved, or deleted.

---

## D2 — Two datasets, not one

**Finding:** The source folder contains two unrelated datasets.
- `The IQ-OTHNCCD lung cancer dataset/` — 1,097 labelled JPEGs, 3 classes.
- `Test cases/` — 197 PNGs, DeepLesion naming (`patient_study_series_slice.png`), **no class labels**,
  with **lesion bounding boxes burned in as pure-green pixels**.

**Decision:** Use them for completely different purposes. IQ-OTH/NCCD is the only supervised
training/evaluation data. DeepLesion is used *solely* as an independent **localization**
validation set.
**Why:** Using unlabelled images for classification metrics would be fabrication. Discarding them
would waste a genuine spatial ground truth.

---

## D3 — Green annotation overlays are inpainted before inference

**Decision:** `remove_green_overlay()` detects pure-green pixels, records the bounding box, then
inpaints the pixels (OpenCV Telea) before the image reaches any model.
**Why:** A burned-in box is a trivially learnable shortcut. A model shown these images would
"detect" the annotation, not the lesion. The box coordinates are retained *separately* as ground
truth, so we get validation without contamination.

---

## D4 — Pseudo-patient grouping (the most important decision in the project)

**Problem:** The dataset README states 1,097 slices come from only **110 patients**, but filenames
carry **no patient identifier**. Consecutive slices from one patient are near-identical.

**Measured impact:** Under a naive random 80/20 split, **81 of 617 similarity groups straddle
train and test**, and **~55% of test images have a near-duplicate sibling in training**.

**Decision:** Build a surrogate grouping — a 16×16 perceptual-hash similarity graph over all
images, with **connected components at Hamming distance ≤ 12** treated as one pseudo-patient
(617 groups). Splits are group-aware; verified **zero group overlap** across train/val/test.

**Threshold choice:** Swept 4/8/12/16/20. Lower thresholds under-merge (925 groups at ≤4, leaving
leakage); higher thresholds over-merge distinct patients and create label-impure groups (3 at ≤16).
**12** sits at the knee: 617 groups, only 2 label-impure.

**Honest limitation:** This is a surrogate, not true patient identity. Residual leakage cannot be
excluded. It is stated in the dataset report, the About page, and the Model Comparison page.

**Consequence:** Our metrics (macro-F1 ≈ 0.90–0.94) are **materially lower than the 99–100%**
commonly published on this dataset. That is the point.

---

## D5 — No HU-based lung windowing on the training data

**Brief asked for:** lung windowing.
**Reality:** Dataset A is 8-bit JPEG that has already been windowed and lossily compressed.
Hounsfield Units are **mathematically unrecoverable**.
**Decision:** Do *not* fake HU windowing on dataset A. Use CLAHE (clip 2.0, 8×8) on the
already-windowed image instead. For **genuine DICOM uploads**, the full correct path *is*
implemented: rescale slope/intercept → HU → embedded window if plausible, else standard lung
window (C = −600, W = 1500).
**Why:** Applying a "lung window" to data with no HU semantics would be a meaningless
transformation dressed up as rigour.

---

## D6 — Aspect-preserving letterbox instead of plain resize

**Decision:** `resize_letterbox()` scales the long edge and zero-pads.
**Why:** 61 images (5.6%) are non-square (623×512, 801×512). A naive stretch distorts lesion
morphology, and nodule shape/margin is diagnostically meaningful. Covered by a unit test.

---

## D7 — Horizontal flip is allowed; vertical flip is not

**Decision:** Train augmentation includes horizontal flip, ±12° rotation, ±6% translation,
0.92–1.08 scale, mild brightness/contrast/noise/gamma. **Excluded:** vertical flip, shear,
elastic warp, large rotation.
**Why:** Horizontal flip mirrors left/right lungs, which is anatomically plausible and materially
helps with only 735 training images. The classifier makes no laterality claim — laterality is
derived geometrically from Grad-CAM on the **un-flipped** image at inference. Vertical flips and
elastic warps create anatomically impossible chests or destroy the nodule being classified.
Val and test are **never** augmented.

---

## D8 — bfloat16, not float16 (a real bug fixed)

**Observed:** The first EfficientNet-B3 run diverged to **NaN** at full-finetune epoch 3 under
fp16 autocast + GradScaler, crashing evaluation.
**Decision:** Use **bfloat16** autocast (RTX 4050 is Ada, native bf16), compute the loss in fp32,
skip non-finite batches, and drop GradScaler (bf16 needs no loss scaling). Also hardened
`compute_metrics` to degrade gracefully on NaN rather than raising.
**Why:** bf16 has fp32's exponent range, so it does not overflow. Retraining was stable and
reached val macro-F1 0.9288.

---

## D9 — Focal loss + balanced sampler, but not class weights as well

**Decision:** `WeightedRandomSampler` handles class balance; `FocalLoss(γ=1.5)` handles example
difficulty. Class weights inside the loss are **not** additionally applied.
**Why:** Applying both double-corrects the 4.7:1 imbalance and over-weights the 70 benign training
images to the point of instability.

---

## D10 — Model selection is significance-aware (a second real bug fixed)

**Observed:** Naive `argmax(validation macro-F1)` was actively misleading:
- On DenseNet-121, **four candidates tied at exactly 0.9470**.
- On EfficientNet-B3, the quantum branch *won* validation (0.9593) then came **last** on test (0.9011).

With 157 validation images, macro-F1 differences below ~0.05 are noise.

**Decision:** `scripts/06_model_selection.py` implements:
1. argmax on validation;
2. **paired bootstrap** (2,000 resamples) to find every candidate *not significantly worse* (p ≥ 0.05);
3. a documented **clinical tie-break** within that set — malignant sensitivity, then macro-F1,
   then calibration error.

**Bug found and fixed during implementation:** the first version of the tie-break read
`test_calibration`, leaking the held-out split into selection. Now every selection term is a
**validation** quantity. The same significance-aware rule was extended to the cross-backbone
choice, which had also been naive argmax.

---

## D11 — Quantum advantage is tested, not asserted

**Decision:** Run an explicit paired-bootstrap test on the **test** split, hybrid/quantum vs the
best classical model, and publish the p-value in the UI regardless of outcome.

**Result:**
| Backbone | Comparison | Δ macro-F1 | 95% CI | p | Verdict |
|---|---|---|---|---|---|
| EfficientNet-B3 | quantum vs classical | +0.012 | −0.055 … +0.078 | 0.711 | **not significant** |
| EfficientNet-B3 | hybrid vs classical | +0.012 | −0.055 … +0.078 | 0.711 | **not significant** |
| DenseNet-121 | quantum vs classical | −0.013 | −0.076 … +0.048 | 0.694 | **not significant** |
| DenseNet-121 | hybrid vs classical | −0.013 | −0.076 … +0.048 | 0.694 | **not significant** |

**Conclusion stated everywhere in the product: no quantum advantage was demonstrated.** The
quantum branch is real, trainable and performs *comparably* — not better.

---

## D12 — 4 qubits deployed, after sweeping 4/6/8

**Finding:** On validation, 4 qubits ≥ 6 ≥ 8 for both backbones. 8 qubits was consistently worst
on test (0.821 / 0.875).
**Why:** With 735 training samples and 32 PCA inputs, more qubits means more circuit parameters
and more overfitting, plus ~3× the simulation time (71 s vs 26 s). Shallow and small wins here.

---

## D13 — No U-Net, no staging model

**Decision:** Neither is built.
**Why:** The dataset contains **no tumor masks** and **no stage labels**. The brief explicitly
forbids fabricating either. The API returns structured `segmentation.available = false` and
`staging.available = false` blocks with explanations, and the UI renders them as "not available"
cards. A unit test asserts the report never emits a stage claim (word-boundary regex) and never
makes an un-negated "confirmed tumour boundary" statement.

---

## D14 — Grad-CAM localization was quantitatively tested — and failed

**Decision:** Rather than showing heatmaps and asserting they work, score them against the 197
real DeepLesion boxes using the pointing game.

**Result (negative):**

| Model | All 197 lesions | Pulmonary subset (n = 78) |
|---|---|---|
| EfficientNet-B3 | hit rate 0.005 (chance 0.0043) | hit rate **0.000** |
| DenseNet-121 | hit rate 0.000 | hit rate **0.000** |
| Attention energy in box | 1.4–1.8× the box's area share | 1.34–1.42× |

Only **39.6%** of DeepLesion boxes fall inside a detected lung field — the rest are abdominal or
mediastinal lesions the model was never trained on — so the pulmonary subset is the fair test.
It still shows no localization ability.

**Interpretation:** On out-of-domain CT, Grad-CAM++ attention does **not** reliably localize
lesions. This is reported on the Home page, the About page, and in every analysis report.
It is the empirical justification for the language rule that attention maps are never called
tumour boundaries.

---

## D15 — Lobe-level anatomical location is refused

**Decision:** Report only laterality (left / right / central) and vertical zone (upper / mid /
lower), derived geometrically from the attention centroid.
**Why:** Lobe segmentation requires either a lobe-segmentation model or lobe labels. Neither
exists here. Given D14, even the coarse location is presented with an explicit caveat.
Laterality follows radiological convention (patient's left appears on the image right).

---

## D16 — Flag-and-quarantine, never delete

**Decision:** 49 images excluded from modelling (43 exact duplicates, 6 label-ambiguous);
**copies** placed in `data/quarantine/`, originals untouched. 61 non-standard-resolution images
were flagged but **kept**, since letterboxing handles them.
**Why:** The brief requires it, and silently deleting medical data destroys auditability.
Full per-file rationale in `reports/cleaning_log.csv`.

---

## D17 — CV scores are reported but excluded from selection

**Observed:** 5-fold CV on the training embeddings gives ~0.997 macro-F1 — implausibly high.
**Cause:** The embeddings come from a CNN that was itself trained on those same images, so CV
folds are not independent.
**Decision:** Record the numbers in `reports/experiments/` for transparency, but **never** use
them for model selection, and state the caveat on the Model Comparison page.

---

## D18 — Vanilla-JS frontend rather than React

**Decision:** Single-page app in plain ES2020 + CSS, served by FastAPI.
**Why:** No Node toolchain, no build step, no `node_modules` — the whole app starts with one
command, which matters for a live demo on an unfamiliar machine. All charts (architecture
diagram, circuit diagram, bootstrap-CI bar charts) are hand-built inline SVG, so there is no
chart-library dependency either.

---

## D19 — PHI is never surfaced from DICOM

**Decision:** The DICOM reader extracts only technical tags (Modality, Manufacturer,
SliceThickness, KVP, PixelSpacing, …). `PatientName`, `PatientID` and similar are deliberately
never read into the response.
**Why:** Even a research prototype should not echo patient identifiers into a web response.
Verified by an end-to-end test with a synthetic DICOM carrying a patient name.


---

## D20 — Cost-sensitive decision policy for the benign class

**Problem:** Test recall was malignant 0.988, normal 0.967, but **benign 0.733**. All four benign
errors were benign → **normal**, i.e. a lesion reported as "no finding" — the clinically dangerous
failure direction. Inspection showed three of the four sat at P(benign) ≈ 0.29–0.35 vs
P(normal) ≈ 0.65, so they were recoverable in principle.

**First attempt, rejected:** tuning a normal-class penalty on the validation split. Validation
holds only 15 benign images with a single error, so the objective was flat across the entire
sweep and selected w = 1.0 (no change). Choosing a weight because it helped the *test* set would
have been test-set fitting — precisely what D10 exists to prevent.

**What worked:** tune the penalty by **StratifiedGroupKFold (5-fold) cross-validation over
train + val**, which exposes ~85 benign examples instead of 15, with groups respected so the
pseudo-patient anti-leakage guarantee still holds. Selection additionally required, on validation,
that malignant recall not regress and normal recall stay ≥ 0.90.

**Selected:** `normal × 0.20`, applied to the routed prediction only.

**Measured effect on the held-out test split:**

| Metric | Before | After |
|---|---|---|
| Benign recall | 0.733 | **0.933** |
| Malignant recall | 0.988 | 0.988 |
| Normal recall | 0.967 | 0.917 |
| Macro-F1 | 0.9011 | **0.9140** |
| Missed findings | 4 | **1** |

**Also tried and beaten:** a two-stage cascade (finding-vs-normal → benign-vs-malignant),
benign-boosted cost-sensitive LogReg/RF heads at 2×/3×/4×, and two-backbone ensembles. All are
recorded in `reports/experiments/benign_improvement.json`.

**Honest caveats:** the CV objective curve was nearly flat (0.985 vs 0.984), so the exact weight is
weakly determined — the validation constraint check is what confirmed it. Everything rests on 15
benign validation and 15 benign test images, so intervals are wide. And this is a *decision-rule*
change, not a better-trained model: **the real fix is more benign training data.**

**Transparency:** individual branch cards in the UI still show raw, unadjusted model outputs. The
report states when the policy changed an outcome, and shows the raw class alongside the final one.


---

## D21 — External data (LIDC-IDRI) was acquired, tested, and NOT deployed

**Motivation:** D20 fixed the benign weakness with a decision threshold, but that is a patch.
The binding constraint was 70 benign training images, so we went and got more data.

### What was acquired

Via the TCIA public NBIA API (no credentials required for public collections):

- **LIDC-IDRI XML annotations** (9 MB): 1,319 radiologist reading files covering all 1,018 series.
- **50 CT series** (3.04 GB, 25 min at ~2 MB/s with 8 parallel workers).

**Key efficiency:** the annotations were parsed **before** downloading any imaging, so the 50
series could be chosen by nodule yield rather than at random.

### Selection was deliberately balanced, not greedy

Ranking purely by benign yield gave ~347 benign vs ~37 malignant nodules. Had we taken that,
*every* LIDC image would have been benign and the model could have learned
"looks like LIDC → benign" — a dataset shortcut worse than the imbalance we set out to fix.
So the subset was built from both ends: 32 series ranked by benign yield + 18 by malignant yield,
giving **266 benign + 111 malignant nodules → 927 labelled slices from 50 real patients**.

### Label mapping (differs from IQ-OTH, and is documented as such)

LIDC malignancy is a radiologist **impression on a 1–5 scale**, from up to 4 independent readers,
**not** pathology-confirmed. Annotations of the same physical nodule were clustered by 3D centroid
proximity, ratings averaged, then mapped: mean < 3 → benign, mean > 3 → malignant,
**mean == 3 → indeterminate and excluded entirely** (74 nodules dropped, never silently assigned).
Slices carrying both benign and malignant nodules were labelled malignant (worst-case, the
clinically safe direction). **No "normal" slices were taken from LIDC** — every LIDC patient was
scanned for suspected pathology, so an unannotated slice is not evidence of a normal lung.

### Experimental design that kept the result honest

LIDC entered the **training split only**. Validation and test remained **byte-identical, 100%
IQ-OTH**. That makes results directly comparable to the pre-LIDC numbers *and* removes the payoff
for learning a dataset shortcut — knowing "this looks like LIDC" is worthless on a pure IQ-OTH
test set.

Training set after merge: **benign 70 → 454 (6.5×)**, malignant 384 → 593, normal 281 unchanged.
Group overlap across splits verified zero (LIDC patients use real patient IDs, offset by 100000).

### Dataset-identity probe

A source classifier on the CNN embedding separated IQ-OTH from LIDC at **ROC-AUC 0.9985**
(LIDC's circular field-of-view is unmistakable), with **NMI(source; label) = 0.288**. The shortcut
was therefore fully available to the model.

### Result: it made things WORSE, significantly

Measured on the unchanged IQ-OTH test set (n = 156), paired bootstrap, 2,000 resamples:

| Branch | Baseline F1 | +LIDC F1 | delta | p | Verdict |
|---|---|---|---|---|---|
| CNN | 0.9167 | 0.8041 | −0.113 | — | worse |
| CNN (after target-domain fine-tune) | 0.9167 | 0.8279 | **−0.090** | **0.013** | **significantly worse** |
| Hybrid_meta | 0.9247 | 0.8314 | **−0.095** | **0.009** | **significantly worse** |
| Quantum_4q | 0.9011 | 0.8712 | −0.031 | 0.349 | not significant |

Per-class, the damage lands where the probe predicted: **normal recall 0.950 → 0.817**. LIDC
contributed no normals, so the model's notion of "normal" was diluted and shifted toward the
LIDC domain.

**The principled fix was tried and was not enough.** Two-stage transfer — pretrain on the merged
multi-domain set, then fine-tune on IQ-OTH only at 0.3× learning rate — recovered part of the loss
(F1 0.804 → 0.828, benign 0.60 → 0.73) but stayed significantly below baseline.

### One real benefit, recorded but not sufficient

With the D20 decision policy applied, the LIDC-trained CNN reached **benign recall 1.000 with
ZERO missed findings** (baseline: 0.933, 1 missed). That is a genuine gain on the metric that
matters most clinically — but it came with normal recall at 0.667 and no macro-F1 improvement
(0.790 → 0.798, p = 0.80), so it does not justify deployment on its own.

### Decision

**The deployed model is unchanged: EfficientNet-B3 → Quantum_4q trained on IQ-OTH only.**
All LIDC artefacts, scripts and metrics are retained (`scripts/09`–`13`,
`reports/experiments/lidc_effect.json`) because the negative result is itself a finding.

### Why this matters more than a win would have

More data is not automatically better data. 927 additional labelled slices from a higher-quality
source (true DICOM, real Hounsfield Units, genuine patient IDs, 4-reader consensus) made a
single-domain model measurably worse, because the domain gap outweighed the extra examples. We
measured that instead of assuming it, and we report it instead of quietly keeping the better
number. The safeguard that made the measurement trustworthy — freezing val and test as pure
target-domain data — is the reusable lesson.

**What would likely be needed to make it work:** domain adaptation (histogram/appearance matching,
or adversarial domain-confusion), sourcing normal slices for the external domain so the classes
are not domain-correlated, or enough external data to train a genuinely domain-invariant
representation. None of those fit in the remaining hackathon scope.


---

## D22 — U-Net segmentation, built once real masks existed

**Context.** Until now the platform refused segmentation, correctly: IQ-OTH/NCCD has no tumour
masks (D13), and Grad-CAM was measured to be no better than chance at localisation (D14). The
honest answer to "where is the tumour?" was *we cannot tell you*.

The LIDC download (D21) changed the input. Its XML contains **per-slice polygon contours** drawn
by up to four radiologists — genuine pixel-level ground truth. That is a different kind of label
from anything previously available, so segmentation became legitimately buildable.

### Masks

Each reader's `<edgeMap>` polygon was rasterised separately, then combined with a **≥2-reader
consensus** rule (the standard 50% convention) so the model trains on agreement rather than one
radiologist's opinion. Result: **922 masks
from 50 patients**. Nodules occupy a **median 0.026% of the image** — roughly 68 pixels — which is
why the loss is Dice-based: plain BCE is minimised almost perfectly by predicting background
everywhere.

### Model and split

`UNet(base=24, depth=4)`, 4.368M parameters, trained at 512×512 (native LIDC
resolution, so tiny nodules stay resolvable). Loss = Dice + 0.5·BCE(pos_weight=20).
**Patient-level split on real LIDC patient IDs**: 35/7/8
patients, zero overlap asserted. The operating threshold (0.35)
was tuned on validation only.

### In-domain result (the number we are entitled to quote)

| Metric | Held-out LIDC patients |
|---|---|
| **Dice** | **0.3924** |
| **IoU** | **0.3335** |
| Slices / patients | 168 / 8 |

Modest by the standards of the LIDC segmentation literature (~0.5–0.8), and that gap is expected:
those results use 3D volumetric context and the full ~1,000-patient collection, whereas this is
2D single-slice on 50 patients. It is a real supervised segmentation with a real number, not an
attention proxy.

### Cross-domain transfer — measured, not assumed

IQ-OTH has no masks, so Dice cannot be computed there and **we did not invent one**. Instead we
used a proxy the labels *do* support: how often the segmenter fires on images labelled normal
versus malignant.

| IQ-OTH label | Fires on | Rate |
|---|---|---|
| normal | 19/120 | 0.158 |
| benign | 3/30 | 0.100 |
| malignant | 86/163 | 0.528 |

**Malignant:normal firing ratio = 3.333×.**

This is a genuinely interesting contrast with D21. The *classifier* trained on LIDC failed to
transfer to IQ-OTH (significantly worse, p = 0.009). The *segmenter* does carry signal across the
same domain gap. The plausible reason: nodule morphology is a local texture/shape cue, far less
sensitive to global appearance differences (field of view, windowing, JPEG artefacts) than a
whole-image classifier that can latch onto domain-level statistics.

### Honesty controls shipped with it

- Detections outside the detected lung field are **discarded** — a "nodule" in the chest wall is a
  false positive by construction.
- Blobs under 12 px are dropped as noise.
- The UI leads with **what it cannot claim** before showing the mask, and states the
  **16% false-positive rate on normal-labelled images** directly on the panel.
- The phrase "confirmed tumour boundary" remains forbidden un-negated, enforced by a unit test.
- `supervised_segmentation` is now removed from `UNSUPPORTED_CAPABILITIES` — but only because the
  capability genuinely exists; staging remains permanently refused.

### Decision

**Deployed** as an additional, clearly-scoped output alongside Grad-CAM. The two answer different
questions — the U-Net answers "where exactly", Grad-CAM answers "what drove the classification" —
and the dashboard says so explicitly rather than blurring them together.


---

## D23 — What "Dice 0.39" actually means: detection vs delineation

**Question raised:** can the segmentation model find nodules *exactly*?

A single Dice number cannot answer that, because it conflates two very different
questions with very different clinical consequences:

- **Detection** — did it find the nodule at all?
- **Delineation** — once found, is the outline accurate?

`scripts/16_segmentation_error_analysis.py` decomposes it on the held-out LIDC patients
(168 slices, 8 patients).

### Result

| Question | Answer |
|---|---|
| Detection rate | **0.554** (93/168) |
| Missed entirely | **75** |
| Dice **when it detects** | **0.7054** |
| Dice over all slices (misses = 0) | 0.3905 |
| Median centre error when detected | **0.24 px (~0.17 mm)** |
| Predicted/true area ratio | 1.062 |
| False-positive blobs per slice | 0.351 |

**So: when it finds a nodule it locates it almost exactly** — the predicted centre sits
within about 0.2 mm of the radiologist consensus centre and the area is within 6% of
truth. **But it only finds 55% of them.**

### The misses are blindness, not uncertainty

This is the decisive diagnostic:

| | Median peak probability inside the true nodule |
|---|---|
| Detected cases | **1.0** |
| Missed cases | **0.0003** |

97% of missed nodules have a peak response below 0.05. The model is not "unsure" about
them — it produces essentially **no response at all**.

**Consequence: the threshold is not the fix.** Sweeping the operating threshold from 0.05
to 0.55 on validation moved detection by under 3 points (0.574 to 0.549). Lowering it only
adds false positives; it does not recover the misses. That rules out the cheap fix and
points at capacity/representation instead.

### Dice by nodule size

| Size | Dice | n |
|---|---|---|
| small (<40 px) | 0.2534 | 48 |
| medium (40-120 px) | 0.5072 | 82 |
| large (>120 px) | 0.3119 | 38 |

### Why

2D single-slice inference. A nodule that is unambiguous when a radiologist scrolls through the volume can be indistinguishable from a vessel cross-section in one slice. The model also saw only 35 training patients.

### What would actually fix it

- 3D or multi-slice input (2.5D stacking of adjacent slices)
- more annotated patients - the full LIDC collection has ~1000
- a detection-first architecture (e.g. nodule candidate proposal + false-positive reduction) rather than pure semantic segmentation

### What shipped

The UI now reports **detection rate and Dice-when-detected separately**, rather than a
single averaged Dice that flatters the delineation and hides the sensitivity problem. The
empty-result state says explicitly that a clean scan is not a clearance because the model
missed roughly half of annotated nodules. The honest one-line summary is:

> **When it finds a nodule it outlines it accurately. It finds about half of them.**


---

## D24 — Three attempts to raise detection, all rejected

**Question:** D23 showed the segmenter localises to ~0.17 mm when it fires but detects only
55% of nodules. Can detection be improved?

Three fixes were tried. **All three failed, and the deployed model is unchanged.**

### Attempt 1 — lower the operating threshold

Rejected before it started. Sweeping the threshold from 0.05 to 0.55 on validation moved
detection by under 3 points (0.574 to 0.549). The diagnostic explains why: missed nodules have
a **median peak probability of 0.0003** inside the true lesion, with 97% below 0.05. The model
is not hesitating, it is blind. There is nothing under the threshold to recover.

### Attempt 2 — 2.5D input (adjacent-slice stacking)

The most principled fix, and the one D23 predicted would work. In a single axial slice a small
nodule and a vessel seen end-on are nearly identical discs; what separates them is continuity
through the volume, which is exactly what a radiologist uses when scrolling. So slices z-1 / z /
z+1 were stacked as three channels (`scripts/17`, `scripts/18`), with the split, seed,
architecture, loss and schedule held identical so only the input changed.

| Metric | 2D | 2.5D |
|---|---|---|
| Detection rate | 0.5536 | **0.5536** |
| Dice when detected | 0.7054 | 0.6509 |
| Dice over all | 0.3905 | 0.3603 |
| Centre error | 0.17 mm | 0.27 mm |
| FP blobs/slice | 0.351 | 0.524 |

Detection was **identical to four decimal places** — the same 75 misses out of 168. Digging in:
**84% of the misses were the same slices**, with 12 recovered and 12 newly broken, i.e. pure
noise. The extra slice context changed which coin landed heads, not how many.

### Attempt 3 — more annotated slices per nodule

The per-nodule slice cap was raised from 3 to 9, growing the training set from 922 to 1,380
masks (+50%) from the same 50 patients.

| Metric | 922 masks | 1,380 masks |
|---|---|---|
| Dice over all | **0.3924** | 0.3453 |
| Cross-domain firing ratio | **3.33x** | 1.31x |

Worse on both counts, and the cross-domain transfer that made the segmenter useful on IQ-OTH
largely **collapsed**. The cause is straightforward in hindsight: slices further from a nodule's
centre show progressively smaller, more ambiguous cross-sections. Raising the cap filled the
training set with near-unlearnable edge slices and diluted the signal. **More data of the wrong
kind is worse than less data of the right kind.**

This attempt had actually overwritten the deployed checkpoint, and it was **rolled back** from
`models/unet_nodule_v1_922slices.pt`; the extraction was re-run at cap 3 so the data on disk
matches the deployed model again.

### What the evidence now says

- The failure is **not** threshold calibration (attempt 1).
- It is **not** missing slice context (attempt 2).
- It is **not** raw slice count (attempt 3).
- Misses concentrate by patient - one test patient lost 63% of its nodules while another lost 4% -
  and missed nodules are smaller (median 54 px vs 75 px detected).

That pattern points at **patient-level diversity and nodule-scale coverage**, not architecture.
We trained on **35 patients out of the 1,018 available in LIDC-IDRI**. The honest next step is
more patients, not a cleverer model - and notably, more *patients* is a different lever from
more *slices per patient*, which attempt 3 showed actively hurts.

### Decision

**Deployed model unchanged**: the 2D U-Net trained on 922 consensus masks from 35 patients.
Detection 55.4%, Dice-when-detected 0.705, centre error ~0.17 mm. All three failed attempts are
retained in `scripts/17`, `scripts/18` and
`reports/experiments/unet_segmentation_v2_1380slices.json` because knowing which fixes do **not**
work is a result, and it stops the next person spending the same day on them.


---

## D25 — Web-sourced images returned confident wrong answers: an input gate

**Reported:** images downloaded from the web were coming back "malignant" with high
confidence. Two files were supplied (`images.jpg`, 552x554; `OIP.jpg`, 250x189 — a
watermarked Alamy stock thumbnail). Both returned **malignant at 92-99% confidence**.

### Diagnosis 1 — resolution collapse

A controlled sweep on **ground-truth NORMAL held-out images**, downscaled and re-encoded:

| Input size | correct (normal) | drifted to benign | flipped to malignant |
|---|---|---|---|
| 512 px | 9/10 | 1 | 0 |
| 400 px | 8/10 | 2 | 0 |
| 350 px | 4/10 | 6 | 0 |
| 320 px and below | collapses | | |
| **250x189** | **0/12** | 0 | **12/12** |

Heavy JPEG compression alone did **not** cause this (12/12 still correct at quality 40).
The trigger is **upscaling**: training images are 512x512 letterboxed down to 300x300, so
an image arriving smaller has to be interpolated up, and that smoothing reads as pathology.
This fully explains `OIP.jpg`.

### Diagnosis 2 — the classifier has no reject option

Synthetic inputs at a perfectly adequate 512x512:

| Input | Prediction | Confidence |
|---|---|---|
| Pure black | benign | 0.87 |
| Pure white | benign | 0.90 |
| Random noise | benign | 0.82 |
| Smooth gradient | **malignant** | **0.94** |

It is a 3-way softmax. It cannot say "this is not a chest CT", so it never did.

### A check that was built and then removed

A Mahalanobis out-of-distribution check on the CNN embedding was implemented, measured,
and **deleted** because it does not discriminate on this model:

| | distance |
|---|---|
| Real CT (in-domain) | median 19, max 28 |
| Pure black | 20.3 |
| Pure white | 19.2 |
| Random noise | 21.6 |

Complete overlap. It raised false alarms on genuine held-out scans while passing blank
images — worse than no check. The score is retained as a diagnostic field only, never as a
gate. Recorded here so nobody rebuilds it.

### What shipped

A two-part gate in `src/inference/engine.py`, graded `ok` / `degraded` / `unreliable`,
taking the worse of the two:

1. **Resolution** — on the **original upload**, not the letterboxed analysis frame.
   `>= 400 px` short side is the validated range; `256-399` is degraded; `< 256` unreliable.
2. **Plausibility** — lung field present and 3-60% of frame, intensity spread `>= 25`,
   Laplacian variance `>= 80`. Simple, inspectable pixel checks; these *do* separate the
   cases the OOD score could not.

When the gate is not `ok`, the workspace shows a prominent banner above everything else and
the risk card carries a reliability warning. Verified behaviour:

| Input | Gate |
|---|---|
| Real IQ-OTH test scans | **ok** (no false alarms) |
| `images.jpg` (552 px, plausible CT) | ok |
| `OIP.jpg` (189 px) | **unreliable** |
| Pure black / white / gradient | **unreliable** |
| Random noise | degraded |

### The one that still passes

`images.jpg` is 552 px and a plausible chest CT, so it passes the gate and still predicts
malignant. There is **no ground truth** for it. It is not out-of-distribution by embedding
distance either. The honest position: this model was trained on a single centre and D21
already measured that cross-domain transfer **fails**, so any web image is outside its
validated envelope regardless of what the gate says. The existing domain caveat covers it —
but a resolution gate cannot fix a generalisation limit, and should not be presented as if
it does.

### Two bugs in my own code, found by this

1. The resolution check used `>= 128 px` — far too lenient.
2. It checked `measured_parameters.width_px`, which is the **processed 300x300 frame** and
   therefore always passed. It never looked at the upload at all.

**Lesson: a quality check that measures the wrong thing is worse than no quality check,
because it produces a green tick.** Four regression tests now pin the thresholds and assert
the gate does not fire on real scans.


---

## D26 — External validation: the model does not generalise

**The question:** how does the deployed classifier perform on a dataset it has never seen?

**The test set.** LIDC-IDRI: **922 slices from 50 patients**
(622 benign, 300 malignant), from
different hospitals, different scanners (GE / Siemens / Philips / Toshiba), a different
country, and a separate reconstruction pipeline. Verified fair: every row of the deployed
model's training split file has `source=IQ-OTH`. Nothing about the classifier was fitted or
tuned on LIDC.

### Result

| | ROC-AUC on P(malignant) | 95% CI |
|---|---|---|
| **In-domain** (IQ-OTH held-out test, n=96) | **1.0000** | 1.000 – 1.000 |
| **External** (LIDC, n=922) | **0.4883** | 0.449 – 0.528 |
| Change | **-0.5117** | |

**0.50 is chance. The external confidence interval (0.449 – 0.528)
straddles it.** On data from other hospitals, the model has **no measurable ability** to tell
benign from malignant.

Patient-level aggregation does not rescue it: AUC 0.5604
(95% CI 0.391 – 0.721), which also includes chance.

### How the failure presents

| Predicted on LIDC | Count | Share |
|---|---|---|
| malignant | 708 | 76.8% |
| normal | 163 | 17.7% |
| benign | 51 | 5.5% |

It calls **77% of everything
malignant**. That produces a flattering-looking sensitivity of
**0.931** next to a specificity of **0.066** —
the behaviour of a model that has largely stopped discriminating and is answering "malignant"
by default. Quoting that sensitivity alone would be actively dishonest.

Separately, **163 slices (17.7%)
were called "normal" even though every slice in this set contains an annotated nodule.**

### Why the in-domain number is 1.0000 and that is not reassuring

Perfect in-domain separation of benign from malignant, alongside chance-level external
performance, is the signature of a model that has learned **dataset-specific appearance**
rather than lesion morphology. IQ-OTH benign and malignant images differ in ways beyond the
nodule — acquisition, windowing, JPEG pipeline — and those shortcuts do not exist in LIDC.
A perfect internal score should have been read as a warning, not an achievement.

### Consistency with earlier findings

This is the same wall hit three times now, from three directions:
- **D14** — Grad-CAM localisation was at chance on out-of-domain lesions.
- **D21** — training on IQ-OTH + LIDC made IQ-OTH performance *significantly worse* (p = 0.009).
- **D26** — the IQ-OTH-trained classifier is at chance on LIDC.

Notably the **segmenter does transfer** (D22: 3.33x higher firing on malignant than normal
across the same domain gap). Local nodule morphology crosses domains; whole-image
classification does not.

### What this changes in the product

1. Every performance figure is now labelled **"in-domain (IQ-OTH held-out)"**. The number was
   always measured on IQ-OTH; it was not previously labelled as *only* meaning that.
2. External validation is surfaced in the product with the AUC of 0.4883 stated
   plainly, next to the in-domain figure.
3. The honest capability statement is now: **this model is validated on IQ-OTH/NCCD and
   demonstrably does not generalise beyond it.** It is a working research pipeline, not a
   deployable screening tool.

### What would be needed to fix it

Multi-centre training data (LIDC alone is insufficient — D21 showed naive merging hurts),
domain adaptation or harmonisation, and prospective external validation on each target site
before any clinical use. None of that fits the remaining scope; saying so is the correct
outcome.

### Caveats on this test itself

- LIDC contains no normal class; this is benign-vs-malignant discrimination, not screening performance.
- LIDC labels are averaged radiologist impression (1-5), not pathology.
- Nodules with a mean malignancy of exactly 3 were excluded as indeterminate.
- Slices are lesion-centred by construction, so prevalence here is not clinical prevalence.

These caveats limit how *precisely* the AUC should be read. They do not rescue it: a
confidence interval straddling 0.50 means no discrimination was demonstrated.


---

## D27 — Fixing the domain-shift failure: the model was never looking at the nodule

D26 measured that the whole-image classifier is at chance on unseen hospitals
(external AUC 0.488, in-domain 1.000). This entry is the investigation and the fix.

### Step 1 — Where does the dataset signature live? (`scripts/22`)

A linear probe on simple appearance descriptors, evaluated with grouped CV so it cannot
memorise patients:

| View | Domain ROC-AUC |
|---|---|
| Whole slice | **1.000** |
| Lung-field masked | 0.939 |
| Random lung-interior crop | **0.937** |

**Cropping alone does not remove the signature.** It is in the lung parenchyma itself —
reconstruction kernel, noise texture, compression. This ruled out "just use ROIs" as a
complete answer before any pipeline was built.

### Step 2 — Can preprocessing remove it? (`scripts/23`)

Eight transforms were measured on two axes at once: domain separability (lower is better)
and whether benign/malignant signal survives (higher is better).

| Transform | Domain AUC | Signal AUC |
|---|---|---|
| identity (then-current) | 1.000 | 0.559 |
| **lung + quantile histogram match** | **0.857** | **0.679** |
| band-pass + z-score | 0.909 | 0.687 |
| gradient only | 0.881 | 0.525 |

Best available was 0.857 — a long way from 0.5. Notably the
*signal* rose (0.559 → 0.679),
because removing the appearance confound leaves structure behind. Implemented as
`preprocess_harmonised()`.

### Step 3 — Do harmonisation and domain-adversarial training help? (`scripts/24`)

Four arms, identical architecture, seed and schedule; one switch differs each time. DANN
uses target IMAGES for the discriminator but never target labels (unsupervised adaptation).

| Arm | In-domain AUC | External AUC | 95% CI |
|---|---|---|---|
| baseline | 1.0000 | **0.4605** | 0.422–0.500 |
| harmonised | 0.9890 | 0.3631 | 0.323–0.401 |
| DANN | 0.9973 | 0.4060 | 0.368–0.447 |
| harmonised + DANN | 0.9817 | 0.3530 | 0.313–0.392 |

**Both interventions made it worse.** Reported as measured. The DANN implementation is
retained and correct (gradient reversal, standard lambda ramp) — it simply does not rescue
a model whose features were wrong to begin with.

### Step 4 — The diagnosis (`scripts/25`)

This is the finding everything else turned on.

| | Value |
|---|---|
| Nodule **size alone** predicting LIDC malignancy | **AUC 0.873** |
| Trained model's AUC on the same data | 0.457 |
| Spearman(model score, nodule area) | **+0.050** (p = 0.13) |

A single scalar — how big the nodule is, which is genuine radiology — achieves
0.873. The trained network's score is
**statistically uncorrelated with it**. The model was never looking at the nodule. It was
classifying the picture.

That reframed the problem: the task is not impossible, the input was wrong.

### Step 5 — The fix (`scripts/26`, `scripts/27`)

Crop to the annotated nodule and classify that instead. Patient-level split on real LIDC
patient IDs (35/7/8 patients).

| Model | AUC on held-out patients | 95% CI |
|---|---|---|
| **ROI features + SVM-RBF** | **0.9145** | 0.864–0.956 |
| ROI features + RandomForest | 0.9083 | 0.861–0.949 |
| ROI features + LogisticRegression | 0.9059 | 0.843–0.954 |
| ROI features + Quantum (8 qubits) | 0.8756 | 0.819–0.927 |
| ROI CNN (EfficientNet-B3) | 0.6797 | 0.583–0.765 |
| *Whole-image model, same data* | *0.4566* | |

**0.46 → 0.91.** Attending to the lesion rather than the image is the whole difference.

Two secondary findings, both worth stating:

* **A 10-feature interpretable model beats EfficientNet-B3** (0.914 vs
  0.680). With 35 training patients the CNN overfits. Reaching
  for a deep model here is the wrong instinct.
* **The quantum branch is significantly WORSE.** Paired bootstrap on held-out patients:
  quantum vs best classical Δ = -0.0385, p = 0.010;
  hybrid vs best classical Δ = -0.0159, p = 0.047. Both significant.
  The circuit does work (0.876 is a respectable AUC) — it is simply
  beaten. Reported as measured.

### The honest limit on this result

The 0.914 is **cross-patient within LIDC**, not cross-dataset. The 0.488 is **cross-dataset**.
They are not the same evaluation and the tables label them separately. Cross-dataset ROI
validation is impossible today because IQ-OTH carries no nodule annotations — the ROI cannot
be constructed for it. Closing that gap needs either IQ-OTH annotations or a detector good
enough to supply them, and the current U-Net detects only 55% (D23).

So the correct claim is: **nodule-focused features generalise across patients and scanners
within an annotated dataset, and the whole-image approach does not generalise at all.** The
cross-dataset ROI claim is not yet earned.

### What changed in the codebase

| Added | Purpose |
|---|---|
| `preprocess_harmonised()` | rank-based intensity harmonisation inside the lung field |
| `src/models/dann.py` | gradient reversal, lambda ramp, domain head — switchable |
| `scripts/22`–`23` | domain-signature and harmonisation probes |
| `scripts/24` | four-arm domain-robustness comparison |
| `scripts/25` | the inversion / size-confound diagnosis |
| `scripts/26`–`27` | ROI pipeline, baselines, hybrid quantum |
| `scripts/28` | assembles Tables 1–3 from artefacts only |

Nothing in the deployed serving path was changed. The clinical platform still runs the
IQ-OTH model, now labelled with its measured readiness level (D28).


---

## D28 — Fixing the ROI CNN's overfitting

**The problem.** EfficientNet-B3 carries 10.7M parameters. The ROI training set is 632 crops
from 35 patients — roughly **17,000 parameters per training image**. It peaked on validation
at epoch 3 and memorised thereafter.

### A methodology fix came first

The earlier comparison used a single **8-patient** test split. At that size a 0.05 AUC
difference is noise, so it could not rank strategies. Everything below uses
**StratifiedGroupKFold over all 50 patients** — every patient tested exactly once, ~922
out-of-fold predictions instead of 168.

That change alone moved the numbers: the CNN went 0.680 → 0.726 and the
hand-crafted model 0.898 → 0.862. **The single-split estimates were
unreliable in both directions**, which is worth knowing independently of the overfitting
question.

### Six strategies, measured

| Strategy | AUC | 95% CI | Patient AUC |
|---|---|---|---|
| **10 hand-crafted features (SVM)** | 0.8617 | 0.836–0.887 | 0.8631 |
| MobileNetV3-Small (1.5M params) | 0.7853 | 0.752–0.814 | 0.8003 |
| Frozen CNN embedding + hand-crafted | 0.7395 | 0.706–0.772 | 0.8631 |
| EfficientNet-B3 full fine-tune (baseline) | 0.7263 | 0.692–0.760 | 0.6997 |
| Frozen encoder + linear probe | 0.6112 | 0.573–0.651 | 0.7697 |
| Small backbone + heavy aug + dropout 0.4 + wd 5e-2 | 0.5548 | 0.516–0.596 | 0.4928 |

### What actually worked

**1. Cutting capacity.** MobileNetV3-Small (1.5M params) beats EfficientNet-B3 (10.7M) by
**+0.0590** with barely overlapping intervals. Seven times fewer parameters,
better generalisation. This is the single most effective change to the network itself.

**2. Not using a CNN at all.** Ten interpretable features — area, equivalent diameter,
perimeter, circularity, solidity, intensity statistics, border contrast — reach
**0.8617**, beating the best deep variant by
+0.0764 with non-overlapping confidence intervals. At 50 patients this is
simply the right model class. It is also inspectable, which matters clinically.

### What did NOT work, and is worth recording

**Heavy regularisation made it much worse** (0.5548, the worst arm). Dropout 0.4 plus
weight decay 5e-2 plus aggressive augmentation and cutout over-constrained a model that was
already small. The instinct "overfitting → regularise harder" is wrong here: the lesion
occupies a small part of a 128px crop, and cutout/scale-jitter frequently destroyed it.

**Frozen ImageNet features underperformed** (0.6112). Generic natural-image
features do not transfer well to small greyscale nodule crops without adaptation.

**Fusion was mixed.** Concatenating the frozen embedding with the hand-crafted features gave
0.7395 at slice level — worse than features alone, because 1,536 embedding dimensions
swamp 10 informative ones — but it tied for the **best patient-level AUC**
(0.8631). With proper feature selection this is the most promising direction.

### Decision

For this data scale the deployed ROI classifier should be the **hand-crafted feature model**,
not a CNN. The CNN path is retained and improved (small backbone), because it becomes the
right choice once training data grows to hundreds of patients — but claiming a deep model is
better here would contradict the measurement.

### The general lesson

Three of the four "obvious" fixes for overfitting — heavier regularisation, frozen pretrained
features, feature fusion — **made things worse or no better**. The two that worked were
reducing capacity and abandoning the deep model for the data scale. With 50 patients, model
selection matters far more than regularisation tuning.


---

## D29 — Is there still overfitting? Measured, with a model change as a result

Overfitting is a GAP, not a yes/no property. Measured under the same grouped 5-fold protocol:

| Model | Train AUC | Held-out AUC | Gap | Verdict |
|---|---|---|---|---|
| SVM-RBF (C=3) | 0.9486 | 0.8617 | **+0.0868** | mild overfitting |
| **Logistic regression** | 0.9000 | **0.8739** | **+0.0261** | **well generalised** |
| SVM-RBF, C=100 (deliberate control) | 0.9809 | 0.8136 | +0.1673 | moderate overfitting |

### Model change

**Logistic regression replaces SVM-RBF as the ROI candidate.** It is better on both axes:
a smaller train/held-out gap (+0.026 vs +0.087) AND a higher held-out AUC
(0.8739 vs 0.8617). Script 29 ranked SVM first on out-of-fold AUC alone;
adding the gap measurement reversed the choice. The C=100 arm is a deliberate positive
control confirming the measurement detects overfitting when it is present.

### Learning curve — would more data help?

| Patients | Held-out AUC |
|---|---|
| 15 | 0.8244 |
| 25 | 0.7923 |
| 35 | 0.8217 |
| 42 | 0.8429 |
| 50 | 0.8742 |

**Still rising.** The curve is still rising: more annotated patients would likely improve performance further. More annotated patients is the highest-value
next investment, ahead of further architecture work.

### The precise answer to "does it work now?"

Three separate questions, three different answers:

1. **Is the ROI model overfitting?** Essentially no. Gap +0.026 with logistic
   regression — a well-generalised model across unseen *patients* within LIDC.

2. **Is the cross-DATASET problem solved?** **No, and it has not been tested.** IQ-OTH has no
   nodule annotations, so ROIs cannot be built for it. The ROI result is cross-patient, not
   cross-dataset. These are different claims and must not be conflated.

3. **Does the deployed platform work?** **No. It is unchanged.** The clinical application still
   serves the whole-image EfficientNet-B3 -> Quantum_4q model, which remains at chance
   externally (AUC 0.488, D26). Readiness stays **Level 2 of 5**. The ROI work is a research
   result that has not been promoted into the serving path, because promoting it would imply
   a cross-dataset claim that has not been earned.
