# Project Audit — Phase 1

**Hybrid Quantum-Classical Lung Cancer Detection → Clinical Decision Support Platform**
SIH 2026 · PS 26139 · Team CodeNova
Audit date: 2026-09-03 · **No code was modified during this audit.**

---

## 1. Executive summary

The project is **scientifically strong and architecturally sound underneath, but presented as a
demo rather than a product.** The ML pipeline, honesty controls and experiment tracking are of a
standard well above typical hackathon work. The gap is entirely in product surface: there is one
static HTML page, no patient/scan concept, no workflow, no persistence, and no roles.

**The correct move is a frontend and product-layer build on top of an unchanged ML core.**
Nothing in `src/` needs rewriting. The inference engine already returns a rich, well-structured
payload that a professional UI can render directly.

| Area | State | Action |
|---|---|---|
| ML pipeline (`src/`) | **Strong** — 2,266 LOC, modular, tested | **Preserve unchanged** |
| Trained models | **Strong** — 24 evaluated candidates, honest selection | **Preserve unchanged** |
| Experiment tracking | **Strong** — 24 documented decisions, 4 negative results | Surface in UI |
| Backend API | **Adequate** — 10 endpoints, single file, stateless | Restructure into layers, extend |
| Frontend | **Weak for the target** — 1 static HTML file, vanilla JS | **Rebuild** (Next.js + TS) |
| Database | **Absent** | Build |
| Auth / RBAC | **Absent** | Build architecture |
| Workflow (triage, review, follow-up) | **Absent** | Build |

---

## 2. Current architecture

```
data/            raw (read-only OneDrive), splits, quarantine, external/lidc (3 GB)
src/             2,266 LOC — config, preprocessing, data, models, quantum, fusion,
                 explainability, evaluation, inference
api/main.py      223 LOC — FastAPI, 10 endpoints, serves the static frontend
frontend/        1,286 LOC — index.html + app.js + styles.css (vanilla, no build step)
scripts/         3,660 LOC — 18 numbered pipeline stages + run_all + serve
tests/           429 LOC — 38 tests, all passing
reports/         16 artefacts incl. decisions.md (24 entries), model_comparison.json
models/          35 checkpoints (~470 MB incl. backups)
```

### Hardware
NVIDIA RTX 4050 Laptop, 6.44 GB VRAM, CUDA 12.4, bfloat16 supported.
Inference measured at **0.17–0.87 s per image end-to-end** (CNN + 4 classical heads + quantum
circuit + Grad-CAM + U-Net segmentation).

### Frontend tooling
**Node v24.18.0, npm 11.16.0 present.** Next.js + TypeScript + Tailwind is viable with no
additional system setup.

---

## 3. Existing features (what actually works)

| Feature | Status | Evidence |
|---|---|---|
| 3-class classification (normal / benign / malignant) | Working | test macro-F1 **0.9011**, AUC **0.980**, malignant sensitivity **0.988** |
| Classical ML heads (RF, SVM, XGBoost, LogReg) | Working | evaluated on shared CNN embedding |
| Quantum branch (PennyLane VQC) | Working | 4/6/8-qubit sweep, 4 qubits deployed, gradients verified |
| Hybrid fusion (4 strategies) | Working | meta-classifier best on test (F1 0.925) |
| Adaptive router | Working | significance-aware, validation-only selection |
| Grad-CAM++ explainability | Working | but **measured no better than chance** at localisation |
| **U-Net nodule segmentation** | Working | Dice-when-detected **0.705**, centre error **0.17 mm**, detection **55.4%** |
| DICOM ingest | Working | rescale slope/intercept, lung windowing, PHI never surfaced |
| Cost-sensitive decision policy | Working | benign recall 0.733 → 0.933 |
| Image quality / lung-field detection | Working | `measured_parameters()` + `lung_mask()` |
| Model comparison + significance testing | Working | 24 candidates, paired bootstrap |

### Deliberately unsupported (must stay unsupported)
- **Cancer staging** — no stage labels exist in any dataset. Permanently refused.
- **Histological subtype** — no labels.
- Both are already returned as explicit "not available" blocks with explanations.

---

## 4. Current backend

`api/main.py`, 223 LOC, single file. Endpoints:

```
GET  /api/health · /api/meta · /api/model-comparison · /api/dataset-report
GET  /api/localization-validation · /api/lidc-experiment · /api/education · /api/experiments
POST /api/analyze
GET  /                      (serves static frontend)
```

**Strengths:** clean error taxonomy (415 unsupported type, 422 undecodable, 413 too large,
503 models unavailable, 507 OOM), lazy singleton engine, NaN sanitisation, PHI suppression.

**Limitations for the target product:**
- **Stateless** — every analysis is discarded; nothing is stored or retrievable by ID
- **No layering** — API, business logic and ML access are interleaved in one file
- **No entities** — no patient, scan, report, user or organisation concept
- **No auth** — every endpoint is open
- **No pagination / filtering** — cannot back a queue or a patient list

---

## 5. Current frontend

Single-page vanilla JS app: `index.html` + `app.js` (~700 LOC) + `styles.css`.

**Honest strengths:** no build step, zero console errors, hand-built inline SVG charts, all data
sourced from real API responses, strong scientific-honesty copy throughout.

**Why it cannot become the target product:**
- Dark "tech demo" aesthetic, not clinical software
- **Emoji used as iconography** (🫁, ⚕️) — explicitly disallowed for the professional interface
- No component system — markup is assembled with template literals
- No design tokens — colours and spacing are ad-hoc CSS variables
- No routing, no state management, no data fetching layer
- No loading skeletons, no empty states, no error boundaries
- No accessibility work: no focus management, no ARIA, no keyboard navigation
- Single fixed layout — no application shell, sidebar or workspace concept

**Verdict: rebuild rather than refactor.** The existing page remains valuable as a reference for
the scientific copy, which is genuinely good and should be carried across verbatim where possible.

---

## 6. Technical debt

| # | Issue | Severity | Fix |
|---|---|---|---|
| 1 | No persistence — analyses vanish after the response | **High** | SQLite via SQLAlchemy; entities per spec |
| 2 | Monolithic `api/main.py` | **High** | Split into `api/routers`, `services`, `db`, `schemas` |
| 3 | No auth / RBAC | **High** | JWT + role enum; architecture now, enforcement configurable |
| 4 | `models/` holds 35 files incl. 8 redundant backups (~470 MB) | Medium | Keep; document which is deployed |
| 5 | Frontend has no tests | Medium | Component-level checks + Playwright smoke |
| 6 | `notebooks/`, `src/fusion/`, `data/processed/` are empty | Low | Remove or populate |
| 7 | Grad-CAM localisation measured at chance | **Known, documented** | Keep the caveat prominent in the new UI |
| 8 | U-Net detects only 55% of nodules | **Known, documented** | Surface detection rate beside Dice everywhere |

---

## 7. What the data does and does not support

This constrains the product, and the UI must respect it.

| Requested feature | Supported? | Plan |
|---|---|---|
| Scan analysis, risk, confidence | **Yes** | Full support |
| Grad-CAM attention | **Yes** (with measured caveat) | Show, never call it segmentation |
| Nodule segmentation | **Yes** (LIDC-trained U-Net) | Show with detection rate + Dice |
| Priority queue / triage | **Yes** — derivable from risk + confidence | Build |
| Preliminary report + doctor review | **Yes** — workflow, not new ML | Build |
| Doctor feedback capture | **Yes** — storage only, **no auto-retraining** | Build |
| Patient timeline | **Yes** — needs persistence | Build, demo data clearly labelled |
| Scan comparison | **Partial** — can compare model outputs and image measurements | Build, worded as "observed imaging/model differences", never "progression" |
| Follow-up management | **Yes** — clinician-managed dates only | Build, **no autonomous intervals** |
| **Cancer staging** | **NO — no labels** | Permanently disabled, explained in UI |
| **Subtype** | **NO — no labels** | Permanently disabled |
| Multilingual patient view | **Yes** — content/i18n, not ML | Build EN/HI/ML/TA scaffold |
| Educational assistant | **Yes** — retrieval over existing `CANCER_EDUCATION` | Build rule-based, **no diagnosis** |
| Model analytics | **Yes** — real data in `model_comparison.json` | Build from real results only |
| Quantum technology page | **Yes** — real circuit specs | Build, no advantage claims |

---

## 8. Recommended upgrade plan

**Guiding constraint: `src/` is not modified.** The ML core is the asset; everything else wraps it.

| Phase | Work | Risk |
|---|---|---|
| 2 | Design system + tokens (typography, spacing, colour, components) | Low |
| 3 | Next.js app shell — sidebar, topbar, routing, TS types | Low |
| 4 | Backend restructure: DB layer, entities, services, routers | **Medium** — must not break `/api/analyze` |
| 5 | Overview dashboard | Low |
| 6 | CT analysis workspace (flagship: viewer / AI vis / findings) | **Medium** |
| 7 | Reporting workflow + doctor review | Low |
| 8 | Priority queue | Low |
| 9 | Patient timeline + scan comparison | Medium |
| 10 | Follow-up management | Low |
| 11 | Model analytics (real metrics only) | Low |
| 12 | Quantum technology page | Low |
| 13 | Landing page | Low |
| 14 | Testing + accessibility + UI review passes | Medium |

### Non-negotiables carried forward
1. **Never claim** confirmed diagnosis, autonomous decisions, prescriptions, or radiologist replacement.
2. **Staging and subtype stay disabled** with visible explanations.
3. **Grad-CAM is never labelled segmentation**; the U-Net mask is never called a confirmed boundary.
4. **Detection rate is always shown next to Dice** — a single Dice number misleads.
5. **Doctor feedback is stored, never auto-retrained into production.**
6. **All demo patient data is synthetic and labelled "Demo Data".**
7. **No HIPAA or medical-device claims** — only "designed with healthcare privacy and security considerations".
8. The four negative results (leakage, no quantum advantage, Grad-CAM at chance, LIDC transfer
   failure) remain visible in the product. They are the project's credibility.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Backend restructure breaks working inference | Keep `/api/analyze` contract identical; run the 38 existing tests after every change |
| Next.js rebuild loses the scientific honesty copy | Port the copy verbatim from `app.js` into typed content modules |
| Product framing dilutes the research honesty | Keep negative results on the landing page and model analytics, not buried |
| Scope is large | Phase order above is dependency-ordered; each phase leaves the app runnable |

---

*Phase 1 complete. No files were modified. Proceeding to Phase 2.*
