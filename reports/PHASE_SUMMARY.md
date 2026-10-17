# Platform Transformation — Phase Summary

**QuantumLung** · SIH 2026 · PS 26139 · Team CodeNova

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Project audit (`PROJECT_AUDIT.md`) | Complete |
| 2 | Design system — tokens, typography, spacing, 20+ primitives | Complete |
| 3 | Application shell — collapsible sidebar, top bar, routing | Complete |
| 4 | Overview dashboard | Complete |
| 5 | CT analysis workspace (flagship) | Complete |
| 6 | ML integration — existing models, unchanged | Complete |
| 7 | Reporting workflow + clinician feedback loop | Complete |
| 8 | AI priority queue | Complete |
| 9 | Patient timeline + scan comparison | Complete |
| 10 | Follow-up management | Complete |
| 11 | Model analytics (real results only) | Complete |
| 12 | Quantum technology visualisation | Complete |
| 13 | Testing — 38 Python tests, 0 TS errors, 0 lint errors | Complete |
| 14 | UI/UX review passes | Complete |

## Not built, and why

| Requested | Status | Reason |
|---|---|---|
| Multilingual patient view (EN/HI/ML/TA) | **Not built** | Requires professionally translated medical content. Machine-translating clinical text is unsafe, and shipping an untranslated language switcher would be worse than omitting it. |
| Educational AI assistant | **Not built** | The content exists at `/api/education`; the conversational surface was not built. |

Both are additive content features that need no model or schema change.

## Verification

- **38/38** Python tests pass — the ML core was not modified
- **0** TypeScript errors, **0** ESLint errors, **0** warnings
- **0** browser console errors across all 11 routes
- No horizontal overflow at 1600px, 1280px or 390px
- Production build succeeds; 12 routes
- Analysis → report → review workflow verified end-to-end

## Scientific integrity preserved

All four measured negative results remain visible in the product:
leakage quantified and removed · no quantum advantage claimed (p ≈ 0.69–0.71) ·
Grad-CAM localisation at chance · external data degraded performance and was not deployed.


---

## Domain-shift redesign (added after external validation)

| Phase | Deliverable | Status |
|---|---|---|
| 1 | Focused audit of the domain-shift question | Complete |
| 2 | Label harmonisation - datasets kept SEPARATE, shared binary task only | Complete |
| 3 | `preprocess_harmonised()` - rank-based intensity harmonisation | Complete |
| 4 | Nodule-centred ROI pipeline | Complete |
| 5 | DANN with gradient reversal, switchable | Complete |
| 6 | Modular backbone comparison | Complete |
| 7 | Classical vs quantum vs hybrid on ROI features | Complete |
| 9 | Tables 1-3, bootstrap CIs, patient-level metrics | Complete |
| 11 | Domain Shift analysis page | Complete |
| 13 | Model readiness levels (`src/evaluation/readiness.py`) | Complete |
| 14 | Experiment artefacts in `reports/experiments/` | Complete |

### Headline

| | AUC |
|---|---|
| Whole-image, unseen hospitals (cross-dataset) | **0.488** |
| Nodule-ROI features, unseen patients (cross-patient) | **0.914** |

### Interventions that did NOT work, reported as measured

- Harmonisation alone: external AUC fell to 0.363
- DANN alone: 0.406
- Harmonisation + DANN: 0.353
- ROI CNN: 0.680, beaten by a 10-feature linear model
- Quantum branch: significantly worse than classical (p = 0.010)

### Readiness

**Level 2/5 - Internally validated.** Blocked at level 3 by external validation.

### Tests

50 passing, including guards that the readiness system cannot award level 3 while the
external CI includes chance, and that a significantly negative quantum result can never be
described as an advantage.
