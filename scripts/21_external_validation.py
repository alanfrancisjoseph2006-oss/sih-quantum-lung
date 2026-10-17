"""
Phase 16 - EXTERNAL VALIDATION on a dataset the deployed classifier has never seen.

WHY THIS IS A FAIR TEST
-----------------------
The deployed model (EfficientNet-B3 -> Quantum_4q) was trained exclusively on the
IQ-OTH/NCCD split file, which contains only IQ-OTH images (verified: every row's
`source` is IQ-OTH). LIDC-IDRI comes from different hospitals, different scanners
(GE / Siemens / Philips / Toshiba), a different country, and was reconstructed and
windowed by a completely separate pipeline. Nothing about the classifier was
fitted or tuned on it.

WHAT LIDC CAN AND CANNOT TELL US
--------------------------------
* It has NO "normal" class. Every slice here contains an annotated nodule, because
  we only extracted slices at annotated lesion positions. So this measures
  benign-vs-malignant discrimination, NOT screening performance.
* Its labels are radiologist IMPRESSION on a 1-5 scale averaged over up to four
  readers, not pathology. Nodules with a mean of exactly 3 were excluded.
* Therefore the headline number is ROC-AUC on P(malignant), which is
  threshold-free. Accuracy alone across domains is confounded by calibration
  shift, and reporting it as the headline would be misleading.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from src.config import CLASSES, DATA, EXPERIMENTS, REPORTS, SEED, SPLITS, set_seed
from src.inference.engine import get_engine
from src.preprocessing.medical_image import load_image, preprocess, to_model_tensor

LIDC = DATA / "external" / "lidc"
MAL = CLASSES.index("malignant")
NOR = CLASSES.index("normal")


def probs_for(engine, path: str) -> np.ndarray:
    """Raw posterior from the deployed branch, before the decision policy."""
    li = load_image(path)
    x = to_model_tensor(preprocess(li.array)).to(engine.device)
    with torch.no_grad():
        _, emb = engine.cnn(x, return_embedding=True)
    branch, _z, _q = engine._branch_probabilities(emb.float().cpu().numpy())
    return branch[f"Quantum_{engine.n_qubits}q"]


def boot_auc(y: np.ndarray, s: np.ndarray, n: int = 2000, seed: int = SEED):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], s[i]))
    v = np.array(vals)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> int:
    set_seed(SEED)
    e = get_engine()
    print(f"deployed model: {e.backbone} -> {e.primary_name} ({e.n_qubits} qubits)\n")

    rows = list(csv.DictReader(open(LIDC / "lidc_masks.csv", newline="")))
    print(f"external set: {len(rows)} LIDC slices, "
          f"{len({r['patient_id'] for r in rows})} patients "
          f"({dict(Counter(r['label'] for r in rows))})")

    P, y_true, pids = [], [], []
    for i, r in enumerate(rows, 1):
        try:
            P.append(probs_for(e, r["path"]))
            y_true.append(1 if r["label"] == "malignant" else 0)
            pids.append(r["patient_id"])
        except Exception as ex:
            print(f"  skip {r['file']}: {ex}")
        if i % 200 == 0:
            print(f"  ...{i}/{len(rows)}", flush=True)

    P = np.array(P)
    y = np.array(y_true)
    pids = np.array(pids)
    pred3 = P.argmax(1)

    print("\n=== WHAT THE MODEL PREDICTED (3-class) ===")
    dist = Counter(CLASSES[k] for k in pred3)
    for c in CLASSES:
        print(f"  {c:10s} {dist.get(c, 0):4d}  ({100 * dist.get(c, 0) / len(pred3):5.1f}%)")
    called_normal = int((pred3 == NOR).sum())
    print(f"\n  Every slice here contains an annotated nodule, yet {called_normal} "
          f"({100 * called_normal / len(pred3):.1f}%) were called 'normal'.")

    # ---- threshold-free: does it RANK malignant above benign? ----
    s_mal = P[:, MAL]
    auc = roc_auc_score(y, s_mal)
    lo, hi = boot_auc(y, s_mal)
    ap = average_precision_score(y, s_mal)
    print("\n=== BENIGN vs MALIGNANT DISCRIMINATION (slice level) ===")
    print(f"  ROC-AUC on P(malignant) : {auc:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  Average precision       : {ap:.4f}   (prevalence {y.mean():.3f})")
    print("  -> 0.50 is chance. Threshold-free, so cross-domain calibration")
    print("     shift does not distort it.")

    # ---- as the product actually behaves ----
    lesion_pred = np.where(pred3 == NOR, -1, (pred3 == MAL).astype(int))
    scored = lesion_pred >= 0
    tn = fp = fn = tp = 0
    if scored.sum():
        cm = confusion_matrix(y[scored], lesion_pred[scored], labels=[0, 1])
        tn, fp, fn, tp = (int(v) for v in cm.ravel())
        print("\n=== AS THE PRODUCT BEHAVES (argmax, 'normal' abstained) ===")
        print(f"  scored {int(scored.sum())}/{len(y)} slices "
              f"({len(y) - int(scored.sum())} abstained as 'normal')")
        print(f"  malignant sensitivity : {tp / max(tp + fn, 1):.4f}")
        print(f"  benign specificity    : {tn / max(tn + fp, 1):.4f}")
        print(f"  accuracy              : {(tp + tn) / max(int(scored.sum()), 1):.4f}")
        print(f"  confusion [[TN {tn} FP {fp}] [FN {fn} TP {tp}]]")

    # ---- patient level ----
    up = sorted(set(pids))
    py = np.array([int(y[pids == p].max()) for p in up])
    ps = np.array([float(s_mal[pids == p].max()) for p in up])
    pauc = plo = phi = None
    if len(np.unique(py)) > 1:
        pauc = roc_auc_score(py, ps)
        plo, phi = boot_auc(py, ps)
        print(f"\n=== PATIENT LEVEL (max P(malignant) per patient, n={len(up)}) ===")
        print(f"  ROC-AUC : {pauc:.4f}   95% CI [{plo:.4f}, {phi:.4f}]")

    # ---- in-domain reference, same task ----
    iq = [r for r in csv.DictReader(open(SPLITS / "splits.csv", newline=""))
          if r["split"] == "test" and r["label"] in ("benign", "malignant")]
    Pi = np.array([probs_for(e, r["path"]) for r in iq])
    yi = np.array([1 if r["label"] == "malignant" else 0 for r in iq])
    iauc = roc_auc_score(yi, Pi[:, MAL])
    ilo, ihi = boot_auc(yi, Pi[:, MAL])
    print("\n=== IN-DOMAIN REFERENCE (IQ-OTH held-out test, same task) ===")
    print(f"  n={len(yi)}  ROC-AUC : {iauc:.4f}   95% CI [{ilo:.4f}, {ihi:.4f}]")
    print(f"\n  CHANGE on external data: {iauc:.4f} -> {auc:.4f}  ({auc - iauc:+.4f})")

    out = {
        "external_dataset": "LIDC-IDRI (TCIA)",
        "why_fair": ("The deployed classifier was trained only on IQ-OTH/NCCD; every row "
                     "of its training split file has source=IQ-OTH. LIDC is a different "
                     "country, hospitals, scanners and reconstruction pipeline."),
        "n_slices": int(len(y)),
        "n_patients": int(len(up)),
        "label_counts": {"benign": int((y == 0).sum()), "malignant": int((y == 1).sum())},
        "prediction_distribution": {c: int(dist.get(c, 0)) for c in CLASSES},
        "lesion_slices_called_normal": called_normal,
        "slice_level": {
            "roc_auc": round(float(auc), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "average_precision": round(float(ap), 4),
            "sensitivity_argmax": round(tp / max(tp + fn, 1), 4),
            "specificity_argmax": round(tn / max(tn + fp, 1), 4),
        },
        "patient_level": ({"roc_auc": round(float(pauc), 4),
                           "ci95": [round(plo, 4), round(phi, 4)]}
                          if pauc is not None else None),
        "in_domain_reference": {"n": int(len(yi)), "roc_auc": round(float(iauc), 4),
                                "ci95": [round(ilo, 4), round(ihi, 4)]},
        "auc_change_external_minus_internal": round(float(auc - iauc), 4),
        "caveats": [
            "LIDC contains no normal class; this is benign-vs-malignant discrimination, "
            "not screening performance.",
            "LIDC labels are averaged radiologist impression (1-5), not pathology.",
            "Nodules with a mean malignancy of exactly 3 were excluded as indeterminate.",
            "Slices are lesion-centred by construction, so prevalence here is not "
            "clinical prevalence.",
        ],
    }
    json.dump(out, open(REPORTS / "external_validation.json", "w"), indent=1)
    json.dump(out, open(EXPERIMENTS / "external_validation.json", "w"), indent=1)
    print("\nwrote reports/external_validation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
