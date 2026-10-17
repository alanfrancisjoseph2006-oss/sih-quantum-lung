"""
Is there still overfitting? Measure the train-vs-held-out gap directly.

Overfitting is not a yes/no property, it is a GAP: how much better a model scores
on data it was fitted to than on data it was not. This measures that gap for
every strategy from scripts/29 under the same grouped 5-fold protocol, plus a
learning curve to show whether more patients would help.

A model can also UNDERFIT - script 29's heavily-regularised arm scored 0.555,
which is a capacity problem, not an overfitting one. Both are reported.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import DATA, REPORTS, SEED, set_seed

LIDC = DATA / "external" / "lidc"


def load_features():
    rows = list(csv.DictReader(open(LIDC / "lidc_masks.csv", newline="")))
    X, y, g = [], [], []
    for r in rows:
        img = cv2.imread(r["path"], cv2.IMREAD_GRAYSCALE)
        m = (cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        if m.sum() < 4:
            continue
        area = float(m.sum())
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        per = float(cv2.arcLength(c, True))
        hull = float(cv2.contourArea(cv2.convexHull(c))) or 1.0
        v = img[m > 0].astype(np.float32)
        ring = cv2.dilate(m, np.ones((9, 9), np.uint8)) - m
        vs = img[ring > 0].astype(np.float32)
        X.append([area, 2 * np.sqrt(area / np.pi), float(v.mean()), float(v.std()),
                  float(np.percentile(v, 90)), float(np.percentile(v, 10)), per,
                  float(4 * np.pi * area / max(per ** 2, 1e-6)), float(area / hull),
                  float(v.mean() - vs.mean()) if vs.size else 0.0])
        y.append(1 if r["label"] == "malignant" else 0)
        g.append(r["patient_id"])
    return np.array(X, np.float32), np.array(y), np.array(g)


def gap_for(make_model, X, y, g, folds=5):
    """Return (train AUC, held-out AUC) averaged over grouped folds."""
    tr_aucs, oof = [], np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(folds, shuffle=True, random_state=SEED).split(X, y, groups=g):
        sc = StandardScaler().fit(X[tr])
        m = make_model().fit(sc.transform(X[tr]), y[tr])
        tr_aucs.append(roc_auc_score(y[tr], m.predict_proba(sc.transform(X[tr]))[:, 1]))
        oof[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return float(np.mean(tr_aucs)), float(roc_auc_score(y, oof))


def main() -> int:
    set_seed(SEED)
    X, y, g = load_features()
    print(f"{len(y)} ROIs, {len(set(g))} patients, {X.shape[1]} features\n")

    models = {
        "SVM-RBF (deployed candidate)":
            lambda: SVC(C=3.0, probability=True, class_weight="balanced", random_state=SEED),
        "Logistic regression":
            lambda: LogisticRegression(max_iter=4000, class_weight="balanced", random_state=SEED),
        "SVM-RBF, high C=100 (deliberately overfit)":
            lambda: SVC(C=100.0, gamma="scale", probability=True,
                        class_weight="balanced", random_state=SEED),
    }

    print("=== TRAIN vs HELD-OUT GAP (grouped 5-fold) ===")
    print(f"{'model':<44} {'train':>7} {'held-out':>9} {'gap':>7}   verdict")
    print("-" * 92)
    gaps = {}
    for name, mk in models.items():
        tr, te = gap_for(mk, X, y, g)
        gap = tr - te
        v = ("severe overfitting" if gap > 0.20 else
             "moderate overfitting" if gap > 0.10 else
             "mild overfitting" if gap > 0.05 else
             "well generalised")
        print(f"{name:<44} {tr:>7.4f} {te:>9.4f} {gap:>+7.4f}   {v}")
        gaps[name] = {"train_auc": round(tr, 4), "heldout_auc": round(te, 4),
                      "gap": round(gap, 4), "verdict": v}

    # ---------------- learning curve ----------------
    print("\n=== LEARNING CURVE (would more patients help?) ===")
    pats = sorted(set(g))
    rng = np.random.default_rng(SEED)
    curve = []
    print(f"{'patients used':>14} {'held-out AUC':>13}")
    print("-" * 30)
    for frac in (0.3, 0.5, 0.7, 0.85, 1.0):
        n = max(6, int(len(pats) * frac))
        aucs = []
        for rep in range(3):
            sub = set(rng.choice(pats, n, replace=False))
            mask = np.array([p in sub for p in g])
            Xs, ys, gs = X[mask], y[mask], g[mask]
            if len(np.unique(ys)) < 2 or len(set(gs)) < 5:
                continue
            _, te = gap_for(lambda: SVC(C=3.0, probability=True,
                                        class_weight="balanced", random_state=SEED),
                            Xs, ys, gs, folds=4)
            aucs.append(te)
        if aucs:
            curve.append({"patients": n, "heldout_auc": round(float(np.mean(aucs)), 4)})
            print(f"{n:>14} {np.mean(aucs):>13.4f}")

    rising = (len(curve) >= 2 and curve[-1]["heldout_auc"] > curve[0]["heldout_auc"] + 0.02)
    plateau = (len(curve) >= 3 and
               abs(curve[-1]["heldout_auc"] - curve[-2]["heldout_auc"]) < 0.015)
    if plateau:
        lc = ("The curve has flattened: more patients of the same kind would give "
              "diminishing returns. Gains would have to come from better features or "
              "3D context, not more of the same data.")
    elif rising:
        lc = ("The curve is still rising: more annotated patients would likely improve "
              "performance further.")
    else:
        lc = "The curve is noisy at this scale; no confident conclusion."
    print(f"\n  {lc}")

    best = gaps["SVM-RBF (deployed candidate)"]
    out = {
        "question": "Is there still overfitting?",
        "n_rois": int(len(y)), "n_patients": len(set(g)),
        "gaps": gaps,
        "learning_curve": curve,
        "learning_curve_reading": lc,
        "answer": {
            "roi_feature_model": (
                f"train {best['train_auc']:.3f} vs held-out {best['heldout_auc']:.3f}, "
                f"gap {best['gap']:+.3f} - {best['verdict']}"),
            "caveat": (
                "This is cross-PATIENT within LIDC. It says nothing about cross-DATASET "
                "generalisation, which has never been tested for the ROI model because "
                "IQ-OTH carries no nodule annotations."),
            "deployed_platform_model": (
                "UNCHANGED. The clinical platform still serves the whole-image "
                "EfficientNet-B3 -> Quantum_4q model, which remains at chance externally "
                "(AUC 0.488, D26). The ROI work is a research result, not a deployment."),
        },
    }
    json.dump(out, open(REPORTS / "overfitting_check.json", "w"), indent=1)
    print("\nwrote reports/overfitting_check.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
