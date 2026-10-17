"""
Phase 7 - Hybrid quantum on COMPACT NODULE FEATURES.

WHY THE QUANTUM BRANCH FINALLY HAS A DEFENSIBLE ROLE HERE
---------------------------------------------------------
In the original pipeline the quantum circuit consumed a 256-d CNN embedding
squeezed through PCA to 4 qubits. That was arbitrary: the embedding was already
the dominant signal, so the circuit added nothing measurable (D11: p = 0.71).

Script 26 changed the picture. On held-out LIDC patients:

    size + intensity (6 hand-crafted features)   AUC 0.898
    size alone (1 feature)                       AUC 0.858
    ROI CNN (EfficientNet-B3)                    AUC 0.680
    whole-image model                            AUC 0.457

With 35 training patients a compact, interpretable feature vector beats a deep
network. And a 6-12 dimensional vector is exactly the regime a small variational
circuit can consume WITHOUT a lossy PCA step - one feature per qubit, nothing
discarded.

So this compares, on identical patient-level splits:

    1. classical only   - logistic regression / SVM / random forest on the features
    2. quantum only     - variational circuit on the same features
    3. hybrid ensemble  - averaged posteriors

No quantum advantage is claimed unless a paired bootstrap supports it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import DATA, EXPERIMENTS, MODELS, REPORTS, SEED, set_seed
from src.quantum.circuits import QuantumClassifier

LIDC = DATA / "external" / "lidc"

FEATURE_NAMES = [
    "area_px", "equiv_diameter", "mean_intensity", "std_intensity",
    "p90_intensity", "p10_intensity", "perimeter", "circularity",
    "solidity", "contrast_vs_surround",
]


def extract(rows) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpretable nodule descriptors - shape, density and border contrast."""
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
        hull = cv2.convexHull(c)
        hull_a = float(cv2.contourArea(hull)) or 1.0
        v = img[m > 0].astype(np.float32)

        # surrounding parenchyma ring, for border contrast
        ring = cv2.dilate(m, np.ones((9, 9), np.uint8)) - m
        vs = img[ring > 0].astype(np.float32)
        contrast = float(v.mean() - vs.mean()) if vs.size else 0.0

        X.append([
            area,
            2.0 * np.sqrt(area / np.pi),
            float(v.mean()), float(v.std()),
            float(np.percentile(v, 90)), float(np.percentile(v, 10)),
            per,
            float(4 * np.pi * area / max(per ** 2, 1e-6)),   # circularity
            float(area / hull_a),                            # solidity
            contrast,
        ])
        y.append(1 if r["label"] == "malignant" else 0)
        g.append(r["patient_id"])
    return np.array(X, np.float32), np.array(y), np.array(g)


def boot_ci(y, s, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        v.append(roc_auc_score(y[i], s[i]))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)


def paired_boot(y, sa, sb, n=2000, seed=SEED):
    """Is model A better than model B? Paired on identical resamples."""
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], sa[i]) - roc_auc_score(y[i], sb[i]))
    d = np.array(d)
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return {"mean_diff": round(float(d.mean()), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4),
                     round(float(np.percentile(d, 97.5)), 4)],
            "p_value": round(float(min(p, 1.0)), 4),
            "significant": bool(min(p, 1.0) < 0.05)}


def summarise(name, y, s, g):
    lo, hi = boot_ci(y, s)
    out = {"name": name, "roc_auc": round(float(roc_auc_score(y, s)), 4),
           "roc_auc_ci95": [round(lo, 4), round(hi, 4)],
           "pr_auc": round(float(average_precision_score(y, s)), 4)}
    up = sorted(set(g))
    py = np.array([int(y[g == p].max()) for p in up])
    ps = np.array([float(s[g == p].max()) for p in up])
    if len(np.unique(py)) > 1:
        out["patient_auc"] = round(float(roc_auc_score(py, ps)), 4)
    return out


def train_quantum(Xtr, ytr, Xte, n_qubits, epochs=120, lr=0.03, seed=SEED):
    set_seed(seed)
    m = QuantumClassifier(Xtr.shape[1], n_qubits, n_classes=2)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = torch.tensor([1.0, max((ytr == 0).sum(), 1) / max((ytr == 1).sum(), 1)],
                     dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        m.train()
        idx = rng.permutation(len(yt))
        for i in range(0, len(idx), 32):
            b = idx[i:i + 32]
            if len(b) < 2:
                continue
            opt.zero_grad()
            crit(m(Xt[b]), yt[b]).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        sch.step()
    m.eval()
    with torch.no_grad():
        return m, torch.softmax(m(torch.tensor(Xte, dtype=torch.float32)), 1).numpy()[:, 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--q-epochs", type=int, default=120)
    args = ap.parse_args()

    set_seed(SEED)
    rows = list(csv.DictReader(open(LIDC / "lidc_masks.csv", newline="")))
    pats = sorted({r["patient_id"] for r in rows})
    rng = np.random.default_rng(SEED)
    rng.shuffle(pats)
    n = len(pats)
    tr_p = set(pats[:int(.70 * n)]); te_p = set(pats[int(.85 * n):])
    tr = [r for r in rows if r["patient_id"] in tr_p]
    te = [r for r in rows if r["patient_id"] in te_p]

    Xtr, ytr, _ = extract(tr)
    Xte, yte, gte = extract(te)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    print(f"patient-level split: {len(tr_p)} train / {len(te_p)} test patients")
    print(f"features {Xtr.shape[1]} ({', '.join(FEATURE_NAMES)})")
    print(f"train {len(ytr)} ROIs ({int(ytr.sum())} malignant) | "
          f"test {len(yte)} ROIs ({int(yte.sum())} malignant)\n")

    results, scores = [], {}

    # ---------------- classical ----------------
    for name, clf in [
        ("LogisticRegression", LogisticRegression(max_iter=4000, class_weight="balanced",
                                                  random_state=SEED)),
        ("SVM_RBF", SVC(C=3.0, probability=True, class_weight="balanced", random_state=SEED)),
        ("RandomForest", RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                                class_weight="balanced", random_state=SEED,
                                                n_jobs=-1)),
    ]:
        clf.fit(Xtr_s, ytr)
        s = clf.predict_proba(Xte_s)[:, 1]
        scores[name] = s
        results.append(summarise(name, yte, s, gte))

    # ---------------- quantum ----------------
    t0 = time.time()
    _, sq = train_quantum(Xtr_s, ytr, Xte_s, args.qubits, args.q_epochs)
    qt = time.time() - t0
    scores[f"Quantum_{args.qubits}q"] = sq
    results.append(summarise(f"Quantum_{args.qubits}q", yte, sq, gte))

    # ---------------- hybrid ----------------
    best_c = max([r for r in results if not r["name"].startswith("Quantum")],
                 key=lambda r: r["roc_auc"])["name"]
    sh = 0.5 * scores[best_c] + 0.5 * sq
    scores["Hybrid"] = sh
    results.append(summarise(f"Hybrid ({best_c} + quantum)", yte, sh, gte))

    print(f"{'model':<30} {'AUC':>8} {'95% CI':>18} {'PR-AUC':>8} {'patient AUC':>12}")
    print("-" * 80)
    for r in results:
        lo, hi = r["roc_auc_ci95"]
        ci = "[{:.3f}, {:.3f}]".format(lo, hi)
        pa = r.get("patient_auc")
        print(f"{r['name']:<30} {r['roc_auc']:>8.4f} {ci:>18} {r['pr_auc']:>8.4f} "
              f"{(f'{pa:.4f}' if pa is not None else '-'):>12}")

    print(f"\n  quantum circuit: {args.qubits} qubits, {args.q_epochs} epochs, "
          f"{qt:.0f}s (simulator)")

    # ---------------- advantage test ----------------
    print("\n=== QUANTUM ADVANTAGE TEST (paired bootstrap, held-out patients) ===")
    adv = {}
    for label, a in (("quantum vs best classical", sq), ("hybrid vs best classical", sh)):
        t = paired_boot(yte, a, scores[best_c])
        adv[label] = {**t, "classical_model": best_c}
        v = "SIGNIFICANT" if t["significant"] else "not significant - no advantage demonstrated"
        print(f"  {label}: delta AUC = {t['mean_diff']:+.4f} "
              f"[{t['ci95'][0]:+.3f},{t['ci95'][1]:+.3f}]  p = {t['p_value']:.3f}  -> {v}")

    out = {
        "task": "benign vs malignant on nodule ROI features (LIDC, patient-level split)",
        "why_features_not_embeddings": (
            "Script 26 measured that 6 hand-crafted features beat an EfficientNet-B3 "
            "on this data (0.898 vs 0.680) because only 35 training patients are "
            "available. A 10-d interpretable vector is also the natural input scale "
            "for a small variational circuit - one feature per qubit, no lossy PCA."),
        "features": FEATURE_NAMES,
        "n_qubits": args.qubits,
        "split": {"train_patients": len(tr_p), "test_patients": len(te_p),
                  "train_rois": int(len(ytr)), "test_rois": int(len(yte))},
        "results": results,
        "quantum_advantage_test": adv,
        "quantum_train_seconds": round(qt, 1),
        "reference_points": {
            "whole_image_model_on_LIDC": 0.4566,
            "roi_cnn": 0.6797,
            "size_only": 0.8584,
        },
    }
    json.dump(out, open(REPORTS / "hybrid_quantum_roi.json", "w"), indent=1)
    json.dump(out, open(EXPERIMENTS / "hybrid_quantum_roi.json", "w"), indent=1)
    print("\nwrote reports/hybrid_quantum_roi.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
