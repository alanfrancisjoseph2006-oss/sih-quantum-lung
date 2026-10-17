"""
Phase 13e - did adding LIDC actually help? Measured on the UNCHANGED IQ-OTH test set.

Both models are evaluated on exactly the same 156 IQ-OTH test images, so the
comparison is apples-to-apples and a dataset shortcut cannot inflate it:

    baseline : trained on IQ-OTH train only          (735 images)
    lidc     : trained on IQ-OTH train + LIDC slices

Significance is a paired bootstrap on the test split, the same machinery used
for the quantum-advantage test. The decision policy from D20 is applied to both
arms so the comparison isolates the effect of the DATA, not of the threshold.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODELS, REPORTS, EXPERIMENTS, CLASSES, SEED
from src.evaluation.metrics import compute_metrics, bootstrap_ci, fast_f1_macro

BEN, MAL, NOR = 0, 1, 2


def paired_bootstrap(y, pa, pb, n_boot=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    n, ncls = len(y), pa.shape[1]
    ka, kb = pa.argmax(1), pb.argmax(1)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(y[i])) < 2:
            continue
        d.append(fast_f1_macro(y[i], ka[i], ncls) - fast_f1_macro(y[i], kb[i], ncls))
    d = np.array(d)
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return {"mean_diff": round(float(d.mean()), 4),
            "ci95_lo": round(float(np.percentile(d, 2.5)), 4),
            "ci95_hi": round(float(np.percentile(d, 97.5)), 4),
            "p_value": round(float(min(p, 1.0)), 4),
            "significant_at_0.05": bool(min(p, 1.0) < 0.05)}


def policy_weights():
    p = REPORTS / "decision_policy.json"
    if not p.exists():
        return np.ones(len(CLASSES))
    w = json.load(open(p)).get("weights", {})
    return np.array([float(w.get(c, 1.0)) for c in CLASSES])


def summarise(name, y, p):
    m = compute_metrics(y, p)
    pred = p.argmax(1)
    return {"name": name,
            "f1_macro": m["f1_macro"], "accuracy": m["accuracy"],
            "benign_recall": m["per_class"]["benign"]["recall_sensitivity"],
            "malignant_recall": m["per_class"]["malignant"]["recall_sensitivity"],
            "normal_recall": m["per_class"]["normal"]["recall_sensitivity"],
            "benign_precision": m["per_class"]["benign"]["precision"],
            "roc_auc": m["roc_auc_macro_ovr"],
            "missed_findings": int(((pred == NOR) & (y != NOR)).sum()),
            "confusion": m["confusion_matrix"]}


def show(r):
    print(f"  {r['name']:22s} F1={r['f1_macro']:.4f} acc={r['accuracy']:.4f} "
          f"AUC={r['roc_auc']:.4f} | benign={r['benign_recall']:.3f} "
          f"malig={r['malignant_recall']:.3f} normal={r['normal_recall']:.3f} "
          f"| missed={r['missed_findings']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="efficientnet_b3")
    ap.add_argument("--lidc", default="effb3_lidc")
    args = ap.parse_args()

    W = policy_weights()
    print(f"decision policy weights: {dict(zip(CLASSES, W))}\n")

    out = {"models": {}, "note": (
        "Both arms evaluated on the identical, unchanged IQ-OTH test split "
        "(n=156). LIDC data entered TRAIN only; val and test remained 100% "
        "IQ-OTH, so a dataset shortcut cannot inflate these numbers.")}

    arms = {}
    for label, tag in (("baseline", args.baseline), ("lidc", args.lidc)):
        f = MODELS / f"probs_{tag}.npz"
        if not f.exists():
            print(f"missing {f.name} - run 04/05 for tag '{tag}'"); return 1
        D = np.load(f)
        y = D["y_test"].astype(int)
        arms[label] = {"y": y, "D": D, "tag": tag}

    ya, yb = arms["baseline"]["y"], arms["lidc"]["y"]
    assert np.array_equal(ya, yb), "test sets differ - comparison would be invalid"
    y = ya
    print(f"test set verified identical across both arms: n={len(y)} "
          f"{dict(zip(CLASSES, np.bincount(y, minlength=3)))}\n")

    # Compare each branch that exists in both arms.
    branches = [("CNN", "cnn_test"), ("Quantum_4q", "q4_test"),
                ("RandomForest", "cls_RandomForest_test"),
                ("SVM_RBF", "cls_SVM_RBF_test"),
                ("Hybrid_meta", "fus_meta_classifier_test")]

    print("=== RAW MODEL OUTPUT (no decision policy) ===")
    for bname, key in branches:
        if key not in arms["baseline"]["D"].files or key not in arms["lidc"]["D"].files:
            continue
        pa = arms["baseline"]["D"][key]
        pb = arms["lidc"]["D"][key]
        ra, rb = summarise(f"{bname} base", y, pa), summarise(f"{bname} +LIDC", y, pb)
        show(ra); show(rb)
        cmp = paired_bootstrap(y, pb, pa)
        verdict = "SIGNIFICANT" if cmp["significant_at_0.05"] else "not significant"
        print(f"    delta macro-F1 = {cmp['mean_diff']:+.4f} "
              f"[{cmp['ci95_lo']:+.3f},{cmp['ci95_hi']:+.3f}] p={cmp['p_value']:.3f} -> {verdict}\n")
        out["models"][bname] = {"baseline": ra, "lidc": rb, "paired_test": cmp}

    print("=== WITH DECISION POLICY APPLIED TO BOTH ARMS ===")
    out["with_policy"] = {}
    for bname, key in branches:
        if key not in arms["baseline"]["D"].files or key not in arms["lidc"]["D"].files:
            continue
        pa = arms["baseline"]["D"][key] * W
        pb = arms["lidc"]["D"][key] * W
        pa = pa / pa.sum(1, keepdims=True)
        pb = pb / pb.sum(1, keepdims=True)
        ra, rb = summarise(f"{bname} base", y, pa), summarise(f"{bname} +LIDC", y, pb)
        show(ra); show(rb)
        cmp = paired_bootstrap(y, pb, pa)
        verdict = "SIGNIFICANT" if cmp["significant_at_0.05"] else "not significant"
        print(f"    delta macro-F1 = {cmp['mean_diff']:+.4f} "
              f"[{cmp['ci95_lo']:+.3f},{cmp['ci95_hi']:+.3f}] p={cmp['p_value']:.3f} -> {verdict}\n")
        out["with_policy"][bname] = {"baseline": ra, "lidc": rb, "paired_test": cmp}

    # headline: best of each arm by test macro-F1 among the compared branches
    def best(arm_key):
        rows = [(b, out["models"][b][arm_key]) for b in out["models"]]
        return max(rows, key=lambda r: r[1]["f1_macro"])
    bb, br = best("baseline")
    lb, lr = best("lidc")
    print("=== HEADLINE ===")
    print(f"  best baseline branch : {bb:14s} F1={br['f1_macro']:.4f} "
          f"benign={br['benign_recall']:.3f} missed={br['missed_findings']}")
    print(f"  best +LIDC branch    : {lb:14s} F1={lr['f1_macro']:.4f} "
          f"benign={lr['benign_recall']:.3f} missed={lr['missed_findings']}")

    out["headline"] = {"best_baseline": {"branch": bb, **br},
                       "best_lidc": {"branch": lb, **lr}}
    json.dump(out, open(EXPERIMENTS / "lidc_effect.json", "w"), indent=1, default=float)
    print(f"\nwrote reports/experiments/lidc_effect.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
