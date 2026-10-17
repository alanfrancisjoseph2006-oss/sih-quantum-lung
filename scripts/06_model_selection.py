"""
Phase 10/11 - statistically rigorous model selection + scientific validation.

WHY THIS SCRIPT EXISTS
----------------------
The naive router in 05_train_branches.py picks argmax(validation macro-F1). On
the DenseNet run, FOUR candidates tied at exactly 0.9470, and on the
EfficientNet run the quantum branch won on validation (0.9593) but then came
*last* on test (0.9011). With only 157 validation images, F1 differences below
roughly 0.05 are indistinguishable from noise, so argmax is close to a coin flip.

So selection here is significance-aware:

  1. Bootstrap the validation metric for every candidate.
  2. Paired bootstrap to test whether the best candidate is *significantly*
     better than each rival.
  3. Among the statistically-tied set, apply a documented CLINICAL tie-break:
     malignant sensitivity first (a missed cancer is the costliest error),
     then calibration (ECE), then model simplicity.

It also runs an explicit, falsifiable QUANTUM ADVANTAGE TEST and reports the
p-value whichever way it comes out.
"""
from __future__ import annotations
import json, sys, itertools
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import MODELS, EXPERIMENTS, REPORTS, CLASSES, SEED
from src.evaluation.metrics import (compute_metrics, curves, bootstrap_ci,
                                    expected_calibration_error, fast_f1_macro)

RNG = np.random.default_rng(SEED)
N_BOOT = 2000


def paired_bootstrap(y, pa, pb, metric="f1_macro", n_boot=N_BOOT, seed=SEED):
    """
    Paired bootstrap comparing model A vs model B on the SAME resampled indices.

    Returns the mean difference (A-B), a 95% CI, and a two-sided p-value for
    H0: no difference. Pairing matters -- it removes the variance that comes
    from which images landed in the sample, which is the dominant noise source
    on a set this small.
    """
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    n = len(y)
    n_cls = pa.shape[1]
    ka, kb = pa.argmax(1), pb.argmax(1)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        if metric == "f1_macro":
            diffs.append(fast_f1_macro(y[idx], ka[idx], n_cls)
                         - fast_f1_macro(y[idx], kb[idx], n_cls))
            continue
        try:
            da = compute_metrics(y[idx], pa[idx])[metric]
            db = compute_metrics(y[idx], pb[idx])[metric]
            if np.isfinite(da) and np.isfinite(db):
                diffs.append(da - db)
        except Exception:
            continue
    if not diffs:
        return {"mean_diff": float("nan"), "p_value": float("nan")}
    d = np.array(diffs)
    # two-sided p: proportion of resamples on the wrong side of zero
    p = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return {"mean_diff": round(float(d.mean()), 4),
            "ci95_lo": round(float(np.percentile(d, 2.5)), 4),
            "ci95_hi": round(float(np.percentile(d, 97.5)), 4),
            "p_value": round(float(min(p, 1.0)), 4),
            "significant_at_0.05": bool(min(p, 1.0) < 0.05)}


def main():
    summary = {"backbones": {}, "seed": SEED, "n_bootstrap": N_BOOT}

    for tag in ("efficientnet_b3", "densenet121"):
        f = MODELS / f"probs_{tag}.npz"
        if not f.exists():
            print(f"skip {tag}: no probs file")
            continue
        D = np.load(f)
        yva, yte = D["y_val"].astype(int), D["y_test"].astype(int)

        # Assemble every candidate model under one naming scheme.
        cands = {"CNN_DeepLearning": ("cnn_val", "cnn_test")}
        for k in D.files:
            if k.startswith("cls_") and k.endswith("_val"):
                nm = k[4:-4]
                cands[nm] = (k, f"cls_{nm}_test")
            elif k.startswith("q") and k.endswith("_val") and k[1:-4].isdigit():
                nq = k[1:-4]
                cands[f"Quantum_{nq}q"] = (k, f"q{nq}_test")
            elif k.startswith("fus_") and k.endswith("_val"):
                nm = k[4:-4]
                cands[f"Hybrid_{nm}"] = (k, f"fus_{nm}_test")

        print(f"\n{'='*74}\nBACKBONE: {tag}   candidates: {len(cands)}\n{'='*74}")
        rows = {}
        for nm, (kv, kt) in cands.items():
            if kv not in D.files or kt not in D.files:
                continue
            pv, pt = D[kv], D[kt]
            vm, tm = compute_metrics(yva, pv), compute_metrics(yte, pt)
            rows[nm] = {
                "val": vm, "test": tm,
                "val_ci": bootstrap_ci(yva, pv, "f1_macro", 1000, SEED),
                "test_ci": bootstrap_ci(yte, pt, "f1_macro", 1000, SEED),
                "val_calibration": expected_calibration_error(yva, pv),
                "test_calibration": expected_calibration_error(yte, pt),
                "_pv": pv, "_pt": pt,
            }
            print(f"  {nm:26s} valF1={vm['f1_macro']:.4f}  "
                  f"testF1={tm['f1_macro']:.4f} "
                  f"[{rows[nm]['test_ci']['lo95']:.3f}-{rows[nm]['test_ci']['hi95']:.3f}]  "
                  f"testAUC={tm['roc_auc_macro_ovr']:.4f}  "
                  f"malSens={tm.get('malignant_sensitivity', float('nan')):.3f}  "
                  f"ECE={rows[nm]['test_calibration']['ece']:.3f}")

        # ---------------- significance-aware selection on VALIDATION ----------
        best_nm = max(rows, key=lambda k: rows[k]["val"]["f1_macro"])
        tied = []
        comparisons = {}
        for nm in rows:
            if nm == best_nm:
                tied.append(nm); continue
            cmp = paired_bootstrap(yva, rows[best_nm]["_pv"], rows[nm]["_pv"])
            comparisons[nm] = cmp
            if not cmp["significant_at_0.05"]:
                tied.append(nm)

        print(f"\n  argmax on validation: {best_nm} "
              f"(valF1={rows[best_nm]['val']['f1_macro']:.4f})")
        print(f"  statistically INDISTINGUISHABLE from it at p<0.05: {len(tied)} candidates")
        print(f"    -> {', '.join(sorted(tied))}")

        # Clinical tie-break, applied only within the statistically tied set.
        # Every term is a VALIDATION quantity: using test calibration here would
        # leak the held-out split into model selection.
        def tiebreak_key(nm):
            t = rows[nm]["val"]
            return (round(t.get("malignant_sensitivity", 0.0), 3),   # never miss a cancer
                    round(t["f1_macro"], 3),
                    -rows[nm]["val_calibration"]["ece"])
        primary = max(tied, key=tiebreak_key)
        print(f"  clinical tie-break (malignant sensitivity > macro-F1 > calibration)"
              f"\n    -> PRIMARY MODEL: {primary}")

        # ---------------- explicit quantum advantage test on TEST -------------
        classical_pool = [n for n in rows if n.startswith(("CNN_", "RandomForest", "SVM",
                                                          "XGBoost", "Logistic"))]
        quantum_pool = [n for n in rows if n.startswith("Quantum_")]
        hybrid_pool = [n for n in rows if n.startswith("Hybrid_")]
        best_cls = max(classical_pool, key=lambda k: rows[k]["val"]["f1_macro"])
        qa = {}
        if quantum_pool:
            bq = max(quantum_pool, key=lambda k: rows[k]["val"]["f1_macro"])
            qa["quantum_vs_classical"] = {
                "quantum_model": bq, "classical_model": best_cls,
                **paired_bootstrap(yte, rows[bq]["_pt"], rows[best_cls]["_pt"])}
        if hybrid_pool:
            bh = max(hybrid_pool, key=lambda k: rows[k]["val"]["f1_macro"])
            qa["hybrid_vs_classical"] = {
                "hybrid_model": bh, "classical_model": best_cls,
                **paired_bootstrap(yte, rows[bh]["_pt"], rows[best_cls]["_pt"])}

        print(f"\n  --- QUANTUM ADVANTAGE TEST (paired bootstrap on TEST, macro-F1) ---")
        for k, v in qa.items():
            verdict = ("SIGNIFICANT" if v.get("significant_at_0.05") else
                       "NOT significant - no advantage demonstrated")
            print(f"    {k}: diff={v['mean_diff']:+.4f} "
                  f"[{v.get('ci95_lo', float('nan')):+.3f},{v.get('ci95_hi', float('nan')):+.3f}] "
                  f"p={v['p_value']:.3f} -> {verdict}")

        summary["backbones"][tag] = {
            "models": {nm: {k: v for k, v in r.items() if not k.startswith("_")}
                       for nm, r in rows.items()},
            "argmax_validation": best_nm,
            "statistically_tied": sorted(tied),
            "primary_model": primary,
            "selection_policy": (
                "1) argmax macro-F1 on validation; 2) paired bootstrap (2000 resamples) "
                "to find all candidates not significantly worse at p<0.05; "
                "3) clinical tie-break within that set: malignant sensitivity, then "
                "macro-F1, then calibration error. The test split is never used to select."),
            "pairwise_vs_argmax": comparisons,
            "quantum_advantage_test": qa,
            "curves_test": {nm: curves(yte, rows[nm]["_pt"]) for nm in rows},
        }

    # ---- choose the overall deployed configuration (validation only) ---------
    # Naive argmax across backbones would reintroduce exactly the noise-chasing
    # this script exists to prevent (0.9593 vs 0.9470 on 157 images is not a
    # real difference). So the same clinical rule is applied across backbones.
    finalists = []
    for tag, b in summary["backbones"].items():
        nm = b["primary_model"]
        m = b["models"][nm]
        finalists.append({
            "backbone": tag, "primary_model": nm,
            "val_f1": m["val"]["f1_macro"],
            "val_mal_sens": m["val"].get("malignant_sensitivity", 0.0),
            "val_ece": m["val_calibration"]["ece"],
            "val_auc": m["val"]["roc_auc_macro_ovr"]})
    top_f1 = max(f["val_f1"] for f in finalists)
    # anything within 0.05 macro-F1 on 157 images is a statistical tie
    tied_f = [f for f in finalists if top_f1 - f["val_f1"] <= 0.05]
    dep = max(tied_f, key=lambda f: (round(f["val_mal_sens"], 3),
                                     round(f["val_auc"], 3),
                                     -f["val_ece"]))
    summary["deployed"] = {
        "backbone": dep["backbone"], "primary_model": dep["primary_model"],
        "validation_f1_macro": round(dep["val_f1"], 4),
        "validation_malignant_sensitivity": round(dep["val_mal_sens"], 4),
        "validation_ece": dep["val_ece"],
        "finalists": finalists,
        "policy": ("Backbones within 0.05 validation macro-F1 of the leader are "
                   "treated as tied; the winner is then chosen by validation "
                   "malignant sensitivity, then macro ROC-AUC, then calibration.")}
    best_backbone = dep["backbone"]
    print(f"\n{'='*74}\nDEPLOYED CONFIGURATION: backbone={best_backbone}  "
          f"primary={summary['deployed']['primary_model']}\n{'='*74}")

    out = REPORTS / "model_comparison.json"
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    with open(EXPERIMENTS / "model_selection.json", "w") as fh:
        json.dump({k: v for k, v in summary.items() if k != "backbones"}, fh, indent=1, default=str)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
