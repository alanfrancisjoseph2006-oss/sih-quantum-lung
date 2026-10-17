"""
Phase 11b - targeted improvement of the weak class (benign).

DIAGNOSIS (measured, not assumed)
---------------------------------
Test recall: malignant 0.988, normal 0.967, **benign 0.733**.
All 4 benign errors are benign -> normal, i.e. a lesion called "no finding" -
the clinically dangerous direction.
Decision-threshold tuning was tried first and REJECTED: validation contains only
15 benign images with 1 error, so it cannot resolve a threshold. Tuning it on
test would be test-set fitting.

Root cause: only 70 benign training images (9.5% of the training set).

STRATEGIES TRIED (all selected on validation, or by grouped CV on train+val)
---------------------------------------------------------------------------
 A. Two-stage cascade: (finding vs normal) then (benign vs malignant).
    Motivated by the finding that benign/malignant separates almost perfectly,
    so the hard decision is really "is there a lesion at all".
 B. Benign-boosted cost-sensitive heads on the existing embeddings.
 C. Grouped cross-validated threshold tuning over train+val, which gives ~85
    benign examples to tune on instead of 15.
 D. Two-backbone ensemble (EfficientNet-B3 + DenseNet-121).

The test split is evaluated ONCE at the end, for reporting only.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold

from src.config import MODELS, REPORTS, EXPERIMENTS, CLASSES, SEED, set_seed, SPLITS
from src.evaluation.metrics import compute_metrics

BEN, MAL, NOR = 0, 1, 2


def load(tag):
    E = np.load(MODELS / f"embeddings_{tag}.npz")
    P = np.load(MODELS / f"probs_{tag}.npz")
    return E, P


def group_ids(split):
    import csv
    g = []
    with open(SPLITS / "splits.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["split"] == split:
                g.append(int(r["group_id"]))
    return np.array(g)


def summary(name, y, p):
    m = compute_metrics(y, p)
    pred = p.argmax(1)
    miss = int(((pred == NOR) & (y != NOR)).sum())
    return {"name": name, "f1_macro": m["f1_macro"],
            "benign_recall": m["per_class"]["benign"]["recall_sensitivity"],
            "malignant_recall": m["per_class"]["malignant"]["recall_sensitivity"],
            "normal_recall": m["per_class"]["normal"]["recall_sensitivity"],
            "missed_findings": miss, "accuracy": m["accuracy"],
            "roc_auc": m["roc_auc_macro_ovr"]}


def show(r):
    print(f"  {r['name']:34s} F1={r['f1_macro']:.4f} benign={r['benign_recall']:.3f} "
          f"malig={r['malignant_recall']:.3f} normal={r['normal_recall']:.3f} "
          f"missed={r['missed_findings']}")


def main():
    set_seed(SEED)
    tag = "efficientnet_b3"
    E, P = load(tag)
    Xtr, ytr = E["train_X"], E["train_y"].astype(int)
    Xva, yva = E["val_X"], E["val_y"].astype(int)
    Xte, yte = E["test_X"], E["test_y"].astype(int)
    gtr, gva = group_ids("train"), group_ids("val")

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    sc = StandardScaler().fit(Xtr)
    pca = PCA(n_components=32, random_state=SEED).fit(sc.transform(Xtr))
    Z = lambda X: pca.transform(sc.transform(X))
    Ztr, Zva, Zte = Z(Xtr), Z(Xva), Z(Xte)

    results = {"baseline": {}, "strategies": {}}
    base_v = summary("BASELINE Quantum_4q (val)", yva, P["q4_val"])
    base_t = summary("BASELINE Quantum_4q (test)", yte, P["q4_test"])
    results["baseline"] = {"val": base_v, "test": base_t}
    print("=== BASELINE ===")
    show(base_v); show(base_t)

    cand = {}   # name -> (val_prob, test_prob)

    # ---------------- A. two-stage cascade ----------------
    print("\n=== A. TWO-STAGE CASCADE ===")
    # stage 1: finding (benign|malignant) vs normal  -- better balanced: 454 vs 281
    s1_y = (ytr != NOR).astype(int)
    s1 = LogisticRegression(max_iter=4000, class_weight="balanced", C=1.0,
                            random_state=SEED).fit(Ztr, s1_y)
    # stage 2: benign vs malignant, trained only on lesion cases
    les = ytr != NOR
    s2 = LogisticRegression(max_iter=4000, class_weight="balanced", C=1.0,
                            random_state=SEED).fit(Ztr[les], ytr[les])

    def cascade(Zx):
        pf = s1.predict_proba(Zx)[:, 1]                  # P(finding)
        pbm = s2.predict_proba(Zx)                       # [P(benign|lesion), P(malig|lesion)]
        out = np.zeros((len(Zx), 3))
        out[:, BEN] = pf * pbm[:, 0]
        out[:, MAL] = pf * pbm[:, 1]
        out[:, NOR] = 1 - pf
        return out / out.sum(1, keepdims=True)

    cand["A_cascade"] = (cascade(Zva), cascade(Zte))
    show(summary("A_cascade (val)", yva, cand["A_cascade"][0]))

    # ---------------- B. benign-boosted heads ----------------
    print("\n=== B. BENIGN-BOOSTED COST-SENSITIVE HEADS ===")
    n = np.bincount(ytr, minlength=3)
    for boost in (2.0, 3.0, 4.0):
        w = {BEN: boost * len(ytr) / (3 * n[BEN]),
             MAL: len(ytr) / (3 * n[MAL]),
             NOR: len(ytr) / (3 * n[NOR])}
        lr = LogisticRegression(max_iter=4000, class_weight=w, C=1.0,
                                random_state=SEED).fit(Ztr, ytr)
        nm = f"B_logreg_benign_x{boost:g}"
        cand[nm] = (lr.predict_proba(Zva), lr.predict_proba(Zte))
        show(summary(nm + " (val)", yva, cand[nm][0]))

        rf = RandomForestClassifier(n_estimators=600, min_samples_leaf=1,
                                    class_weight=w, random_state=SEED,
                                    n_jobs=-1).fit(Ztr, ytr)
        nm = f"B_rf_benign_x{boost:g}"
        cand[nm] = (rf.predict_proba(Zva), rf.predict_proba(Zte))
        show(summary(nm + " (val)", yva, cand[nm][0]))

    # ---------------- C. grouped-CV threshold tuning ----------------
    # Tune the normal-class penalty using train+val under grouped CV, so the
    # threshold sees ~85 benign examples instead of the 15 in val alone.
    print("\n=== C. GROUPED-CV THRESHOLD TUNING (train+val) ===")
    Xall = np.vstack([Ztr, Zva]); yall = np.concatenate([ytr, yva])
    gall = np.concatenate([gtr, gva])
    ws = np.arange(0.20, 1.01, 0.05)
    scores = np.zeros(len(ws))
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr_i, te_i in sgkf.split(Xall, yall, groups=gall):
        m = LogisticRegression(max_iter=4000, class_weight="balanced",
                               random_state=SEED).fit(Xall[tr_i], yall[tr_i])
        pp = m.predict_proba(Xall[te_i]); yy = yall[te_i]
        for k, w in enumerate(ws):
            pred = (pp * np.array([1, 1, w])).argmax(1)
            # objective: macro-F1, which balances benign recall against
            # over-flagging normals
            import sklearn.metrics as M
            scores[k] += M.f1_score(yy, pred, average="macro", zero_division=0)
    scores /= 5
    best_w = float(ws[int(scores.argmax())])
    print(f"  grouped-CV best normal-weight = {best_w:.2f} (CV macro-F1 {scores.max():.4f})")
    print("  CV curve:", " ".join(f"{w:.2f}:{s:.3f}" for w, s in zip(ws, scores)))
    Wv = np.array([1, 1, best_w])
    cand["C_cv_threshold_on_quantum"] = (P["q4_val"] * Wv, P["q4_test"] * Wv)
    show(summary("C_cv_threshold (val)", yva, cand["C_cv_threshold_on_quantum"][0]))

    # ---------------- D. two-backbone ensemble ----------------
    print("\n=== D. TWO-BACKBONE ENSEMBLE ===")
    try:
        P2 = np.load(MODELS / "probs_densenet121.npz")
        assert np.array_equal(P2["y_val"].astype(int), yva), "split mismatch"
        for k1, k2, nm in [("cnn_val", "cnn_test", "D_ens_cnn"),
                           ("q4_val", "q4_test", "D_ens_quantum")]:
            ev = (P[k1] + P2[k1]) / 2
            et = (P[k2] + P2[k2]) / 2
            cand[nm] = (ev, et)
            show(summary(nm + " (val)", yva, ev))
        # cross-branch: EfficientNet CNN + DenseNet SVM
        ev = (P["cnn_val"] + P2["cls_SVM_RBF_val"]) / 2
        et = (P["cnn_test"] + P2["cls_SVM_RBF_test"]) / 2
        cand["D_ens_cnn_plus_densesvm"] = (ev, et)
        show(summary("D_ens_cnn_plus_densesvm (val)", yva, ev))
    except Exception as e:
        print("  skipped:", e)

    # ---------------- selection on VALIDATION ----------------
    print("\n=== SELECTION (validation only) ===")
    print("  policy: maximise benign recall, tie-break on macro-F1, subject to")
    print("          malignant recall >= baseline and normal recall >= 0.90")
    base_mal = base_v["malignant_recall"]
    ok = {}
    for nm, (pv, pt) in cand.items():
        r = summary(nm, yva, pv)
        if r["malignant_recall"] >= base_mal - 1e-9 and r["normal_recall"] >= 0.90:
            ok[nm] = r
        results["strategies"][nm] = {"val": r}
    if not ok:
        print("  no candidate satisfied the safety constraints; keeping baseline")
        winner = None
    else:
        winner = max(ok, key=lambda k: (round(ok[k]["benign_recall"], 3),
                                        round(ok[k]["f1_macro"], 4)))
        for nm, r in sorted(ok.items(), key=lambda x: -x[1]["benign_recall"]):
            print(f"   {'->' if nm == winner else '  '} {nm:32s} "
                  f"benign={r['benign_recall']:.3f} F1={r['f1_macro']:.4f}")

    # ---------------- single test evaluation ----------------
    print("\n=== TEST (evaluated once, reporting only) ===")
    show(base_t)
    if winner:
        tr = summary(f"WINNER {winner}", yte, cand[winner][1])
        show(tr)
        results["winner"] = {"name": winner, "val": ok[winner], "test": tr}
        d_ben = tr["benign_recall"] - base_t["benign_recall"]
        d_f1 = tr["f1_macro"] - base_t["f1_macro"]
        print(f"\n  benign recall {base_t['benign_recall']:.3f} -> {tr['benign_recall']:.3f} "
              f"({d_ben:+.3f})")
        print(f"  macro-F1      {base_t['f1_macro']:.4f} -> {tr['f1_macro']:.4f} ({d_f1:+.4f})")
        print(f"  missed findings {base_t['missed_findings']} -> {tr['missed_findings']}")
        for nm, (pv, pt) in cand.items():
            results["strategies"][nm]["test"] = summary(nm, yte, pt)

    with open(EXPERIMENTS / "benign_improvement.json", "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print(f"\nwrote reports/experiments/benign_improvement.json")


if __name__ == "__main__":
    main()
