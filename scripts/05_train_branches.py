"""
Phases 4, 7, 8, 10 - everything that consumes the cached CNN embeddings.

  1. Classical baselines on the embedding : Random Forest, SVM, XGBoost, LogReg
  2. Quantum branch                       : PennyLane VQC, swept over 4/6/8 qubits
  3. Hybrid fusion                        : 4 strategies, selected on VALIDATION
  4. Adaptive router                      : picks the primary model on VALIDATION

Critical discipline: PCA/scalers are fit on TRAIN ONLY, every model is selected
on VALIDATION, and the TEST split is touched exactly once at the end for
reporting. The test set is never used to choose anything.
"""
from __future__ import annotations
import argparse, json, time, sys, pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb

from src.config import (MODELS, EXPERIMENTS, SEED, CLASSES, NUM_CLASSES,
                        PCA_COMPONENTS, N_QUBITS_SWEEP, Q_LAYERS, set_seed)
from src.quantum.circuits import QuantumClassifier
from src.evaluation.metrics import compute_metrics

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
def load_embeddings(tag: str):
    d = np.load(MODELS / f"embeddings_{tag}.npz")
    return {k: d[k] for k in d.files}


def fit_reducers(Xtr: np.ndarray, n_comp: int):
    """Scaler + PCA fitted on TRAIN ONLY -- fitting on val/test would leak."""
    sc = StandardScaler().fit(Xtr)
    n_comp = min(n_comp, Xtr.shape[0], Xtr.shape[1])
    pca = PCA(n_components=n_comp, random_state=SEED).fit(sc.transform(Xtr))
    return sc, pca


def apply_reducers(sc, pca, X):
    return pca.transform(sc.transform(X))


# ---------------------------------------------------------------------------
def train_classical(Ztr, ytr, Zva, yva, Zte, yte) -> dict:
    """Classical comparison baselines on the reduced CNN embedding."""
    out = {}
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=SEED, n_jobs=-1),
        "SVM_RBF": SVC(C=10.0, gamma="scale", probability=True,
                       class_weight="balanced", random_state=SEED),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.5,
            objective="multi:softprob", num_class=NUM_CLASSES,
            tree_method="hist", random_state=SEED, eval_metric="mlogloss"),
        "LogisticRegression": LogisticRegression(
            max_iter=3000, C=1.0, class_weight="balanced", random_state=SEED),
    }
    # XGBoost has no class_weight, so use per-sample weights instead.
    cls_w = len(ytr) / (NUM_CLASSES * np.bincount(ytr, minlength=NUM_CLASSES).clip(1))
    sw = cls_w[ytr]

    for name, m in models.items():
        t0 = time.time()
        if name == "XGBoost":
            m.fit(Ztr, ytr, sample_weight=sw)
        else:
            m.fit(Ztr, ytr)
        fit_s = time.time() - t0

        pva, pte = m.predict_proba(Zva), m.predict_proba(Zte)
        vm, tm = compute_metrics(yva, pva), compute_metrics(yte, pte)

        # 5-fold CV on train, for a stability estimate independent of the split
        try:
            cv = cross_val_score(m, Ztr, ytr, cv=StratifiedKFold(5, shuffle=True,
                                 random_state=SEED), scoring="f1_macro", n_jobs=1)
            cvm, cvs = float(cv.mean()), float(cv.std())
        except Exception:
            cvm = cvs = float("nan")

        out[name] = {"val": vm, "test": tm, "fit_sec": round(fit_s, 2),
                     "cv5_f1_macro_mean": round(cvm, 4), "cv5_f1_macro_std": round(cvs, 4),
                     "val_prob": pva, "test_prob": pte, "model": m}
        print(f"  {name:20s} valF1={vm['f1_macro']:.4f} testF1={tm['f1_macro']:.4f} "
              f"testAUC={tm['roc_auc_macro_ovr']:.4f} cv5={cvm:.4f}+/-{cvs:.4f} ({fit_s:.1f}s)")
    return out


# ---------------------------------------------------------------------------
def train_quantum(Ztr, ytr, Zva, yva, Zte, yte, n_qubits: int,
                  epochs: int = 60, lr: float = 0.02, batch: int = 32) -> dict:
    """
    Train the variational quantum classifier on reduced features.

    Runs on CPU: the statevector simulator is not GPU-accelerated here, and for
    <=8 qubits CPU is faster than paying host<->device transfer per batch.
    """
    set_seed(SEED)
    model = QuantumClassifier(Ztr.shape[1], n_qubits, NUM_CLASSES, Q_LAYERS).to("cpu")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    w = torch.tensor(len(ytr) / (NUM_CLASSES * np.bincount(ytr, minlength=NUM_CLASSES).clip(1)),
                     dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)

    Xtr_t = torch.tensor(Ztr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xva_t = torch.tensor(Zva, dtype=torch.float32)
    Xte_t = torch.tensor(Zte, dtype=torch.float32)

    best = {"f1": -1, "state": None, "epoch": 0}
    hist = []
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        idx = rng.permutation(len(ytr))
        tot, nb = 0.0, 0
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            if len(b) < 2:            # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            loss = crit(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss); nb += 1
        sched.step()

        model.eval()
        with torch.no_grad():
            pva = torch.softmax(model(Xva_t), 1).numpy()
        f1 = compute_metrics(yva, pva)["f1_macro"]
        hist.append({"epoch": ep + 1, "loss": round(tot / max(nb, 1), 4),
                     "val_f1": round(f1, 4)})
        if f1 > best["f1"]:
            best = {"f1": f1, "state": {k: v.clone() for k, v in model.state_dict().items()},
                    "epoch": ep + 1}
        if (ep + 1) % 15 == 0:
            print(f"    q{n_qubits} ep{ep+1:03d} loss={tot/max(nb,1):.4f} valF1={f1:.4f} "
                  f"(best {best['f1']:.4f} @ {best['epoch']})", flush=True)

    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        pva = torch.softmax(model(Xva_t), 1).numpy()
        pte = torch.softmax(model(Xte_t), 1).numpy()
        _, qf_tr = model(Xtr_t, return_qfeat=True)
        _, qf_va = model(Xva_t, return_qfeat=True)
        _, qf_te = model(Xte_t, return_qfeat=True)

    vm, tm = compute_metrics(yva, pva), compute_metrics(yte, pte)
    train_s = time.time() - t0
    print(f"  Quantum-{n_qubits}q      valF1={vm['f1_macro']:.4f} testF1={tm['f1_macro']:.4f} "
          f"testAUC={tm['roc_auc_macro_ovr']:.4f} ({train_s:.0f}s, best ep {best['epoch']})")

    return {"n_qubits": n_qubits, "val": vm, "test": tm, "val_prob": pva, "test_prob": pte,
            "qfeat": {"train": qf_tr.numpy(), "val": qf_va.numpy(), "test": qf_te.numpy()},
            "model": model, "history": hist, "train_sec": round(train_s, 1),
            "best_epoch": best["epoch"], "circuit": model.q.circuit_info()}


# ---------------------------------------------------------------------------
class FusionNet(nn.Module):
    """Small MLP over [classical probs | quantum probs | quantum features]."""

    def __init__(self, in_dim: int, hidden: int = 32, n_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.SiLU(),
            nn.Dropout(0.2), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)


def train_fusion(feats_tr, ytr, feats_va, yva, feats_te, yte,
                 epochs: int = 200, lr: float = 5e-3) -> dict:
    set_seed(SEED)
    model = FusionNet(feats_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = torch.tensor(len(ytr) / (NUM_CLASSES * np.bincount(ytr, minlength=NUM_CLASSES).clip(1)),
                     dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.03)

    Xtr = torch.tensor(feats_tr, dtype=torch.float32)
    Ytr = torch.tensor(ytr, dtype=torch.long)
    Xva = torch.tensor(feats_va, dtype=torch.float32)
    Xte = torch.tensor(feats_te, dtype=torch.float32)

    best = {"f1": -1, "state": None, "epoch": 0}
    rng = np.random.default_rng(SEED)
    for ep in range(epochs):
        model.train()
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]
            if len(b) < 2:
                continue
            opt.zero_grad()
            crit(model(Xtr[b]), Ytr[b]).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pva = torch.softmax(model(Xva), 1).numpy()
        f1 = compute_metrics(yva, pva)["f1_macro"]
        if f1 > best["f1"]:
            best = {"f1": f1, "state": {k: v.clone() for k, v in model.state_dict().items()},
                    "epoch": ep + 1}
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        pva = torch.softmax(model(Xva), 1).numpy()
        pte = torch.softmax(model(Xte), 1).numpy()
    return {"val": compute_metrics(yva, pva), "test": compute_metrics(yte, pte),
            "val_prob": pva, "test_prob": pte, "model": model, "best_epoch": best["epoch"]}


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="efficientnet_b3")
    ap.add_argument("--q-epochs", type=int, default=60)
    args = ap.parse_args()

    set_seed(SEED)
    E = load_embeddings(args.tag)
    Xtr, ytr = E["train_X"], E["train_y"].astype(int)
    Xva, yva = E["val_X"], E["val_y"].astype(int)
    Xte, yte = E["test_X"], E["test_y"].astype(int)
    cnn_tr, cnn_va, cnn_te = E["train_p"], E["val_p"], E["test_p"]
    print(f"embeddings: train{Xtr.shape} val{Xva.shape} test{Xte.shape}")

    sc, pca = fit_reducers(Xtr, PCA_COMPONENTS)
    Ztr, Zva, Zte = (apply_reducers(sc, pca, X) for X in (Xtr, Xva, Xte))
    evr = float(pca.explained_variance_ratio_.sum())
    print(f"PCA {Xtr.shape[1]} -> {Ztr.shape[1]} dims, explained variance {evr:.4f}\n")

    results = {"cnn_deep_learning": {
        "val": compute_metrics(yva, cnn_va), "test": compute_metrics(yte, cnn_te)}}
    print(f"  {'CNN(EfficientNet-B3)':20s} "
          f"valF1={results['cnn_deep_learning']['val']['f1_macro']:.4f} "
          f"testF1={results['cnn_deep_learning']['test']['f1_macro']:.4f}")

    print("\n=== CLASSICAL BASELINES ON CNN EMBEDDING ===")
    classical = train_classical(Ztr, ytr, Zva, yva, Zte, yte)

    print("\n=== QUANTUM BRANCH SWEEP ===")
    quantum = {}
    for nq in N_QUBITS_SWEEP:
        quantum[nq] = train_quantum(Ztr, ytr, Zva, yva, Zte, yte, nq, epochs=args.q_epochs)

    best_nq = max(quantum, key=lambda k: quantum[k]["val"]["f1_macro"])
    q = quantum[best_nq]
    print(f"\nbest qubit count on VALIDATION: {best_nq} "
          f"(valF1={q['val']['f1_macro']:.4f})")

    # ---- best classical on validation, used as the fusion partner ----
    best_cls_name = max(classical, key=lambda k: classical[k]["val"]["f1_macro"])
    bc = classical[best_cls_name]
    print(f"best classical head on VALIDATION: {best_cls_name} "
          f"(valF1={bc['val']['f1_macro']:.4f})")

    # ---- fusion strategies ----
    print("\n=== HYBRID FUSION STRATEGIES ===")
    fusion = {}

    # 1. weighted average of classical + quantum probabilities; weight on VAL
    best_w, best_f1 = 0.5, -1
    for wgt in np.linspace(0, 1, 21):
        p = wgt * bc["val_prob"] + (1 - wgt) * q["val_prob"]
        f1 = compute_metrics(yva, p)["f1_macro"]
        if f1 > best_f1:
            best_f1, best_w = f1, float(wgt)
    pv = best_w * bc["val_prob"] + (1 - best_w) * q["val_prob"]
    pt = best_w * bc["test_prob"] + (1 - best_w) * q["test_prob"]
    fusion["weighted_average"] = {"val": compute_metrics(yva, pv), "test": compute_metrics(yte, pt),
                                  "val_prob": pv, "test_prob": pt, "weight_classical": best_w}
    print(f"  {'weighted_average':22s} w_classical={best_w:.2f} "
          f"valF1={fusion['weighted_average']['val']['f1_macro']:.4f} "
          f"testF1={fusion['weighted_average']['test']['f1_macro']:.4f}")

    # 2. neural fusion over concatenated [classical probs | quantum probs | qfeats]
    def cat(cp, qp, qf):
        return np.concatenate([cp, qp, qf], 1).astype(np.float32)
    ftr = cat(bc["model"].predict_proba(Ztr),
              torch.softmax(q["model"](torch.tensor(Ztr, dtype=torch.float32)), 1).detach().numpy(),
              q["qfeat"]["train"])
    fva = cat(bc["val_prob"], q["val_prob"], q["qfeat"]["val"])
    fte = cat(bc["test_prob"], q["test_prob"], q["qfeat"]["test"])
    fusion["neural_fusion"] = train_fusion(ftr, ytr, fva, yva, fte, yte)
    print(f"  {'neural_fusion':22s} in_dim={ftr.shape[1]} "
          f"valF1={fusion['neural_fusion']['val']['f1_macro']:.4f} "
          f"testF1={fusion['neural_fusion']['test']['f1_macro']:.4f}")

    # 3. meta-classifier (logistic regression stacking)
    meta = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)
    meta.fit(ftr, ytr)
    pv, pt = meta.predict_proba(fva), meta.predict_proba(fte)
    fusion["meta_classifier"] = {"val": compute_metrics(yva, pv), "test": compute_metrics(yte, pt),
                                 "val_prob": pv, "test_prob": pt, "model": meta}
    print(f"  {'meta_classifier':22s} "
          f"valF1={fusion['meta_classifier']['val']['f1_macro']:.4f} "
          f"testF1={fusion['meta_classifier']['test']['f1_macro']:.4f}")

    # 4. concatenation: CNN probs + quantum features into an XGBoost
    ctr = np.concatenate([cnn_tr, q["qfeat"]["train"]], 1)
    cva = np.concatenate([cnn_va, q["qfeat"]["val"]], 1)
    cte = np.concatenate([cnn_te, q["qfeat"]["test"]], 1)
    xcat = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.08,
                             objective="multi:softprob", num_class=NUM_CLASSES,
                             tree_method="hist", random_state=SEED, eval_metric="mlogloss")
    cw = len(ytr) / (NUM_CLASSES * np.bincount(ytr, minlength=NUM_CLASSES).clip(1))
    xcat.fit(ctr, ytr, sample_weight=cw[ytr])
    pv, pt = xcat.predict_proba(cva), xcat.predict_proba(cte)
    fusion["concatenation_xgb"] = {"val": compute_metrics(yva, pv), "test": compute_metrics(yte, pt),
                                   "val_prob": pv, "test_prob": pt, "model": xcat}
    print(f"  {'concatenation_xgb':22s} "
          f"valF1={fusion['concatenation_xgb']['val']['f1_macro']:.4f} "
          f"testF1={fusion['concatenation_xgb']['test']['f1_macro']:.4f}")

    best_fusion = max(fusion, key=lambda k: fusion[k]["val"]["f1_macro"])
    print(f"\nbest fusion strategy on VALIDATION: {best_fusion} "
          f"(valF1={fusion[best_fusion]['val']['f1_macro']:.4f})")

    # ---- ADAPTIVE ROUTER: choose the primary model purely on VALIDATION ----
    candidates = {
        "cnn_deep_learning": results["cnn_deep_learning"]["val"]["f1_macro"],
        f"classical_{best_cls_name}": bc["val"]["f1_macro"],
        f"quantum_{best_nq}q": q["val"]["f1_macro"],
        f"hybrid_{best_fusion}": fusion[best_fusion]["val"]["f1_macro"],
    }
    primary = max(candidates, key=candidates.get)
    print("\n=== ADAPTIVE ROUTER (selection on validation only) ===")
    for k, v in sorted(candidates.items(), key=lambda x: -x[1]):
        print(f"  {k:34s} valF1={v:.4f}{'   <== SELECTED PRIMARY' if k == primary else ''}")

    # ---- persist everything ----
    with open(MODELS / f"branches_{args.tag}.pkl", "wb") as fh:
        pickle.dump({"scaler": sc, "pca": pca,
                     "classical": {k: v["model"] for k, v in classical.items()},
                     "best_classical": best_cls_name,
                     "meta_classifier": fusion["meta_classifier"]["model"],
                     "concat_xgb": fusion["concatenation_xgb"]["model"],
                     "weighted_average_w": fusion["weighted_average"]["weight_classical"],
                     "best_fusion": best_fusion, "best_nq": best_nq,
                     "primary": primary, "classes": CLASSES}, fh)
    torch.save({"state_dict": q["model"].state_dict(), "n_qubits": best_nq,
                "in_dim": Ztr.shape[1], "n_layers": Q_LAYERS,
                "circuit": q["circuit"]}, MODELS / f"quantum_{args.tag}.pt")
    torch.save({"state_dict": fusion["neural_fusion"]["model"].state_dict(),
                "in_dim": int(ftr.shape[1])}, MODELS / f"fusion_{args.tag}.pt")
    np.savez_compressed(MODELS / f"qfeat_{args.tag}.npz", **{
        f"{k}_qfeat": v for k, v in q["qfeat"].items()})

    def strip(d):
        return {k: v for k, v in d.items()
                if k not in ("model", "val_prob", "test_prob", "qfeat")}

    exp = {"experiment": f"branches_{args.tag}",
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED,
           "embedding_dim": int(Xtr.shape[1]), "pca_components": int(Ztr.shape[1]),
           "pca_explained_variance": round(evr, 4),
           "quantum_layers": Q_LAYERS, "qubit_sweep": N_QUBITS_SWEEP,
           "cnn": results["cnn_deep_learning"],
           "classical": {k: strip(v) for k, v in classical.items()},
           "quantum": {str(k): strip(v) for k, v in quantum.items()},
           "fusion": {k: strip(v) for k, v in fusion.items()},
           "selection": {"best_classical": best_cls_name, "best_n_qubits": best_nq,
                         "best_fusion": best_fusion, "primary_model": primary,
                         "validation_f1_candidates": candidates,
                         "policy": "argmax macro-F1 on the validation split; "
                                   "the test split is never used for selection"}}
    with open(EXPERIMENTS / f"branches_{args.tag}.json", "w") as fh:
        json.dump(exp, fh, indent=1, default=str)

    # probabilities kept for later curve/calibration analysis
    np.savez_compressed(MODELS / f"probs_{args.tag}.npz",
        y_val=yva, y_test=yte,
        cnn_val=cnn_va, cnn_test=cnn_te,
        **{f"cls_{k}_val": v["val_prob"] for k, v in classical.items()},
        **{f"cls_{k}_test": v["test_prob"] for k, v in classical.items()},
        **{f"q{k}_val": v["val_prob"] for k, v in quantum.items()},
        **{f"q{k}_test": v["test_prob"] for k, v in quantum.items()},
        **{f"fus_{k}_val": v["val_prob"] for k, v in fusion.items()},
        **{f"fus_{k}_test": v["test_prob"] for k, v in fusion.items()})
    print(f"\nsaved branches_{args.tag}.pkl, quantum_{args.tag}.pt, "
          f"fusion_{args.tag}.pt, experiments/branches_{args.tag}.json")


if __name__ == "__main__":
    main()
