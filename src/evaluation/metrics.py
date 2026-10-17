"""
Evaluation metrics.

For an imbalanced medical task, accuracy is close to meaningless -- always
predicting 'malignant' already scores 52%. So every report here leads with
macro-F1, per-class recall (sensitivity), specificity and ROC-AUC, exactly as
the brief requires.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score, balanced_accuracy_score,
                             cohen_kappa_score, roc_curve, precision_recall_curve,
                             average_precision_score)

from src.config import CLASSES


def _specificity(cm: np.ndarray) -> np.ndarray:
    """Per-class specificity = TN / (TN + FP), one-vs-rest."""
    out = []
    total = cm.sum()
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        out.append(tn / max(tn + fp, 1))
    return np.array(out)


def compute_metrics(y_true, y_prob, classes: list[str] = CLASSES) -> dict:
    """
    y_true : (N,) int labels
    y_prob : (N, C) probabilities
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_prob.ndim == 1:
        y_prob = np.stack([1 - y_prob, y_prob], 1)
    # Guard against a diverged model emitting NaN/Inf: fall back to a uniform
    # distribution so evaluation reports a bad score instead of crashing.
    if not np.isfinite(y_prob).all():
        bad = ~np.isfinite(y_prob).all(1)
        y_prob = y_prob.copy()
        y_prob[bad] = 1.0 / y_prob.shape[1]
        y_prob = np.nan_to_num(y_prob, nan=1.0 / y_prob.shape[1], posinf=1.0, neginf=0.0)
    y_pred = y_prob.argmax(1)
    C = len(classes)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(C)))
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(C)), zero_division=0)
    spec = _specificity(cm)

    present = np.unique(y_true)
    try:
        auc_macro = (roc_auc_score(y_true, y_prob, multi_class="ovr",
                                   average="macro", labels=list(range(C)))
                     if len(present) > 1 else float("nan"))
    except ValueError:
        auc_macro = float("nan")

    per_class_auc = {}
    for i, c in enumerate(classes):
        yi = (y_true == i).astype(int)
        per_class_auc[c] = (float(roc_auc_score(yi, y_prob[:, i]))
                            if 0 < yi.sum() < len(yi) else float("nan"))

    # Clinically the decisive question is "is this malignant or not".
    mal = classes.index("malignant") if "malignant" in classes else None
    binary = {}
    if mal is not None:
        yb = (y_true == mal).astype(int)
        pb = y_prob[:, mal]
        pred_b = (y_pred == mal).astype(int)
        tp = int(((pred_b == 1) & (yb == 1)).sum()); fp = int(((pred_b == 1) & (yb == 0)).sum())
        tn = int(((pred_b == 0) & (yb == 0)).sum()); fn = int(((pred_b == 0) & (yb == 1)).sum())
        binary = {
            "malignant_sensitivity": tp / max(tp + fn, 1),
            "malignant_specificity": tn / max(tn + fp, 1),
            "malignant_ppv": tp / max(tp + fp, 1),
            "malignant_npv": tn / max(tn + fn, 1),
            "malignant_auc": float(roc_auc_score(yb, pb)) if 0 < yb.sum() < len(yb) else float("nan"),
            "malignant_ap": float(average_precision_score(yb, pb)) if 0 < yb.sum() < len(yb) else float("nan"),
            "malignant_fn_count": fn,
        }

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(p.mean()),
        "recall_macro": float(r.mean()),
        "f1_macro": float(f1.mean()),
        "f1_weighted": float(np.average(f1, weights=np.maximum(sup, 1))),
        "roc_auc_macro_ovr": float(auc_macro),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class": {
            c: {"precision": float(p[i]), "recall_sensitivity": float(r[i]),
                "specificity": float(spec[i]), "f1": float(f1[i]),
                "support": int(sup[i]), "roc_auc_ovr": per_class_auc[c]}
            for i, c in enumerate(classes)},
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": classes,
        **binary,
    }


def curves(y_true, y_prob, classes: list[str] = CLASSES) -> dict:
    """ROC and PR curve points, one-vs-rest, for the dashboard charts."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    out = {"roc": {}, "pr": {}}
    for i, c in enumerate(classes):
        yi = (y_true == i).astype(int)
        if not (0 < yi.sum() < len(yi)):
            continue
        fpr, tpr, _ = roc_curve(yi, y_prob[:, i])
        pr, rc, _ = precision_recall_curve(yi, y_prob[:, i])
        k = max(1, len(fpr) // 120)
        out["roc"][c] = {"fpr": fpr[::k].round(4).tolist(), "tpr": tpr[::k].round(4).tolist(),
                         "auc": round(float(roc_auc_score(yi, y_prob[:, i])), 4)}
        k2 = max(1, len(pr) // 120)
        out["pr"][c] = {"precision": pr[::k2].round(4).tolist(),
                        "recall": rc[::k2].round(4).tolist(),
                        "ap": round(float(average_precision_score(yi, y_prob[:, i])), 4)}
    return out


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> dict:
    """
    ECE over max-probability confidence.

    Included because a decision-support tool that is confidently wrong is more
    dangerous than one that is uncertain, so calibration is worth measuring.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    conf = y_prob.max(1)
    pred = y_prob.argmax(1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        acc, cf = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(acc - cf)
        bins.append({"bin": f"{edges[i]:.1f}-{edges[i+1]:.1f}", "n": int(m.sum()),
                     "accuracy": round(float(acc), 4), "confidence": round(float(cf), 4)})
    return {"ece": round(float(ece), 4), "bins": bins}


def fast_f1_macro(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """
    Macro-F1 without sklearn overhead.

    Bootstrapping calls this ~100k times; `compute_metrics` recomputes ROC-AUC
    on every call, which dominated runtime. This is the same number, ~50x faster.
    """
    f1s = 0.0
    for c in range(n_classes):
        tp = np.count_nonzero((y_pred == c) & (y_true == c))
        fp = np.count_nonzero((y_pred == c) & (y_true != c))
        fn = np.count_nonzero((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s += (2.0 * tp / denom) if denom else 0.0
    return f1s / n_classes


def bootstrap_ci(y_true, y_prob, metric: str = "f1_macro",
                 n_boot: int = 1000, seed: int = 42) -> dict:
    """
    Percentile bootstrap CI.

    With only 156 test images (15 of them benign), point estimates are noisy.
    Reporting an interval is the honest way to present them.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    n_cls = y_prob.shape[1]
    y_pred = y_prob.argmax(1)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        if metric == "f1_macro":
            vals.append(fast_f1_macro(y_true[idx], y_pred[idx], n_cls))
            continue
        try:
            v = compute_metrics(y_true[idx], y_prob[idx])[metric]
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            continue
    if not vals:
        return {"metric": metric, "mean": float("nan"), "lo95": float("nan"), "hi95": float("nan")}
    vals = np.array(vals)
    return {"metric": metric, "mean": round(float(vals.mean()), 4),
            "lo95": round(float(np.percentile(vals, 2.5)), 4),
            "hi95": round(float(np.percentile(vals, 97.5)), 4),
            "n_boot": len(vals)}
