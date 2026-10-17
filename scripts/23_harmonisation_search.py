"""
Phase 3/5 - Which preprocessing actually REMOVES the dataset signature?

Script 22 found the signature survives ROI cropping (domain AUC 0.94), so it
lives in the lung texture itself. Before building any pipeline, find a transform
that collapses dataset separability while preserving anatomical structure.

Two numbers matter for every candidate, and they trade off:

    DOMAIN AUC   - can a probe still name the dataset?   LOWER is better (0.5 = ideal)
    SIGNAL AUC   - can a probe still separate benign     HIGHER is better
                   from malignant WITHIN LIDC?

A transform that destroys both is useless: an image of pure noise has domain
AUC 0.5. The goal is the transform with the largest drop in domain AUC for the
smallest drop in signal.

Both are evaluated with grouped cross-validation on real patient IDs.
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

from src.config import DATA, REPORTS, SEED, SPLITS, set_seed
from src.preprocessing.medical_image import load_image, lung_mask, preprocess

LIDC = DATA / "external" / "lidc"
SIZE = 300
N_PER = 260


# --------------------------------------------------------------- transforms
def t_identity(g, lm):
    return g


def t_lung_masked(g, lm):
    o = g.copy(); o[lm == 0] = 0
    return o


def t_zscore(g, lm):
    """Per-image standardisation inside the lung field only."""
    o = g.astype(np.float32)
    m = lm > 0
    if m.sum() < 50:
        return g
    mu, sd = o[m].mean(), o[m].std() + 1e-6
    o = (o - mu) / sd
    o = np.clip(o * 40 + 128, 0, 255)
    o[~m] = 0
    return o.astype(np.uint8)


def t_hist_eq(g, lm):
    o = cv2.equalizeHist(g); o[lm == 0] = 0
    return o


def t_gradient(g, lm):
    """Discard absolute intensity entirely - keep edge structure only."""
    gx = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = np.clip(mag / (np.percentile(mag, 99) + 1e-6) * 255, 0, 255)
    mag[lm == 0] = 0
    return mag.astype(np.uint8)


def t_bandpass(g, lm):
    """Remove low-frequency shading and very high-frequency noise/grain -
    the two bands where scanner and compression signatures concentrate."""
    x = g.astype(np.float32)
    low = cv2.GaussianBlur(x, (0, 0), 8)
    hi = cv2.GaussianBlur(x, (0, 0), 1.2)
    band = hi - low
    band = np.clip(band / (np.percentile(np.abs(band), 99) + 1e-6) * 110 + 128, 0, 255)
    band[lm == 0] = 0
    return band.astype(np.uint8)


def t_bandpass_zscore(g, lm):
    return t_zscore(t_bandpass(g, lm), lm)


def t_quantile(g, lm, ref=None):
    """Map the lung-field intensity distribution onto a fixed reference shape -
    a simple histogram-matching harmonisation."""
    o = g.astype(np.float32); m = lm > 0
    if m.sum() < 50:
        return g
    v = o[m]
    r = (np.argsort(np.argsort(v)) + 0.5) / len(v)      # empirical CDF ranks
    o[m] = np.clip(r * 255, 0, 255)
    o[~m] = 0
    return o.astype(np.uint8)


TRANSFORMS = {
    "identity (current)": t_identity,
    "lung masked": t_lung_masked,
    "lung + z-score": t_zscore,
    "lung + hist-eq": t_hist_eq,
    "lung + quantile (hist match)": t_quantile,
    "gradient only": t_gradient,
    "band-pass": t_bandpass,
    "band-pass + z-score": t_bandpass_zscore,
}


# --------------------------------------------------------------- descriptors
def features(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    hist = cv2.calcHist([img], [0], None, [32], [0, 256]).ravel()
    hist = hist / max(hist.sum(), 1)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    f = np.fft.fftshift(np.abs(np.fft.fft2(x)))
    c = f.shape[0] // 2
    band = [f[max(0, c - r):c + r, max(0, c - r):c + r].mean() for r in (4, 8, 16, 32)]
    return np.concatenate([
        hist,
        [x.mean(), x.std(), np.percentile(x, 5), np.percentile(x, 95),
         mag.mean(), mag.std(), cv2.Laplacian(x, cv2.CV_32F).var()],
        np.log1p(band),
    ]).astype(np.float32)


def grouped_auc(X, y, groups, seed=SEED):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    aucs = []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        clf = LogisticRegression(max_iter=3000).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else float("nan")


def main() -> int:
    set_seed(SEED)
    rng = np.random.default_rng(SEED)

    iq = list(csv.DictReader(open(SPLITS / "splits.csv", newline="")))
    li = list(csv.DictReader(open(LIDC / "lidc_masks.csv", newline="")))
    rng.shuffle(iq); rng.shuffle(li)

    items = ([(r["path"], 0, f"IQ-{r['group_id']}", None) for r in iq[:N_PER]] +
             [(r["path"], 1, f"LIDC-{r['patient_id']}",
               1 if r["label"] == "malignant" else 0) for r in li[:N_PER]])

    print(f"loading {len(items)} slices ...")
    cache = []
    for i, (path, dom, grp, lab) in enumerate(items, 1):
        try:
            g = preprocess(load_image(path).array, SIZE)
        except Exception:
            continue
        lm = lung_mask(g)
        if (lm > 0).mean() < 0.02:
            continue
        cache.append((g, lm, dom, grp, lab))
        if i % 150 == 0:
            print(f"  ...{i}/{len(items)}", flush=True)
    print(f"usable {len(cache)}")

    dom_y = np.array([c[2] for c in cache])
    dom_g = np.array([c[3] for c in cache])
    sig_idx = [i for i, c in enumerate(cache) if c[4] is not None]
    sig_y = np.array([cache[i][4] for i in sig_idx])
    sig_g = np.array([cache[i][3] for i in sig_idx])

    print(f"\ndomain probe on {len(dom_y)} slices | "
          f"signal probe on {len(sig_y)} LIDC slices "
          f"({int(sig_y.sum())} malignant)\n")
    print(f"{'transform':<30} {'domain AUC':>11} {'signal AUC':>11}   {'verdict'}")
    print("-" * 88)

    results = {}
    for name, fn in TRANSFORMS.items():
        F = np.array([features(fn(g, lm)) for g, lm, _, _, _ in cache])
        d = grouped_auc(F, dom_y, dom_g)
        s = grouped_auc(F[sig_idx], sig_y, sig_g)
        if d < 0.70 and s > 0.60:
            v = "PROMISING - signature suppressed, signal retained"
        elif d < 0.70:
            v = "signature suppressed but signal also lost"
        elif s > 0.65:
            v = "signal kept, signature remains"
        else:
            v = "no improvement"
        print(f"{name:<30} {d:>11.4f} {s:>11.4f}   {v}")
        results[name] = {"domain_auc": round(d, 4), "signal_auc": round(s, 4),
                         "verdict": v}

    base_d = results["identity (current)"]["domain_auc"]
    base_s = results["identity (current)"]["signal_auc"]
    best = min(results.items(), key=lambda kv: kv[1]["domain_auc"])
    print(f"\n  baseline: domain {base_d:.4f} / signal {base_s:.4f}")
    print(f"  lowest domain separability: '{best[0]}' "
          f"domain {best[1]['domain_auc']:.4f} / signal {best[1]['signal_auc']:.4f}")
    print("\n  NOTE: signal AUC here comes from simple appearance descriptors, not a")
    print("  trained CNN. It is a floor, not a ceiling - it shows whether benign/")
    print("  malignant information SURVIVES the transform, not how well a real")
    print("  model would use it.")

    json.dump({
        "question": "Which preprocessing removes dataset signature while keeping signal?",
        "n_slices": len(cache), "n_signal_slices": int(len(sig_y)),
        "baseline": {"domain_auc": base_d, "signal_auc": base_s},
        "results": results,
        "lowest_domain": {"transform": best[0], **best[1]},
        "caveat": ("Signal AUC is measured with hand-crafted descriptors and a linear "
                   "model; it is a lower bound on what a CNN could extract. Domain AUC "
                   "is the quantity being minimised."),
    }, open(REPORTS / "harmonisation_search.json", "w"), indent=1)
    print("\nwrote reports/harmonisation_search.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
