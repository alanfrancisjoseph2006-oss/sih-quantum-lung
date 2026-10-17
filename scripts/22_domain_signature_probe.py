"""
Phase 1/11 - How much DATASET SIGNATURE lives in the image, and where?

D26 measured that the whole-image classifier is at chance externally (AUC 0.488)
while scoring 1.000 in-domain. The hypothesis is that it learned dataset
appearance rather than nodule morphology.

This script tests that hypothesis directly and cheaply, WITHOUT training the
clinical model at all. It trains a *domain* classifier - "which dataset did this
come from?" - on three different views of the same images:

    A. whole slice                (what the current model sees)
    B. lung-field masked slice    (background and body wall removed)
    C. small lung-interior ROI    (a nodule-sized crop)

If dataset identity is trivially recoverable from A but much harder from C, then
the dataset signature lives in the parts of the image the ROI pipeline discards,
and Phase 4 is the right fix. If C is just as separable, the signature is inside
the lung tissue itself and cropping will not save us - which would be worth
knowing before building the whole pipeline.

Evaluation is GROUPED by patient (LIDC) / pseudo-patient (IQ-OTH) so the probe
cannot win by memorising individual scans.
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
ROI = 64          # nodule-scale crop
N_PER_DATASET = 260


def views(path: str, rng: np.random.Generator) -> dict[str, np.ndarray] | None:
    """Return the three views of one slice, or None if no lung field is found."""
    try:
        g = preprocess(load_image(path).array, SIZE)
    except Exception:
        return None

    lm = lung_mask(g)
    if (lm > 0).mean() < 0.02:
        return None

    masked = g.copy()
    masked[lm == 0] = 0

    # ROI: a nodule-sized crop centred on a random point inside the lung field,
    # so the crop contains lung parenchyma and nothing else.
    ys, xs = np.nonzero(lm)
    i = rng.integers(0, len(ys))
    cy, cx = int(ys[i]), int(xs[i])
    h = ROI // 2
    cy = int(np.clip(cy, h, SIZE - h)); cx = int(np.clip(cx, h, SIZE - h))
    roi = g[cy - h:cy + h, cx - h:cx + h]

    return {"whole": g, "lung_masked": masked, "roi": roi}


def features(img: np.ndarray) -> np.ndarray:
    """
    Deliberately simple appearance descriptors - histogram, gradient statistics
    and a coarse frequency summary. If a linear model on THESE can name the
    dataset, the signature is blatant and a deep network will find it instantly.
    """
    x = img.astype(np.float32)
    hist = cv2.calcHist([img], [0], None, [32], [0, 256]).ravel()
    hist = hist / max(hist.sum(), 1)
    gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    lap = cv2.Laplacian(x, cv2.CV_32F)
    f = np.fft.fftshift(np.abs(np.fft.fft2(x)))
    c = f.shape[0] // 2
    band = [f[max(0, c - r):c + r, max(0, c - r):c + r].mean()
            for r in (4, 8, 16, 32)]
    return np.concatenate([
        hist,
        [x.mean(), x.std(), np.percentile(x, 5), np.percentile(x, 95),
         mag.mean(), mag.std(), lap.var(),
         float((x < 10).mean()), float((x > 245).mean())],
        np.log1p(band),
    ]).astype(np.float32)


def main() -> int:
    set_seed(SEED)
    rng = np.random.default_rng(SEED)

    iq = list(csv.DictReader(open(SPLITS / "splits.csv", newline="")))
    li = list(csv.DictReader(open(LIDC / "lidc_masks.csv", newline="")))
    rng.shuffle(iq); rng.shuffle(li)

    samples = []
    for r in iq[:N_PER_DATASET]:
        samples.append((r["path"], 0, f"IQ-{r['group_id']}"))
    for r in li[:N_PER_DATASET]:
        samples.append((r["path"], 1, f"LIDC-{r['patient_id']}"))

    print(f"probing {len(samples)} slices "
          f"({sum(1 for s in samples if s[1]==0)} IQ-OTH, "
          f"{sum(1 for s in samples if s[1]==1)} LIDC)")

    data = {k: [] for k in ("whole", "lung_masked", "roi")}
    y, groups = [], []
    for i, (path, dom, grp) in enumerate(samples, 1):
        v = views(path, rng)
        if v is None:
            continue
        for k in data:
            data[k].append(features(v[k]))
        y.append(dom); groups.append(grp)
        if i % 150 == 0:
            print(f"  ...{i}/{len(samples)}", flush=True)

    y = np.array(y); groups = np.array(groups)
    print(f"usable: {len(y)}  ({int((y==0).sum())} IQ-OTH / {int((y==1).sum())} LIDC)")

    print("\n=== CAN A LINEAR MODEL NAME THE DATASET? (grouped 5-fold) ===")
    print(f"{'view':<16} {'domain ROC-AUC':>15}   interpretation")
    print("-" * 76)
    results = {}
    for name in ("whole", "lung_masked", "roi"):
        X = np.array(data[name])
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        aucs = []
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        for tr, te in sgkf.split(X, y, groups=groups):
            if len(np.unique(y[te])) < 2:
                continue
            clf = LogisticRegression(max_iter=3000, C=1.0).fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        auc = float(np.mean(aucs))
        sd = float(np.std(aucs))
        verdict = ("dataset identity is trivially recoverable" if auc > 0.95 else
                   "strong dataset signature" if auc > 0.85 else
                   "moderate signature" if auc > 0.70 else
                   "weak signature" if auc > 0.60 else
                   "little or no dataset signature")
        print(f"{name:<16} {auc:>10.4f} ±{sd:.3f}   {verdict}")
        results[name] = {"domain_auc": round(auc, 4), "std": round(sd, 4),
                         "verdict": verdict}

    drop = results["whole"]["domain_auc"] - results["roi"]["domain_auc"]
    print(f"\n  whole -> ROI change in dataset separability: {drop:+.4f}")
    if results["roi"]["domain_auc"] < 0.75 and drop > 0.15:
        concl = ("Cropping to a lung-interior ROI substantially removes the dataset "
                 "signature. The signature lives largely OUTSIDE the lung field, so an "
                 "ROI-based pipeline is a well-founded fix.")
    elif results["roi"]["domain_auc"] > 0.90:
        concl = ("The dataset signature survives cropping - it is present in the lung "
                 "parenchyma itself (reconstruction kernel, noise texture, compression). "
                 "ROI cropping ALONE will not fix generalisation; intensity "
                 "harmonisation and/or domain-adversarial training are required.")
    else:
        concl = ("Cropping reduces but does not remove the dataset signature. ROI helps "
                 "and should be combined with harmonisation and domain adaptation.")
    print(f"\n  CONCLUSION: {concl}")

    out = {
        "question": "How much dataset signature is recoverable from each view?",
        "method": ("Logistic regression on simple appearance descriptors (histogram, "
                   "gradient statistics, frequency bands), evaluated with "
                   "StratifiedGroupKFold so the probe cannot memorise individual "
                   "patients."),
        "n_slices": int(len(y)),
        "n_iqoth": int((y == 0).sum()), "n_lidc": int((y == 1).sum()),
        "roi_size_px": ROI, "analysis_size_px": SIZE,
        "results": results,
        "whole_minus_roi": round(drop, 4),
        "conclusion": concl,
        "caveat": ("A high domain AUC does not by itself prove the clinical model uses "
                   "the signature - only that it is available to be used. Combined with "
                   "D26 (external AUC 0.488) the inference is well supported."),
    }
    json.dump(out, open(REPORTS / "domain_signature_probe.json", "w"), indent=1)
    print("\nwrote reports/domain_signature_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
