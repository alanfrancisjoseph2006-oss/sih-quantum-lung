"""
Phase 2 - Why is external AUC consistently BELOW 0.5?

Script 24 found every arm lands at 0.35-0.46 externally. An AUC below chance in
all four arms is not noise: the model is systematically ranking LIDC malignant
nodules BELOW LIDC benign ones. Something is inverted, and the most likely
culprit is that "malignant" does not mean the same thing in the two datasets.

Hypotheses tested here:

  H1  Nodule SIZE is confounded. If LIDC benign nodules happen to be larger and
      more conspicuous than LIDC malignant ones, and the IQ-OTH-trained model has
      learned "big conspicuous opacity -> malignant", the ranking inverts.

  H2  The model's malignancy score tracks size/conspicuity rather than
      malignancy-specific morphology.

  H3  Inverting the score recovers real signal (i.e. |AUC - 0.5| is meaningful),
      which would indicate a systematic mapping rather than random failure.

This is a label-semantics investigation, not a modelling one. If the tasks are
not the same task, no amount of domain adaptation will align them - and that is
a conclusion worth reaching explicitly rather than by exhaustion.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.config import DATA, MODELS, REPORTS, SEED, set_seed
from src.models.dann import DANNClassifier

LIDC = DATA / "external" / "lidc"


def main() -> int:
    set_seed(SEED)
    rows = list(csv.DictReader(open(LIDC / "lidc_masks.csv", newline="")))

    # ---------- H1: is nodule size confounded with label? ----------
    area = np.array([float(r["mask_area_px"]) for r in rows])
    y = np.array([1 if r["label"] == "malignant" else 0 for r in rows])
    ben, mal = area[y == 0], area[y == 1]

    u, p_size = mannwhitneyu(mal, ben, alternative="two-sided")
    auc_size = roc_auc_score(y, area)

    print("=== H1: nodule size vs label (LIDC) ===")
    print(f"  benign    n={len(ben):4d}  median area {np.median(ben):7.1f} px  "
          f"mean {ben.mean():7.1f}")
    print(f"  malignant n={len(mal):4d}  median area {np.median(mal):7.1f} px  "
          f"mean {mal.mean():7.1f}")
    print(f"  Mann-Whitney p = {p_size:.3g}")
    print(f"  AUC of SIZE ALONE predicting malignant = {auc_size:.4f}")
    if auc_size > 0.60:
        print("  -> larger nodules ARE more often malignant in LIDC")
    elif auc_size < 0.40:
        print("  -> larger nodules are more often BENIGN in LIDC (inverted vs intuition)")
    else:
        print("  -> size carries little label information in LIDC")

    # ---------- H2: does the model score track size? ----------
    ck_path = MODELS / "dr_baseline.pt"
    if not ck_path.exists():
        print("\n(no dr_baseline.pt - run scripts/24 first)")
        return 1
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DANNClassifier(ck["backbone"], n_classes=2,
                           use_dann=ck.get("use_dann", False),
                           pretrained=False).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    # Preprocessing is rebuilt inline: the training script has a numeric
    # module name and is therefore not importable.
    from src.preprocessing.medical_image import load_image, preprocess
    from src.config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
    mean = np.array(IMAGENET_MEAN, np.float32)[:, None, None]
    std = np.array(IMAGENET_STD, np.float32)[:, None, None]

    scores = []
    with torch.no_grad():
        for i in range(0, len(rows), 32):
            batch = rows[i:i + 32]
            xs = []
            for r in batch:
                g = preprocess(load_image(r["path"]).array, IMG_SIZE)
                x = g.astype(np.float32) / 255.0
                xs.append((np.stack([x, x, x], 0) - mean) / std)
            xb = torch.from_numpy(np.stack(xs)).to(device)
            logits, _ = model(xb)
            scores.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
    s = np.concatenate(scores)

    rho, p_rho = spearmanr(s, area)
    auc_model = roc_auc_score(y, s)
    print("\n=== H2: what does the model's malignancy score track? ===")
    print(f"  Spearman(score, nodule area) = {rho:+.4f}  (p = {p_rho:.3g})")
    print(f"  model AUC vs true label      = {auc_model:.4f}")
    print(f"  AUC of size alone            = {auc_size:.4f}")
    if abs(rho) > 0.25:
        print(f"  -> the score is substantially driven by nodule SIZE/conspicuity")
    else:
        print("  -> the score is not strongly size-driven")

    # ---------- H3: is the inversion systematic? ----------
    auc_inv = roc_auc_score(y, -s)
    print("\n=== H3: is the failure systematic or random? ===")
    print(f"  AUC as-is    : {auc_model:.4f}")
    print(f"  AUC inverted : {auc_inv:.4f}")
    if auc_inv > 0.60:
        verdict = ("The inversion is SYSTEMATIC. The model carries real information "
                   "about LIDC, but with the sign flipped - strong evidence that "
                   "'malignant' does not denote the same visual phenotype in the two "
                   "datasets.")
    elif auc_model < 0.45 and auc_inv < 0.60:
        verdict = ("Mildly inverted but close to chance either way. The model carries "
                   "little usable information about LIDC in either direction.")
    else:
        verdict = "No systematic inversion; the failure looks like noise around chance."
    print(f"  -> {verdict}")

    # ---------- size-matched sanity check ----------
    # If size is the confound, comparing within a narrow size band should remove it.
    med = np.median(area)
    band = (area > np.percentile(area, 35)) & (area < np.percentile(area, 65))
    auc_band = (roc_auc_score(y[band], s[band])
                if len(np.unique(y[band])) > 1 else float("nan"))
    print(f"\n=== size-matched subset (35th-65th percentile area, n={int(band.sum())}) ===")
    print(f"  model AUC within a narrow size band = {auc_band:.4f}")
    print("  (if the confound were purely size, this should move toward 0.5)")

    out = {
        "question": "Why is external AUC consistently below 0.5?",
        "h1_size_confound": {
            "benign_median_area_px": float(np.median(ben)),
            "malignant_median_area_px": float(np.median(mal)),
            "mannwhitney_p": float(p_size),
            "auc_of_size_alone": round(float(auc_size), 4),
        },
        "h2_score_tracks_size": {
            "spearman_score_vs_area": round(float(rho), 4),
            "p_value": float(p_rho),
            "model_auc": round(float(auc_model), 4),
        },
        "h3_systematic_inversion": {
            "auc_as_is": round(float(auc_model), 4),
            "auc_inverted": round(float(auc_inv), 4),
            "verdict": verdict,
        },
        "size_matched_auc": (round(float(auc_band), 4)
                             if not np.isnan(auc_band) else None),
        "interpretation_warning": (
            "Inverting a model's score to improve a metric is NOT a fix. It would be "
            "fitting to the external test set, and the sign could differ on the next "
            "dataset. This analysis is diagnostic only; no inverted model is deployed."),
    }
    json.dump(out, open(REPORTS / "inversion_analysis.json", "w"), indent=1)
    print("\nwrote reports/inversion_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
