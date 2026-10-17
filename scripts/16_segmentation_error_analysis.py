"""
Phase 14c - what does Dice 0.39 actually mean?

A single Dice number conflates two very different questions:

    DETECTION   - did it find the nodule at all?
    DELINEATION - once found, is the outline accurate?

A model can detect every nodule and still score a poor Dice if its boundaries are
loose, and the clinical value of those two failure modes is completely different.
So this script decomposes the number on the held-out LIDC patients:

  * per-slice detection rate (any predicted blob overlapping the true nodule)
  * centroid distance in pixels and mm when detected
  * Dice measured ONLY over detected cases
  * area ratio (predicted / true) - over- or under-segmentation
  * false-positive blobs per slice
  * a size breakdown, because small nodules are where these models fail

It also renders a qualitative grid so the result can be judged by eye, not just
by a number.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA, MODELS, REPORTS, FIGURES, EXPERIMENTS, SEED, set_seed
from src.models.unet import UNet

LIDC = DATA / "external" / "lidc"
SEG = 512
MIN_BLOB = 12
PIXEL_MM = 0.7          # typical LIDC in-plane spacing; used only for a readable mm figure


def load_split():
    rows = list(csv.DictReader(open(LIDC / "lidc_masks.csv", newline="")))
    rec = json.load(open(EXPERIMENTS / "unet_segmentation.json"))
    n_te = rec["split"]["patients_test"]
    pats = sorted({r["patient_id"] for r in rows})
    rng = np.random.default_rng(SEED)
    rng.shuffle(pats)
    n = len(pats)
    te = set(pats[int(.85 * n):])
    return [r for r in rows if r["patient_id"] in te], rec


def main():
    set_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(MODELS / "unet_nodule.pt", map_location=dev, weights_only=False)
    model = UNet(1, base=ck.get("base", 24)).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    thr = float(ck.get("threshold", 0.5))

    rows, rec = load_split()
    print(f"held-out test slices: {len(rows)} from "
          f"{len({r['patient_id'] for r in rows})} patients   threshold={thr:.2f}\n")

    det, miss = 0, 0
    dices, cdists, area_ratios, fp_counts = [], [], [], []
    by_size = {"small (<40px)": [], "medium (40-120px)": [], "large (>120px)": []}
    panels = []

    for r in rows:
        img = cv2.imread(r["path"], cv2.IMREAD_GRAYSCALE)
        gt = (cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        if img.shape[0] != SEG:
            img = cv2.resize(img, (SEG, SEG), interpolation=cv2.INTER_AREA)
            gt = cv2.resize(gt, (SEG, SEG), interpolation=cv2.INTER_NEAREST)
        if gt.sum() == 0:
            continue

        with torch.no_grad():
            x = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None].to(dev)
            prob = torch.sigmoid(model(x).float())[0, 0].cpu().numpy()
        pred = (prob > thr).astype(np.uint8)

        # drop noise blobs
        n, lab, stats, cents = cv2.connectedComponentsWithStats(pred, 8)
        keep = np.zeros_like(pred)
        blobs = []
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB:
                keep[lab == i] = 1
                blobs.append((i, stats[i, cv2.CC_STAT_AREA], cents[i]))
        pred = keep

        inter = int((pred & gt).sum())
        gt_area = int(gt.sum())
        pr_area = int(pred.sum())
        dice = 2 * inter / max(pr_area + gt_area, 1)

        # detection = any kept blob overlaps the true nodule
        hit = inter > 0
        det += int(hit); miss += int(not hit)

        # false positives = kept blobs with no overlap with GT
        fp = sum(1 for i, a, c in blobs if int((( lab == i) & (gt > 0)).sum()) == 0)
        fp_counts.append(fp)

        bucket = ("small (<40px)" if gt_area < 40 else
                  "medium (40-120px)" if gt_area <= 120 else "large (>120px)")
        by_size[bucket].append(dice)

        if hit:
            dices.append(dice)
            area_ratios.append(pr_area / max(gt_area, 1))
            gy, gx = np.nonzero(gt); py, px = np.nonzero(pred & gt)
            cdists.append(float(np.hypot(px.mean() - gx.mean(), py.mean() - gy.mean())))
            if len(panels) < 6 and gt_area > 30:
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cg, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cp, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, cg, -1, (0, 255, 0), 2)     # green = radiologist
                cv2.drawContours(rgb, cp, -1, (0, 0, 255), 2)     # red   = model
                ys, xs = np.nonzero(gt)
                cy, cx = int(ys.mean()), int(xs.mean())
                s = 110
                y0, x0 = max(0, cy - s), max(0, cx - s)
                crop = rgb[y0:y0 + 2 * s, x0:x0 + 2 * s]
                if crop.shape[0] > 20 and crop.shape[1] > 20:
                    crop = cv2.resize(crop, (240, 240))
                    cv2.putText(crop, f"Dice {dice:.2f}", (8, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    panels.append(crop)

    total = det + miss
    res = {
        "n_slices": total,
        "detection_rate": round(det / max(total, 1), 4),
        "missed_entirely": miss,
        "dice_over_detected_only": round(float(np.mean(dices)), 4) if dices else None,
        "dice_over_all": round(float(np.mean(dices + [0.0] * miss)), 4),
        "median_centroid_error_px": round(float(np.median(cdists)), 2) if cdists else None,
        "median_centroid_error_mm": round(float(np.median(cdists)) * PIXEL_MM, 2) if cdists else None,
        "median_area_ratio_pred_over_true": round(float(np.median(area_ratios)), 3) if area_ratios else None,
        "false_positive_blobs_per_slice": round(float(np.mean(fp_counts)), 3),
        "dice_by_nodule_size": {k: (round(float(np.mean(v)), 4), len(v))
                                for k, v in by_size.items() if v},
    }

    print("=== DETECTION vs DELINEATION ===")
    print(f"  detection rate (found the nodule at all) : {res['detection_rate']:.3f}  "
          f"({det}/{total})")
    print(f"  missed entirely                          : {miss}")
    print(f"  Dice over DETECTED cases only            : {res['dice_over_detected_only']}")
    print(f"  Dice over ALL cases (misses count as 0)  : {res['dice_over_all']}")
    print(f"  median centroid error                    : {res['median_centroid_error_px']} px "
          f"(~{res['median_centroid_error_mm']} mm)")
    print(f"  median predicted/true area ratio         : {res['median_area_ratio_pred_over_true']}")
    print(f"  false-positive blobs per slice           : {res['false_positive_blobs_per_slice']}")
    print("\n=== DICE BY NODULE SIZE ===")
    for k, (d, n_) in res["dice_by_nodule_size"].items():
        print(f"  {k:20s} Dice={d:.4f}  (n={n_})")

    if panels:
        while len(panels) < 6:
            panels.append(np.zeros_like(panels[0]))
        grid = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:6])])
        cv2.imwrite(str(FIGURES / "segmentation_examples.png"), grid)
        print(f"\nwrote figures/segmentation_examples.png "
              f"(green = radiologist consensus, red = model prediction)")

    json.dump(res, open(REPORTS / "segmentation_error_analysis.json", "w"), indent=1)
    print("wrote reports/segmentation_error_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
