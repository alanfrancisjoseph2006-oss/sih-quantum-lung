"""
Phase 4/6/9 - Nodule-centred ROI classification.

WHY THIS, AND WHY NOW
---------------------
Chain of measured findings that led here:

  D26          whole-image classifier: in-domain AUC 1.000, external 0.488
  script 22    dataset identity recoverable at AUC 1.000 from a whole slice,
               and still 0.937 from a RANDOM lung crop -> cropping alone is not
               the fix
  script 23    best harmonisation only reaches domain AUC 0.857
  script 24    harmonisation and DANN both made external performance WORSE
  script 25    **nodule SIZE alone predicts LIDC malignancy at AUC 0.873, and the
               trained model's score correlates with size at rho = +0.05
               (p = 0.13) - it is not looking at the nodule at all**

So the problem is not that the task is impossible. A trivial, clinically real
feature achieves 0.873. The problem is that whole-image training never forced
the network to attend to the lesion.

This script tests the direct fix: crop to the annotated nodule and classify THAT.

HONEST FRAMING
    LIDC has real contours, so ROIs here are ground-truth nodule crops, and the
    split is by REAL patient ID. This measures whether nodule-focused learning
    works. It does NOT yet demonstrate cross-dataset generalisation, because
    IQ-OTH has no nodule annotations and so cannot supply comparable ROIs. That
    limitation is reported, not papered over.

BASELINES (a model must beat these to be worth anything)
    1. size only        - a single scalar, logistic regression
    2. size + intensity - simple hand-crafted descriptors
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.config import (DATA, EXPERIMENTS, IMAGENET_MEAN, IMAGENET_STD, MODELS,
                        REPORTS, SEED, USE_AMP, hardware_summary, set_seed)

LIDC = DATA / "external" / "lidc"
ROI_PX = 96          # generous enough for large masses, tight enough to exclude anatomy
OUT_PX = 128


# ------------------------------------------------------------------- ROI
def nodule_roi(img: np.ndarray, mask: np.ndarray, jitter_rng=None) -> np.ndarray | None:
    """
    Crop a fixed-size window centred on the nodule.

    The window is a FIXED pixel size rather than scaled to the lesion, so the
    network can still see how large the nodule is relative to the frame. Scaling
    each lesion to fill the crop would throw away size - the single most
    predictive feature available (AUC 0.873 on its own).
    """
    ys, xs = np.nonzero(mask > 127)
    if len(ys) == 0:
        return None
    cy, cx = int(ys.mean()), int(xs.mean())
    if jitter_rng is not None:
        cy += int(jitter_rng.integers(-8, 9))
        cx += int(jitter_rng.integers(-8, 9))
    h = ROI_PX // 2
    H, W = img.shape[:2]
    cy = int(np.clip(cy, h, H - h)); cx = int(np.clip(cx, h, W - h))
    roi = img[cy - h:cy + h, cx - h:cx + h]
    if roi.shape[0] != ROI_PX or roi.shape[1] != ROI_PX:
        return None
    return cv2.resize(roi, (OUT_PX, OUT_PX), interpolation=cv2.INTER_CUBIC)


class RoiDS(Dataset):
    def __init__(self, rows, augment=False, seed=SEED):
        self.rows, self.augment = rows, augment
        self.rng = np.random.default_rng(seed)
        self.cache: dict[int, np.ndarray] = {}
        self.mean = np.array(IMAGENET_MEAN, np.float32)[:, None, None]
        self.std = np.array(IMAGENET_STD, np.float32)[:, None, None]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        key = i
        if key not in self.cache:
            img = cv2.imread(r["path"], cv2.IMREAD_GRAYSCALE)
            msk = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE)
            roi = nodule_roi(img, msk)
            self.cache[key] = roi if roi is not None else np.zeros((OUT_PX, OUT_PX), np.uint8)
        roi = self.cache[key]

        if self.augment:
            if self.rng.random() < 0.5:
                roi = np.ascontiguousarray(roi[:, ::-1])
            if self.rng.random() < 0.5:
                roi = np.ascontiguousarray(roi[::-1, :])
            k = int(self.rng.integers(0, 4))
            if k:
                roi = np.ascontiguousarray(np.rot90(roi, k))
            roi = np.clip(roi.astype(np.float32) * self.rng.uniform(.9, 1.1)
                          + self.rng.uniform(-10, 10), 0, 255).astype(np.uint8)

        x = roi.astype(np.float32) / 255.0
        x = (np.stack([x, x, x], 0) - self.mean) / self.std
        y = 1 if r["label"] == "malignant" else 0
        return torch.from_numpy(x), y


# --------------------------------------------------------------- metrics
def boot_ci(y, s, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        v.append(roc_auc_score(y[i], s[i]))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)


def report(name, y, s, groups=None):
    lo, hi = boot_ci(y, s)
    out = {"name": name, "n": int(len(y)),
           "roc_auc": round(float(roc_auc_score(y, s)), 4),
           "roc_auc_ci95": [round(lo, 4), round(hi, 4)],
           "pr_auc": round(float(average_precision_score(y, s)), 4)}
    pred = (s >= 0.5).astype(int) if s.max() <= 1.0 else (s >= np.median(s)).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    out.update(sensitivity=round(tp / max(tp + fn, 1), 4),
               specificity=round(tn / max(tn + fp, 1), 4))
    if groups is not None:
        up = sorted(set(groups))
        py = np.array([int(y[groups == g].max()) for g in up])
        ps = np.array([float(s[groups == g].max()) for g in up])
        if len(np.unique(py)) > 1:
            plo, phi = boot_ci(py, ps)
            out["patient_level"] = {"n": len(up),
                                    "roc_auc": round(float(roc_auc_score(py, ps)), 4),
                                    "roc_auc_ci95": [round(plo, 4), round(phi, 4)]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="efficientnet_b3")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    set_seed(SEED)
    hw = hardware_summary()
    device = torch.device(hw["device"])
    dt = (torch.bfloat16 if device.type == "cuda"
          and torch.cuda.is_bf16_supported() and USE_AMP else None)

    rows = list(csv.DictReader(open(LIDC / "lidc_masks.csv", newline="")))
    pats = sorted({r["patient_id"] for r in rows})
    rng = np.random.default_rng(SEED)
    rng.shuffle(pats)
    n = len(pats)
    tr_p = set(pats[:int(.70 * n)]); va_p = set(pats[int(.70 * n):int(.85 * n)])
    te_p = set(pats[int(.85 * n):])
    tr = [r for r in rows if r["patient_id"] in tr_p]
    va = [r for r in rows if r["patient_id"] in va_p]
    te = [r for r in rows if r["patient_id"] in te_p]
    assert not (tr_p & va_p) and not (tr_p & te_p) and not (va_p & te_p)

    print(f"PATIENT-LEVEL split on real LIDC IDs: "
          f"{len(tr_p)}/{len(va_p)}/{len(te_p)} patients, "
          f"{len(tr)}/{len(va)}/{len(te)} ROIs")
    for nm, sp in (("train", tr), ("val", va), ("test", te)):
        m = sum(1 for r in sp if r["label"] == "malignant")
        print(f"  {nm:5s} malignant {m:4d} / benign {len(sp)-m:4d}")

    # -------------------- baselines --------------------
    def handcrafted(sp):
        X, y = [], []
        for r in sp:
            img = cv2.imread(r["path"], cv2.IMREAD_GRAYSCALE)
            msk = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE) > 127
            a = float(msk.sum())
            v = img[msk] if msk.sum() else np.array([0.0])
            X.append([a, np.sqrt(a / np.pi) * 2, float(v.mean()), float(v.std()),
                      float(np.percentile(v, 90)), float(np.percentile(v, 10))])
            y.append(1 if r["label"] == "malignant" else 0)
        return np.array(X, np.float32), np.array(y)

    Xtr, ytr = handcrafted(tr); Xte, yte = handcrafted(te)
    gte = np.array([r["patient_id"] for r in te])

    size_only = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        Xtr[:, :1], ytr)
    r_size = report("size only (1 feature)", yte,
                    size_only.predict_proba(Xte[:, :1])[:, 1], gte)

    sc = StandardScaler().fit(Xtr)
    hc = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    r_hc = report("size + intensity (6 features)", yte,
                  hc.predict_proba(sc.transform(Xte))[:, 1], gte)

    # -------------------- ROI CNN --------------------
    import timm
    model = timm.create_model(args.backbone, pretrained=True, num_classes=2).to(device)
    dl = lambda d, sh: DataLoader(d, batch_size=args.batch_size, shuffle=sh, num_workers=0)
    tl = dl(RoiDS(tr, augment=True), True)
    vl = dl(RoiDS(va), False)
    tel = dl(RoiDS(te), False)

    nm_ = sum(1 for r in tr if r["label"] == "malignant")
    w = torch.tensor([1.0, max(len(tr) - nm_, 1) / max(nm_, 1)],
                     dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    @torch.no_grad()
    def sc_(loader):
        model.eval(); S, Y = [], []
        for x, y in loader:
            with torch.autocast("cuda", dtype=dt or torch.float32, enabled=dt is not None):
                o = model(x.to(device))
            S.append(torch.softmax(o.float(), 1)[:, 1].cpu().numpy()); Y.append(y.numpy())
        return np.concatenate(Y), np.concatenate(S)

    best = {"auc": -1, "state": None, "ep": 0}
    t0 = time.time()
    print(f"\ntraining ROI CNN ({args.backbone}, {ROI_PX}px crop -> {OUT_PX}px)")
    for ep in range(args.epochs):
        model.train(); tot = nb = 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dt or torch.float32, enabled=dt is not None):
                o = model(x)
            loss = crit(o.float(), y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); tot += float(loss); nb += 1
        sched.step()
        yv, sv = sc_(vl)
        a = roc_auc_score(yv, sv) if len(np.unique(yv)) > 1 else 0.0
        if a > best["auc"]:
            best = {"auc": a, "ep": ep + 1,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep{ep+1:02d} loss={tot/max(nb,1):.4f} val AUC={a:.4f}", flush=True)

    model.load_state_dict(best["state"])
    yte_, ste_ = sc_(tel)
    r_cnn = report(f"ROI CNN ({args.backbone})", yte_, ste_, gte)
    mins = (time.time() - t0) / 60

    print(f"\n{'='*84}\nHELD-OUT LIDC PATIENTS (n={len(te_p)} patients, {len(te)} ROIs)\n{'='*84}")
    print(f"{'model':<34} {'AUC':>8} {'95% CI':>18} {'sens':>7} {'spec':>7} {'patient AUC':>12}")
    print("-" * 84)
    for r in (r_size, r_hc, r_cnn):
        pa = r.get("patient_level", {}).get("roc_auc")
        lo, hi = r["roc_auc_ci95"]
        ci = "[{:.3f}, {:.3f}]".format(lo, hi)
        pas = "{:.4f}".format(pa) if pa is not None else "-"
        print(f"{r['name']:<34} {r['roc_auc']:>8.4f} {ci:>18} "
              f"{r['sensitivity']:>7.3f} {r['specificity']:>7.3f} {pas:>12}")

    beats = r_cnn["roc_auc"] > r_size["roc_auc"]
    print(f"\n  ROI CNN vs size-only baseline: "
          f"{r_cnn['roc_auc']:.4f} vs {r_size['roc_auc']:.4f} "
          f"({'beats' if beats else 'DOES NOT beat'} it)")
    print(f"  reference - whole-image model on the SAME LIDC data: 0.4566 (script 25)")
    print(f"  [{mins:.1f} min, best epoch {best['ep']}]")

    torch.save({"state_dict": model.state_dict(), "backbone": args.backbone,
                "roi_px": ROI_PX, "out_px": OUT_PX}, MODELS / "roi_nodule_cnn.pt")

    out = {
        "task": "benign vs malignant on nodule-centred ROIs",
        "dataset": "LIDC-IDRI only (has ground-truth contours)",
        "split": {"policy": "patient-level on real LIDC patient IDs",
                  "patients": {"train": len(tr_p), "val": len(va_p), "test": len(te_p)},
                  "rois": {"train": len(tr), "val": len(va), "test": len(te)}},
        "roi": {"crop_px": ROI_PX, "resized_to": OUT_PX,
                "note": ("fixed-size crop so relative lesion size is preserved - "
                         "size alone predicts malignancy at AUC 0.873")},
        "results": {"size_only": r_size, "handcrafted": r_hc, "roi_cnn": r_cnn},
        "whole_image_reference_on_same_data": 0.4566,
        "limitation": (
            "This demonstrates nodule-focused learning WITHIN LIDC under a "
            "patient-level split. It does NOT demonstrate cross-dataset "
            "generalisation: IQ-OTH has no nodule annotations, so comparable ROIs "
            "cannot be built for it. Cross-dataset ROI validation requires either "
            "IQ-OTH annotations or a detector accurate enough to supply them."),
        "train_minutes": round(mins, 1),
    }
    json.dump(out, open(REPORTS / "roi_classifier.json", "w"), indent=1)
    json.dump(out, open(EXPERIMENTS / "roi_classifier.json", "w"), indent=1)
    print("\nwrote reports/roi_classifier.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
