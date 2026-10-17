"""
Phase 6 - Fixing the ROI CNN's overfitting, measured rather than assumed.

THE PROBLEM
    EfficientNet-B3 has 10.7M parameters. The ROI training set has 632 crops from
    35 patients - roughly 17,000 parameters per training image. It reached its
    best validation score at epoch 3 and then memorised. Held-out AUC 0.680,
    beaten outright by a 6-feature linear model at 0.898.

METHOD FIX FIRST
    The previous comparison used a single 8-patient test split. With 8 patients a
    difference of 0.05 AUC is noise. Everything here uses StratifiedGroupKFold
    over ALL 50 patients, so every patient is tested exactly once and the
    estimate rests on ~920 predictions instead of 168. That change alone makes
    the comparison trustworthy.

STRATEGIES COMPARED
    A  baseline            EfficientNet-B3, full fine-tune          (reproduce the failure)
    B  small_backbone      MobileNetV3-Small, 1.5M params
    C  frozen_features     pretrained encoder FROZEN + linear probe (no fine-tuning)
    D  regularised         small backbone + heavy aug + weight decay + early stop
    E  fusion              frozen CNN features CONCATENATED with hand-crafted features
    F  handcrafted_only    10 interpretable features                (the bar to beat)

The honest expectation, given the data scale, is that less capacity wins. That is
the result worth reporting either way.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (DATA, EXPERIMENTS, IMAGENET_MEAN, IMAGENET_STD, REPORTS,
                        SEED, USE_AMP, hardware_summary, set_seed)

LIDC = DATA / "external" / "lidc"
ROI_PX, OUT_PX = 96, 128


# --------------------------------------------------------------- data
def load_all():
    rows = list(csv.DictReader(open(LIDC / "lidc_masks.csv", newline="")))
    imgs, feats, y, g = [], [], [], []
    for r in rows:
        img = cv2.imread(r["path"], cv2.IMREAD_GRAYSCALE)
        m = (cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        if m.sum() < 4:
            continue
        ys, xs = np.nonzero(m)
        cy, cx = int(ys.mean()), int(xs.mean())
        h = ROI_PX // 2
        H, W = img.shape
        cy = int(np.clip(cy, h, H - h)); cx = int(np.clip(cx, h, W - h))
        roi = cv2.resize(img[cy - h:cy + h, cx - h:cx + h], (OUT_PX, OUT_PX))

        area = float(m.sum())
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        per = float(cv2.arcLength(c, True))
        hull = float(cv2.contourArea(cv2.convexHull(c))) or 1.0
        v = img[m > 0].astype(np.float32)
        ring = cv2.dilate(m, np.ones((9, 9), np.uint8)) - m
        vs = img[ring > 0].astype(np.float32)

        imgs.append(roi)
        feats.append([area, 2 * np.sqrt(area / np.pi), float(v.mean()), float(v.std()),
                      float(np.percentile(v, 90)), float(np.percentile(v, 10)), per,
                      float(4 * np.pi * area / max(per ** 2, 1e-6)),
                      float(area / hull),
                      float(v.mean() - vs.mean()) if vs.size else 0.0])
        y.append(1 if r["label"] == "malignant" else 0)
        g.append(r["patient_id"])
    return (np.array(imgs), np.array(feats, np.float32),
            np.array(y), np.array(g))


class RoiArr(Dataset):
    def __init__(self, imgs, y, augment=False, heavy=False, seed=SEED):
        self.imgs, self.y = imgs, y
        self.augment, self.heavy = augment, heavy
        self.rng = np.random.default_rng(seed)
        self.mean = np.array(IMAGENET_MEAN, np.float32)[:, None, None]
        self.std = np.array(IMAGENET_STD, np.float32)[:, None, None]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        r = self.imgs[i]
        if self.augment:
            if self.rng.random() < .5:
                r = np.ascontiguousarray(r[:, ::-1])
            if self.rng.random() < .5:
                r = np.ascontiguousarray(r[::-1, :])
            k = int(self.rng.integers(0, 4))
            if k:
                r = np.ascontiguousarray(np.rot90(r, k))
            if self.heavy:
                # scale jitter + intensity jitter + occlusion
                s = self.rng.uniform(0.85, 1.18)
                M = cv2.getRotationMatrix2D((OUT_PX / 2, OUT_PX / 2),
                                            self.rng.uniform(-25, 25), s)
                r = cv2.warpAffine(r, M, (OUT_PX, OUT_PX), borderValue=0)
                r = np.clip(r.astype(np.float32) * self.rng.uniform(.8, 1.2)
                            + self.rng.uniform(-20, 20), 0, 255)
                if self.rng.random() < .3:
                    r = r + self.rng.normal(0, self.rng.uniform(3, 10), r.shape)
                r = np.clip(r, 0, 255)
                if self.rng.random() < .25:      # cutout
                    cs = int(self.rng.integers(12, 30))
                    y0 = int(self.rng.integers(0, OUT_PX - cs))
                    x0 = int(self.rng.integers(0, OUT_PX - cs))
                    r[y0:y0 + cs, x0:x0 + cs] = 0
                r = r.astype(np.uint8)
        x = r.astype(np.float32) / 255.0
        x = (np.stack([x, x, x], 0) - self.mean) / self.std
        return torch.from_numpy(x), int(self.y[i])


# --------------------------------------------------------------- helpers
def train_cnn(imgs_tr, ytr, imgs_te, backbone, device, dt, *, epochs, lr,
              freeze=False, heavy_aug=False, wd=1e-4, dropout=0.0, patience=None,
              imgs_va=None, yva=None, seed=SEED):
    import timm
    set_seed(seed)
    m = timm.create_model(backbone, pretrained=True, num_classes=2,
                          drop_rate=dropout).to(device)
    if freeze:
        for n_, p in m.named_parameters():
            if not any(k in n_ for k in ("classifier", "fc", "head")):
                p.requires_grad = False

    w = torch.tensor([1.0, max((ytr == 0).sum(), 1) / max((ytr == 1).sum(), 1)],
                     dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    tl = DataLoader(RoiArr(imgs_tr, ytr, True, heavy_aug, seed), batch_size=32,
                    shuffle=True, num_workers=0)

    def score(imgs):
        m.eval(); out = []
        dl = DataLoader(RoiArr(imgs, np.zeros(len(imgs), int)), batch_size=64)
        with torch.no_grad():
            for x, _ in dl:
                with torch.autocast("cuda", dtype=dt or torch.float32, enabled=dt is not None):
                    o = m(x.to(device))
                out.append(torch.softmax(o.float(), 1)[:, 1].cpu().numpy())
        return np.concatenate(out)

    best_state, best_auc, bad = None, -1, 0
    for ep in range(epochs):
        m.train()
        for x, yy in tl:
            x, yy = x.to(device), yy.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dt or torch.float32, enabled=dt is not None):
                o = m(x)
            loss = crit(o.float(), yy)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
        sch.step()
        if patience is not None and imgs_va is not None:
            a = roc_auc_score(yva, score(imgs_va)) if len(np.unique(yva)) > 1 else 0
            if a > best_auc:
                best_auc, bad = a, 0
                best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        m.load_state_dict(best_state)
    return score(imgs_te), m


def frozen_embeddings(imgs, backbone, device, dt, seed=SEED):
    """Pretrained encoder used as a FIXED feature extractor - nothing is trained."""
    import timm
    set_seed(seed)
    m = timm.create_model(backbone, pretrained=True, num_classes=0,
                          global_pool="avg").to(device).eval()
    dl = DataLoader(RoiArr(imgs, np.zeros(len(imgs), int)), batch_size=64)
    out = []
    with torch.no_grad():
        for x, _ in dl:
            with torch.autocast("cuda", dtype=dt or torch.float32, enabled=dt is not None):
                f = m(x.to(device))
            out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    args = ap.parse_args()

    set_seed(SEED)
    hw = hardware_summary()
    device = torch.device(hw["device"])
    dt = (torch.bfloat16 if device.type == "cuda"
          and torch.cuda.is_bf16_supported() and USE_AMP else None)

    imgs, feats, y, g = load_all()
    print(f"{len(y)} ROIs from {len(set(g))} patients "
          f"({int(y.sum())} malignant / {int((1-y).sum())} benign)")
    print(f"grouped {args.folds}-fold CV - every patient tested exactly once\n")

    # Frozen embeddings are deterministic, so compute them once for all folds.
    print("extracting frozen embeddings ...", flush=True)
    emb = frozen_embeddings(imgs, "efficientnet_b3", device, dt)
    print(f"  {emb.shape}\n")

    strategies = ["A_baseline_effnet_b3", "B_small_backbone", "C_frozen_features",
                  "D_regularised", "E_fusion", "F_handcrafted_only"]
    oof = {s: np.zeros(len(y)) for s in strategies}

    sgkf = StratifiedGroupKFold(args.folds, shuffle=True, random_state=SEED)
    t0 = time.time()
    for fold, (tr, te) in enumerate(sgkf.split(imgs, y, groups=g), 1):
        # inner split for early stopping, grouped so it stays honest
        inner = StratifiedGroupKFold(4, shuffle=True, random_state=SEED)
        itr, iva = next(inner.split(imgs[tr], y[tr], groups=g[tr]))
        tr_i, va_i = tr[itr], tr[iva]
        print(f"fold {fold}/{args.folds}: train {len(tr)} / test {len(te)} ROIs "
              f"({len(set(g[te]))} patients)", flush=True)

        oof["A_baseline_effnet_b3"][te], _ = train_cnn(
            imgs[tr], y[tr], imgs[te], "efficientnet_b3", device, dt,
            epochs=args.epochs, lr=3e-4)

        oof["B_small_backbone"][te], _ = train_cnn(
            imgs[tr], y[tr], imgs[te], "mobilenetv3_small_100", device, dt,
            epochs=args.epochs, lr=6e-4)

        # C: frozen encoder + linear probe
        sc = StandardScaler().fit(emb[tr])
        clf = LogisticRegression(max_iter=3000, C=0.05, class_weight="balanced",
                                 random_state=SEED).fit(sc.transform(emb[tr]), y[tr])
        oof["C_frozen_features"][te] = clf.predict_proba(sc.transform(emb[te]))[:, 1]

        oof["D_regularised"][te], _ = train_cnn(
            imgs[tr_i], y[tr_i], imgs[te], "mobilenetv3_small_100", device, dt,
            epochs=args.epochs, lr=6e-4, heavy_aug=True, wd=5e-2, dropout=0.4,
            patience=3, imgs_va=imgs[va_i], yva=y[va_i])

        # E: frozen embedding + hand-crafted features
        Xtr = np.hstack([emb[tr], feats[tr]])
        Xte = np.hstack([emb[te], feats[te]])
        sc2 = StandardScaler().fit(Xtr)
        clf2 = LogisticRegression(max_iter=4000, C=0.05, class_weight="balanced",
                                  random_state=SEED).fit(sc2.transform(Xtr), y[tr])
        oof["E_fusion"][te] = clf2.predict_proba(sc2.transform(Xte))[:, 1]

        # F: hand-crafted only
        sc3 = StandardScaler().fit(feats[tr])
        clf3 = SVC(C=3.0, probability=True, class_weight="balanced",
                   random_state=SEED).fit(sc3.transform(feats[tr]), y[tr])
        oof["F_handcrafted_only"][te] = clf3.predict_proba(sc3.transform(feats[te]))[:, 1]

    mins = (time.time() - t0) / 60

    def ci(s, n=1000):
        rng = np.random.default_rng(SEED); v = []
        for _ in range(n):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) < 2:
                continue
            v.append(roc_auc_score(y[i], s[i]))
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    print(f"\n{'='*80}\nOUT-OF-FOLD RESULTS (all {len(y)} ROIs, {len(set(g))} patients)\n{'='*80}")
    print(f"{'strategy':<26} {'AUC':>8} {'95% CI':>18} {'patient AUC':>12}")
    print("-" * 80)
    results = {}
    up = sorted(set(g))
    for s in strategies:
        a = roc_auc_score(y, oof[s])
        lo, hi = ci(oof[s])
        py = np.array([int(y[g == p].max()) for p in up])
        ps = np.array([float(oof[s][g == p].max()) for p in up])
        pa = roc_auc_score(py, ps) if len(np.unique(py)) > 1 else float("nan")
        print(f"{s:<26} {a:>8.4f} {'[{:.3f}, {:.3f}]'.format(lo, hi):>18} {pa:>12.4f}")
        results[s] = {"roc_auc": round(float(a), 4),
                      "ci95": [round(lo, 4), round(hi, 4)],
                      "patient_auc": round(float(pa), 4)}

    base = results["A_baseline_effnet_b3"]["roc_auc"]
    best = max(results.items(), key=lambda kv: kv[1]["roc_auc"])
    print(f"\n  baseline (EfficientNet-B3 full fine-tune): {base:.4f}")
    print(f"  best strategy: {best[0]} at {best[1]['roc_auc']:.4f} "
          f"({best[1]['roc_auc'] - base:+.4f})")
    print(f"  [{mins:.1f} min]")

    json.dump({
        "problem": ("EfficientNet-B3 (10.7M params) on 632 ROIs from 35 patients "
                    "overfits: best val at epoch 3, held-out AUC 0.680 vs 0.898 for a "
                    "6-feature linear model."),
        "method_note": ("Evaluated with StratifiedGroupKFold over all 50 patients so "
                        "every patient is tested once. The previous single 8-patient "
                        "split was too small to compare strategies."),
        "n_rois": int(len(y)), "n_patients": len(set(g)), "folds": args.folds,
        "results": results,
        "baseline": base,
        "best": {"strategy": best[0], **best[1]},
    }, open(REPORTS / "overfitting_fixes.json", "w"), indent=1)
    json.dump(results, open(EXPERIMENTS / "overfitting_fixes.json", "w"), indent=1)
    print("\nwrote reports/overfitting_fixes.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
