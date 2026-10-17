"""
Phase 14b - train the nodule segmentation U-Net and test whether it transfers.

SPLIT
  Patient-level on LIDC's REAL patient IDs (50 patients -> 35/7/8). No slice from
  a patient appears in two splits.

WHAT IS MEASURED
  1. In-domain (LIDC held-out patients): real Dice and IoU against radiologist
     consensus contours. These are the numbers we are entitled to quote.

  2. Cross-domain (IQ-OTH): there are NO masks, so Dice is impossible and we do
     not invent it. Instead we use a label-grounded proxy that the data DOES
     support: how often the segmenter fires on images labelled `normal` versus
     `malignant`. If it transfers meaningfully it should fire far more on
     malignant than on normal. If it fires equally on both it is producing
     decoration, and we will say so.

     This matters because scripts/13 measured that LIDC->IQ-OTH transfer FAILED
     for classification. There is no reason to assume segmentation transfers
     just because we would like it to.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (DATA, MODELS, REPORTS, EXPERIMENTS, SEED, set_seed,
                        hardware_summary, USE_AMP)
from src.models.unet import UNet, DiceBCELoss, dice_score, iou_score
from src.data.dataset import read_splits
from src.preprocessing.medical_image import load_image, preprocess, lung_mask

LIDC = DATA / "external" / "lidc"
SEG_SIZE = 512          # native LIDC resolution - keeps tiny nodules resolvable


class NoduleSeg(Dataset):
    def __init__(self, rows, augment=False, size=SEG_SIZE, seed=SEED):
        self.rows, self.augment, self.size = rows, augment, size
        self.rng = np.random.default_rng(seed)
        self.cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if i not in self.cache:
            r = self.rows[i]
            img = cv2.imread(r["path"], cv2.IMREAD_GRAYSCALE)
            msk = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE)
            if img.shape[0] != self.size:
                img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
                msk = cv2.resize(msk, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
            self.cache[i] = (img, (msk > 127).astype(np.uint8))
        img, msk = self.cache[i]

        if self.augment:
            if self.rng.random() < 0.5:
                img, msk = img[:, ::-1].copy(), msk[:, ::-1].copy()
            ang = self.rng.uniform(-10, 10)
            tx, ty = self.rng.uniform(-.04, .04, 2) * self.size
            M = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), ang,
                                        self.rng.uniform(0.95, 1.05))
            M[0, 2] += tx; M[1, 2] += ty
            img = cv2.warpAffine(img, M, (self.size, self.size), flags=cv2.INTER_LINEAR)
            msk = cv2.warpAffine(msk, M, (self.size, self.size), flags=cv2.INTER_NEAREST)
            img = np.clip(img.astype(np.float32) * self.rng.uniform(.92, 1.08)
                          + self.rng.uniform(-10, 10), 0, 255).astype(np.uint8)

        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        y = torch.from_numpy(msk.astype(np.float32)).unsqueeze(0)
        return x, y


def split_by_patient(rows, seed=SEED):
    pats = sorted({r["patient_id"] for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(pats)
    n = len(pats)
    tr, va = pats[:int(.70 * n)], pats[int(.70 * n):int(.85 * n)]
    te = pats[int(.85 * n):]
    pick = lambda ps: [r for r in rows if r["patient_id"] in set(ps)]
    return pick(tr), pick(va), pick(te), (tr, va, te)


@torch.no_grad()
def evaluate(model, loader, device, thr=0.5):
    model.eval()
    D, I = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        p = torch.sigmoid(model(x).float())
        D.append(dice_score(p, y, thr).cpu().numpy())
        I.append(iou_score(p, y, thr).cpu().numpy())
    D, I = np.concatenate(D), np.concatenate(I)
    return float(np.nanmean(D)), float(np.nanmean(I))


@torch.no_grad()
def transfer_probe(model, device, thr=0.5, min_area=12):
    """
    Cross-domain firing rate on IQ-OTH, which has NO masks.

    Reports how often the segmenter produces a lung-field detection on images
    labelled normal vs benign vs malignant. Detections outside the lung field
    are discarded as anatomically implausible before counting.
    """
    model.eval()
    out = defaultdict(lambda: {"n": 0, "fired": 0, "areas": []})
    for split in ("val", "test"):
        for r in read_splits(split):
            g = load_image(r["path"]).array
            img = preprocess(g, SEG_SIZE, use_clahe=True)
            lm = lung_mask(img)
            x = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None].to(device)
            p = torch.sigmoid(model(x).float())[0, 0].cpu().numpy()
            det = ((p > thr) & (lm > 0)).astype(np.uint8)
            n, _, stats, _ = cv2.connectedComponentsWithStats(det, 8)
            blobs = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n)
                     if stats[i, cv2.CC_STAT_AREA] >= min_area]
            d = out[r["label"]]
            d["n"] += 1
            if blobs:
                d["fired"] += 1
                d["areas"].append(float(sum(blobs)) / det.size)
    res = {}
    for lab, d in out.items():
        res[lab] = {"n": d["n"], "fired": d["fired"],
                    "fire_rate": round(d["fired"] / max(d["n"], 1), 4),
                    "mean_area_frac_when_fired":
                        round(float(np.mean(d["areas"])), 6) if d["areas"] else 0.0}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=24)
    args = ap.parse_args()

    set_seed(SEED)
    hw = hardware_summary()
    device = torch.device(hw["device"])
    print(json.dumps(hw, indent=1))

    rows = list(csv.DictReader(open(LIDC / "lidc_masks.csv", newline="")))
    tr, va, te, (ptr, pva, pte) = split_by_patient(rows)
    print(f"\nslices  train={len(tr)} val={len(va)} test={len(te)}")
    print(f"patients train={len(ptr)} val={len(pva)} test={len(pte)}  (no overlap by construction)")
    assert not (set(ptr) & set(pva)) and not (set(ptr) & set(pte)) and not (set(pva) & set(pte))

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=0, pin_memory=True)
    tl = dl(NoduleSeg(tr, augment=True), True)
    vl = dl(NoduleSeg(va), False)
    tel = dl(NoduleSeg(te), False)

    model = UNet(1, base=args.base).to(device)
    print(f"U-Net params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    crit = DiceBCELoss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    amp_dtype = (torch.bfloat16 if device.type == "cuda"
                 and torch.cuda.is_bf16_supported() and USE_AMP else None)

    best = {"dice": -1, "state": None, "epoch": 0}
    hist, t0 = [], time.time()
    for ep in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype or torch.float32,
                                enabled=amp_dtype is not None):
                logits = model(x)
            loss = crit(logits.float(), y)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss); nb += 1
        sched.step()
        vd, vi = evaluate(model, vl, device)
        hist.append({"epoch": ep + 1, "loss": round(tot / max(nb, 1), 4),
                     "val_dice": round(vd, 4), "val_iou": round(vi, 4)})
        star = ""
        if vd > best["dice"]:
            best = {"dice": vd, "epoch": ep + 1,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            star = "  <-- best"
        if (ep + 1) % 5 == 0 or star:
            print(f"ep{ep+1:03d} loss={tot/max(nb,1):.4f} valDice={vd:.4f} "
                  f"valIoU={vi:.4f}{star}", flush=True)

    model.load_state_dict(best["state"])
    train_s = time.time() - t0
    print(f"\nrestored best (val Dice={best['dice']:.4f} @ epoch {best['epoch']}) "
          f"in {train_s/60:.1f} min")

    # threshold tuned on VALIDATION only
    best_thr, best_vd = 0.5, -1
    for thr in np.arange(0.1, 0.91, 0.05):
        d, _ = evaluate(model, vl, device, float(thr))
        if d > best_vd:
            best_vd, best_thr = d, float(thr)
    print(f"threshold tuned on validation: {best_thr:.2f} (val Dice {best_vd:.4f})")

    td, ti = evaluate(model, tel, device, best_thr)
    print(f"\n=== IN-DOMAIN (LIDC held-out patients, n={len(te)} slices / {len(pte)} patients) ===")
    print(f"  Dice = {td:.4f}   IoU = {ti:.4f}")

    print(f"\n=== CROSS-DOMAIN TRANSFER PROBE (IQ-OTH, no masks exist) ===")
    probe = transfer_probe(model, device, best_thr)
    for lab in ("normal", "benign", "malignant"):
        if lab in probe:
            p = probe[lab]
            print(f"  {lab:10s} fires on {p['fired']:3d}/{p['n']:3d} = {p['fire_rate']:.3f} "
                  f"| mean area when fired {p['mean_area_frac_when_fired']*100:.3f}%")
    nr = probe.get("normal", {}).get("fire_rate", 0.0)
    mr = probe.get("malignant", {}).get("fire_rate", 0.0)
    ratio = mr / max(nr, 1e-6)
    print(f"  malignant:normal firing ratio = {ratio:.2f}x")
    if ratio < 1.3:
        verdict = ("Fires almost equally on normal and malignant IQ-OTH images. The "
                   "segmenter does NOT transfer to this domain; its output there is "
                   "not evidence of a nodule.")
    elif ratio < 2.0:
        verdict = ("Fires somewhat more on malignant than normal, but the margin is "
                   "weak. Transfer is partial and the output must be treated as a "
                   "suggestion only.")
    else:
        verdict = ("Fires substantially more on malignant than normal images, "
                   "indicating the segmenter carries real signal across domains.")
    print(f"  -> {verdict}")

    torch.save({"state_dict": model.state_dict(), "base": args.base,
                "size": SEG_SIZE, "threshold": best_thr,
                "val_dice": best["dice"], "test_dice": td, "test_iou": ti},
               MODELS / "unet_nodule.pt")
    torch.save({"state_dict": model.state_dict(), "base": args.base},
               MODELS / "unet_nodule_backup.pt")

    rec = {"experiment": "unet_nodule_segmentation",
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED,
           "hardware": hw, "train_time_min": round(train_s / 60, 2),
           "architecture": f"UNet(base={args.base}, depth=4)",
           "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
           "loss": "Dice + 0.5*BCE(pos_weight=20)",
           "input_size": SEG_SIZE,
           "hyperparameters": {"epochs": args.epochs, "batch_size": args.batch_size,
                               "lr": args.lr, "amp_dtype": str(amp_dtype)},
           "split": {"patients_train": len(ptr), "patients_val": len(pva),
                     "patients_test": len(pte), "slices_train": len(tr),
                     "slices_val": len(va), "slices_test": len(te),
                     "policy": "patient-level on real LIDC patient IDs"},
           "threshold_tuned_on_validation": best_thr,
           "in_domain_test": {"dice": round(td, 4), "iou": round(ti, 4),
                              "n_slices": len(te), "n_patients": len(pte)},
           "cross_domain_probe": probe,
           "malignant_to_normal_firing_ratio": round(ratio, 3),
           "transfer_verdict": verdict,
           "history": hist}
    json.dump(rec, open(EXPERIMENTS / "unet_segmentation.json", "w"), indent=1, default=str)
    print(f"\nsaved models/unet_nodule.pt and experiments/unet_segmentation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
