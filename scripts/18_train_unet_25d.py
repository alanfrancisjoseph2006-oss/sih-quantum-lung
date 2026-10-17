"""
Phase 15b - retrain the segmenter on 2.5D stacks and measure the detection gain.

Everything is held identical to the 2D run except the input: same patients in
each split, same seed, same architecture width, same loss, same schedule. Only
the number of input channels changes (1 -> 3, being slices z-1 / z / z+1).

That isolation matters: any change in detection rate is attributable to the
extra slice context and not to a lucky reshuffle or a different recipe.

Reports the same decomposition as D23 - detection rate, Dice-when-detected and
centroid error - so the two runs are directly comparable.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (DATA, MODELS, REPORTS, FIGURES, EXPERIMENTS, SEED,
                        set_seed, hardware_summary, USE_AMP)
from src.models.unet import UNet, DiceBCELoss, dice_score, iou_score

LIDC = DATA / "external" / "lidc"
SEG = 512
MIN_BLOB = 12


class Stack25D(Dataset):
    def __init__(self, rows, augment=False, seed=SEED):
        self.rows, self.augment = rows, augment
        self.rng = np.random.default_rng(seed)
        self.cache = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if i not in self.cache:
            r = self.rows[i]
            st = cv2.imread(r["stack_path"], cv2.IMREAD_COLOR)     # HxWx3
            mk = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE)
            if st.shape[0] != SEG:
                st = cv2.resize(st, (SEG, SEG), interpolation=cv2.INTER_AREA)
                mk = cv2.resize(mk, (SEG, SEG), interpolation=cv2.INTER_NEAREST)
            self.cache[i] = (st, (mk > 127).astype(np.uint8))
        st, mk = self.cache[i]

        if self.augment:
            if self.rng.random() < 0.5:
                st, mk = st[:, ::-1].copy(), mk[:, ::-1].copy()
            ang = self.rng.uniform(-10, 10)
            tx, ty = self.rng.uniform(-.04, .04, 2) * SEG
            M = cv2.getRotationMatrix2D((SEG / 2, SEG / 2), ang,
                                        self.rng.uniform(0.95, 1.05))
            M[0, 2] += tx; M[1, 2] += ty
            st = cv2.warpAffine(st, M, (SEG, SEG), flags=cv2.INTER_LINEAR)
            mk = cv2.warpAffine(mk, M, (SEG, SEG), flags=cv2.INTER_NEAREST)
            st = np.clip(st.astype(np.float32) * self.rng.uniform(.92, 1.08)
                         + self.rng.uniform(-10, 10), 0, 255).astype(np.uint8)

        x = torch.from_numpy(st.astype(np.float32).transpose(2, 0, 1) / 255.0)
        y = torch.from_numpy(mk.astype(np.float32)).unsqueeze(0)
        return x, y


def same_split_as_2d(rows):
    """Reproduce the exact patient split used by scripts/15_train_unet.py."""
    pats = sorted({r["patient_id"] for r in rows})
    rng = np.random.default_rng(SEED)
    rng.shuffle(pats)
    n = len(pats)
    tr, va, te = pats[:int(.70 * n)], pats[int(.70 * n):int(.85 * n)], pats[int(.85 * n):]
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
def decompose(model, rows, device, thr):
    """Detection rate / Dice-when-detected / centroid error, as in D23."""
    model.eval()
    det, miss, dices, cd, fps = 0, 0, [], [], []
    for r in rows:
        st = cv2.imread(r["stack_path"], cv2.IMREAD_COLOR)
        gt = (cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        if st.shape[0] != SEG:
            st = cv2.resize(st, (SEG, SEG), interpolation=cv2.INTER_AREA)
            gt = cv2.resize(gt, (SEG, SEG), interpolation=cv2.INTER_NEAREST)
        if gt.sum() == 0:
            continue
        x = torch.from_numpy(st.astype(np.float32).transpose(2, 0, 1) / 255.0)[None].to(device)
        p = torch.sigmoid(model(x).float())[0, 0].cpu().numpy()
        pred = (p > thr).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(pred, 8)
        keep = np.zeros_like(pred); fp = 0
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB:
                keep[lab == i] = 1
                if int(((lab == i) & (gt > 0)).sum()) == 0:
                    fp += 1
        pred = keep
        inter = int((pred & gt).sum())
        fps.append(fp)
        if inter > 0:
            det += 1
            dices.append(2 * inter / max(int(pred.sum()) + int(gt.sum()), 1))
            gy, gx = np.nonzero(gt); py, px = np.nonzero(pred & gt)
            cd.append(float(np.hypot(px.mean() - gx.mean(), py.mean() - gy.mean())))
        else:
            miss += 1
    tot = det + miss
    return {"n_slices": tot, "detection_rate": round(det / max(tot, 1), 4),
            "missed_entirely": miss,
            "dice_over_detected_only": round(float(np.mean(dices)), 4) if dices else None,
            "dice_over_all": round(float(np.mean(dices + [0.0] * miss)), 4),
            "median_centroid_error_px": round(float(np.median(cd)), 2) if cd else None,
            "median_centroid_error_mm": round(float(np.median(cd)) * 0.7, 2) if cd else None,
            "false_positive_blobs_per_slice": round(float(np.mean(fps)), 3)}


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
    rows = list(csv.DictReader(open(LIDC / "lidc_stacks25d.csv", newline="")))
    tr, va, te, (ptr, pva, pte) = same_split_as_2d(rows)
    print(f"2.5D stacks: train={len(tr)} val={len(va)} test={len(te)}")
    print(f"patients:    train={len(ptr)} val={len(pva)} test={len(pte)} "
          f"(identical to the 2D run)")
    assert not (set(ptr) & set(pva)) and not (set(ptr) & set(pte)) and not (set(pva) & set(pte))

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=0, pin_memory=True)
    tl, vl = dl(Stack25D(tr, augment=True), True), dl(Stack25D(va), False)

    model = UNet(in_ch=3, base=args.base).to(device)
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
    print(f"\nrestored best (val Dice={best['dice']:.4f} @ ep {best['epoch']}) "
          f"in {train_s/60:.1f} min")

    # threshold tuned on VALIDATION detection+dice, as before
    best_thr, best_v = 0.5, -1
    for thr in np.arange(0.10, 0.81, 0.05):
        d = decompose(model, va, device, float(thr))["dice_over_all"]
        if d > best_v:
            best_v, best_thr = d, float(thr)
    print(f"threshold tuned on validation: {best_thr:.2f} (val DiceAll {best_v:.4f})")

    new = decompose(model, te, device, best_thr)
    print(f"\n=== 2.5D on held-out patients ({new['n_slices']} slices / {len(pte)} patients) ===")
    for k, v in new.items():
        print(f"  {k:34s} {v}")

    # ---- head-to-head vs the 2D baseline ----
    old_p = REPORTS / "segmentation_error_analysis.json"
    old = json.load(open(old_p)) if old_p.exists() else {}
    if old:
        print("\n=== 2D  ->  2.5D  (identical patients, seed, recipe) ===")
        for k, label in (("detection_rate", "detection rate"),
                         ("dice_over_detected_only", "Dice when detected"),
                         ("dice_over_all", "Dice over all"),
                         ("median_centroid_error_mm", "centre error (mm)"),
                         ("false_positive_blobs_per_slice", "FP blobs/slice")):
            o, n_ = old.get(k), new.get(k)
            if o is None or n_ is None:
                continue
            arrow = "improved" if (n_ > o if "error" not in label and "FP" not in label
                                   else n_ < o) else "worse"
            print(f"  {label:22s} {o:>8} -> {n_:>8}   {arrow}")

    torch.save({"state_dict": model.state_dict(), "base": args.base, "in_ch": 3,
                "size": SEG, "threshold": best_thr, "val_dice": best["dice"],
                "test_dice": new["dice_over_all"], "test_iou": None,
                "mode": "2.5D"}, MODELS / "unet_nodule_25d.pt")
    torch.save({"state_dict": model.state_dict(), "base": args.base, "in_ch": 3},
               MODELS / "unet_nodule_25d_backup.pt")

    rec = {"experiment": "unet_nodule_segmentation_25d",
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED,
           "hardware": hw, "train_time_min": round(train_s / 60, 2),
           "input": "2.5D stack of slices z-1 / z / z+1 (3 channels)",
           "architecture": f"UNet(in_ch=3, base={args.base}, depth=4)",
           "split": {"patients_train": len(ptr), "patients_val": len(pva),
                     "patients_test": len(pte), "note": "identical patients to the 2D run"},
           "threshold_tuned_on_validation": best_thr,
           "test_decomposed": new, "baseline_2d": old, "history": hist}
    json.dump(rec, open(EXPERIMENTS / "unet_segmentation_25d.json", "w"),
              indent=1, default=str)
    print(f"\nsaved models/unet_nodule_25d.pt and experiments/unet_segmentation_25d.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
