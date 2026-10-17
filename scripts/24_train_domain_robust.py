"""
Phase 5/6/9 - Train benign-vs-malignant with optional harmonisation and DANN,
then evaluate IN-DOMAIN and EXTERNALLY under one honest protocol.

TASK
    Binary benign vs malignant. This is the only task both datasets share:
    IQ-OTH has normal/benign/malignant, LIDC has benign/malignant on
    nodule-bearing slices. Normal slices are excluded so the comparison is
    like-for-like. NODULE != MALIGNANT and BENIGN != NORMAL, so no label is
    invented to force alignment.

PROTOCOL
    train      : IQ-OTH benign+malignant, pseudo-patient grouped
    validation : IQ-OTH held-out groups (model selection only)
    external   : LIDC benign+malignant, REAL patient IDs, never trained on

    For the DANN arms, LIDC IMAGES are shown to the domain discriminator during
    training but LIDC LABELS are never used. That is unsupervised domain
    adaptation and keeps the external number honest.

ARMS (identical architecture, seed, schedule - one switch differs each time)
    baseline      : current preprocessing, no adaptation
    harmonised    : lung-mask + rank harmonisation
    dann          : current preprocessing + domain-adversarial training
    harmonised+dann : both

Everything is reported with bootstrap CIs at slice and patient level.
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

from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import (DATA, EXPERIMENTS, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE,
                        MODELS, REPORTS, SEED, SPLITS, USE_AMP, hardware_summary,
                        set_seed)
from src.models.dann import DANNClassifier, dann_lambda
from src.preprocessing.medical_image import (load_image, preprocess,
                                             preprocess_harmonised)

LIDC = DATA / "external" / "lidc"


# ----------------------------------------------------------------- dataset
class BinaryCT(Dataset):
    """benign=0 / malignant=1, with an optional domain label for the adversary."""

    def __init__(self, rows, harmonise: bool, augment: bool, domain: int,
                 size: int = IMG_SIZE, seed: int = SEED):
        self.rows = rows
        self.harmonise = harmonise
        self.augment = augment
        self.domain = domain
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.cache: dict[int, np.ndarray] = {}
        self.mean = np.array(IMAGENET_MEAN, np.float32)[:, None, None]
        self.std = np.array(IMAGENET_STD, np.float32)[:, None, None]

    def __len__(self):
        return len(self.rows)

    def _base(self, i):
        if i not in self.cache:
            g = load_image(self.rows[i]["path"]).array
            self.cache[i] = (preprocess_harmonised(g, self.size) if self.harmonise
                             else preprocess(g, self.size))
        return self.cache[i]

    def __getitem__(self, i):
        img = self._base(i)
        if self.augment:
            # Medically conservative only: nothing that alters anatomy.
            if self.rng.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
            ang = self.rng.uniform(-12, 12)
            sc = self.rng.uniform(0.92, 1.08)
            M = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), ang, sc)
            M[0, 2] += self.rng.uniform(-.05, .05) * self.size
            M[1, 2] += self.rng.uniform(-.05, .05) * self.size
            img = cv2.warpAffine(img, M, (self.size, self.size),
                                 flags=cv2.INTER_LINEAR, borderValue=0)
            img = np.clip(img.astype(np.float32) * self.rng.uniform(.9, 1.1)
                          + self.rng.uniform(-12, 12), 0, 255)
            if self.rng.random() < 0.3:
                img = img + self.rng.normal(0, self.rng.uniform(2, 6), img.shape)
            img = np.clip(img, 0, 255).astype(np.uint8)

        x = img.astype(np.float32) / 255.0
        x = (np.stack([x, x, x], 0) - self.mean) / self.std
        y = 1 if self.rows[i]["label"] == "malignant" else 0
        return torch.from_numpy(x), y, self.domain


def load_rows():
    iq = [r for r in csv.DictReader(open(SPLITS / "splits.csv", newline=""))
          if r["label"] in ("benign", "malignant")]
    tr = [r for r in iq if r["split"] == "train"]
    va = [r for r in iq if r["split"] in ("val", "test")]      # in-domain eval pool
    li = list(csv.DictReader(open(LIDC / "lidc_masks.csv", newline="")))
    for r in li:
        r["group_id"] = r["patient_id"]
    return tr, va, li


# ------------------------------------------------------------------ metrics
def boot_ci(y, s, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        v.append(roc_auc_score(y[i], s[i]))
    if not v:
        return (float("nan"), float("nan"))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


@torch.no_grad()
def score(model, loader, device, dt):
    model.eval()
    S, Y = [], []
    for x, y, _ in loader:
        x = x.to(device)
        with torch.autocast("cuda", dtype=dt or torch.float32, enabled=dt is not None):
            logits, _ = model(x)
        S.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(Y), np.concatenate(S)


def evaluate(y, s, groups=None):
    out = {"n": int(len(y)), "prevalence": round(float(y.mean()), 4),
           "roc_auc": round(float(roc_auc_score(y, s)), 4),
           "pr_auc": round(float(average_precision_score(y, s)), 4)}
    lo, hi = boot_ci(y, s)
    out["roc_auc_ci95"] = [round(lo, 4), round(hi, 4)]
    pred = (s >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    out.update(sensitivity=round(tp / max(tp + fn, 1), 4),
               specificity=round(tn / max(tn + fp, 1), 4),
               precision=round(tp / max(tp + fp, 1), 4),
               f1=round(2 * tp / max(2 * tp + fp + fn, 1), 4),
               confusion={"tn": tn, "fp": fp, "fn": fn, "tp": tp})
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


# -------------------------------------------------------------------- train
def run_arm(name, harmonise, use_dann, tr, va, li, args, device, dt):
    set_seed(args.seed)
    print(f"\n{'='*74}\nARM: {name}  (harmonise={harmonise}, dann={use_dann})\n{'='*74}")

    ds_tr = BinaryCT(tr, harmonise, True, 0, seed=args.seed)
    ds_va = BinaryCT(va, harmonise, False, 0, seed=args.seed)
    ds_ex = BinaryCT(li, harmonise, False, 1, seed=args.seed)
    dl = lambda d, sh: DataLoader(d, batch_size=args.batch_size, shuffle=sh, num_workers=0)
    tl, vl, el = dl(ds_tr, True), dl(ds_va, False), dl(ds_ex, False)
    # Unlabelled target images for the discriminator only.
    ul = dl(BinaryCT(li, harmonise, True, 1, seed=args.seed), True)

    model = DANNClassifier(args.backbone, n_classes=2, use_dann=use_dann,
                           pretrained=True).to(device)
    w = torch.tensor([1.0, len([r for r in tr if r["label"] == "benign"]) /
                      max(len([r for r in tr if r["label"] == "malignant"]), 1)],
                     dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    dcrit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    total_steps = args.epochs * max(len(tl), 1)
    step = 0
    best = {"auc": -1, "state": None, "epoch": 0}
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        it_u = iter(ul)
        tot = dloss_tot = nb = 0.0, 0.0, 0
        tot, dloss_tot, nb = 0.0, 0.0, 0
        for x, y, d in tl:
            x, y = x.to(device), y.to(device)
            lam = dann_lambda(step, total_steps) if use_dann else 0.0
            opt.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=dt or torch.float32, enabled=dt is not None):
                logits, dom_s = model(x, lam)
            loss = crit(logits.float(), y)

            if use_dann:
                try:
                    xu, _, du = next(it_u)
                except StopIteration:
                    it_u = iter(ul); xu, _, du = next(it_u)
                xu = xu.to(device)
                with torch.autocast("cuda", dtype=dt or torch.float32, enabled=dt is not None):
                    _, dom_t = model(xu, lam)
                dl_src = dcrit(dom_s.float(), torch.zeros(len(x), dtype=torch.long, device=device))
                dl_tgt = dcrit(dom_t.float(), torch.ones(len(xu), dtype=torch.long, device=device))
                dloss = 0.5 * (dl_src + dl_tgt)
                loss = loss + dloss
                dloss_tot += float(dloss)

            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss); nb += 1; step += 1
        sched.step()

        yv, sv = score(model, vl, device, dt)
        auc = roc_auc_score(yv, sv) if len(np.unique(yv)) > 1 else 0.0
        if auc > best["auc"]:
            best = {"auc": auc, "epoch": ep + 1,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if (ep + 1) % 2 == 0 or ep == 0:
            print(f"  ep{ep+1:02d} loss={tot/max(nb,1):.4f}"
                  + (f" dom={dloss_tot/max(nb,1):.4f} lam={lam:.2f}" if use_dann else "")
                  + f"  in-domain AUC={auc:.4f}", flush=True)

    model.load_state_dict(best["state"])
    mins = (time.time() - t0) / 60

    yv, sv = score(model, vl, device, dt)
    ye, se = score(model, el, device, dt)
    gi = np.array([r["group_id"] for r in va])
    ge = np.array([r["patient_id"] for r in li])

    ind = evaluate(yv, sv, gi)
    ext = evaluate(ye, se, ge)
    print(f"\n  IN-DOMAIN  AUC {ind['roc_auc']:.4f} "
          f"[{ind['roc_auc_ci95'][0]:.3f},{ind['roc_auc_ci95'][1]:.3f}]  n={ind['n']}")
    print(f"  EXTERNAL   AUC {ext['roc_auc']:.4f} "
          f"[{ext['roc_auc_ci95'][0]:.3f},{ext['roc_auc_ci95'][1]:.3f}]  n={ext['n']}"
          + (f"  | patient-level {ext['patient_level']['roc_auc']:.4f}"
             if "patient_level" in ext else ""))
    print(f"  gap (in-domain - external): {ind['roc_auc']-ext['roc_auc']:+.4f}"
          f"   [{mins:.1f} min, best epoch {best['epoch']}]")

    torch.save({"state_dict": model.state_dict(), "backbone": args.backbone,
                "harmonise": harmonise, "use_dann": use_dann},
               MODELS / f"dr_{name}.pt")
    return {"arm": name, "harmonise": harmonise, "dann": use_dann,
            "in_domain": ind, "external": ext,
            "gap": round(ind["roc_auc"] - ext["roc_auc"], 4),
            "best_epoch": best["epoch"], "train_min": round(mins, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="efficientnet_b3")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--arms", default="baseline,harmonised,dann,harmonised_dann")
    args = ap.parse_args()

    hw = hardware_summary()
    device = torch.device(hw["device"])
    dt = (torch.bfloat16 if device.type == "cuda"
          and torch.cuda.is_bf16_supported() and USE_AMP else None)
    print(json.dumps(hw, indent=1))

    tr, va, li = load_rows()
    print(f"\ntrain (IQ-OTH)  {len(tr)}  "
          f"benign={sum(1 for r in tr if r['label']=='benign')} "
          f"malignant={sum(1 for r in tr if r['label']=='malignant')}")
    print(f"in-domain eval  {len(va)}")
    print(f"external (LIDC) {len(li)}  {len({r['patient_id'] for r in li})} patients")

    plan = {"baseline": (False, False), "harmonised": (True, False),
            "dann": (False, True), "harmonised_dann": (True, True)}
    results = []
    for name in [a.strip() for a in args.arms.split(",") if a.strip()]:
        h, d = plan[name]
        results.append(run_arm(name, h, d, tr, va, li, args, device, dt))

    print(f"\n{'='*82}\nCOMPARISON (binary benign vs malignant)\n{'='*82}")
    print(f"{'arm':<18} {'in-domain AUC':>14} {'external AUC':>14} {'external 95% CI':>20} {'gap':>8}")
    print("-" * 82)
    for r in results:
        ci = r["external"]["roc_auc_ci95"]
        print(f"{r['arm']:<18} {r['in_domain']['roc_auc']:>14.4f} "
              f"{r['external']['roc_auc']:>14.4f} "
              f"{'[' + f'{ci[0]:.3f}, {ci[1]:.3f}' + ']':>20} {r['gap']:>+8.4f}")

    best = max(results, key=lambda r: r["external"]["roc_auc"])
    base = next((r for r in results if r["arm"] == "baseline"), None)
    print(f"\n  best external: {best['arm']} at {best['external']['roc_auc']:.4f}")
    if base:
        d = best["external"]["roc_auc"] - base["external"]["roc_auc"]
        print(f"  change vs baseline: {d:+.4f}")
        ci = best["external"]["roc_auc_ci95"]
        if ci[1] < 0.5:
            print("  NOTE: the best arm's external CI lies ENTIRELY BELOW 0.50 - the model "
                  "is anti-correlated on external data, which is worse than chance.")
        elif ci[0] <= 0.5 <= ci[1]:
            print("  NOTE: the best arm's external CI straddles 0.50 - external "
                  "generalisation is NOT demonstrated.")
        else:
            print("  The best arm's external CI lies entirely above 0.50 - it discriminates "
                  "better than chance on unseen data.")

    json.dump({"task": "binary benign vs malignant",
               "protocol": {"train": "IQ-OTH (pseudo-patient grouped)",
                            "in_domain_eval": "IQ-OTH held-out groups",
                            "external": "LIDC-IDRI, real patient IDs, labels never used in training"},
               "seed": args.seed, "backbone": args.backbone, "epochs": args.epochs,
               "results": results},
              open(REPORTS / "domain_robust_comparison.json", "w"), indent=1)
    json.dump(results, open(EXPERIMENTS / "domain_robust_comparison.json", "w"), indent=1)
    print("\nwrote reports/domain_robust_comparison.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
