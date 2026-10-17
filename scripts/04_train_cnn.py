"""
Phase 4/5 - train the classical CNN branch.

Two-stage schedule:
  Stage 1: freeze the ImageNet encoder, train only the bottleneck + head.
           With 735 images this stops the pretrained filters being destroyed by
           large early gradients.
  Stage 2: unfreeze everything at a lower LR with cosine decay.

Model selection is on validation MACRO-F1, not accuracy -- accuracy would let a
model that ignores the 70 benign training images look good.
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (BATCH_SIZE, EPOCHS_HEAD, EPOCHS_FULL, LR_HEAD, LR_FULL,
                        WEIGHT_DECAY, LABEL_SMOOTHING, EARLY_STOP_PATIENCE, SEED,
                        MODELS, EXPERIMENTS, BACKBONE, set_seed, hardware_summary,
                        CLASSES, USE_AMP)
from src.data.dataset import make_loaders
from src.models.cnn import build_model, FocalLoss
from src.evaluation.metrics import compute_metrics

# Ada (RTX 4050) supports bfloat16. bf16 has the same exponent range as fp32, so
# unlike fp16 it does not overflow to NaN during fine-tuning, and needs no
# GradScaler. An fp16 run of this exact script diverged to NaN at full-finetune
# epoch 3; switching to bf16 fixed it. Documented in reports/decisions.md.
def _amp_dtype(device):
    if device.type != "cuda" or not USE_AMP:
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    P, Y, E = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        dt = _amp_dtype(device)
        with torch.autocast("cuda", dtype=dt or torch.float32, enabled=dt is not None):
            logits, emb = model(x, return_embedding=True)
        P.append(torch.softmax(logits.float(), 1).cpu().numpy())
        E.append(emb.float().cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(Y), np.concatenate(P), np.concatenate(E)


def run_stage(model, loader, val_loader, device, epochs, lr, criterion,
              trainable_encoder: bool, scaler, tag, history, best):
    model.set_encoder_trainable(trainable_encoder)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    for ep in range(epochs):
        model.train()
        tl, nb = 0.0, 0
        t0 = time.time()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            dt = _amp_dtype(device)
            with torch.autocast("cuda", dtype=dt or torch.float32, enabled=dt is not None):
                logits = model(x)
            # Loss in fp32 regardless of autocast dtype: focal loss exponentiates,
            # which is exactly where reduced precision bites.
            loss = criterion(logits.float(), y)
            if not torch.isfinite(loss):
                print("  skipped non-finite loss batch", flush=True)
                continue
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
            tl += loss.item(); nb += 1
        sched.step()

        yv, pv, _ = evaluate(model, val_loader, device)
        m = compute_metrics(yv, pv)
        rec = {"stage": tag, "epoch": ep + 1, "train_loss": round(tl / max(nb, 1), 4),
               "val_f1_macro": round(m["f1_macro"], 4),
               "val_bal_acc": round(m["balanced_accuracy"], 4),
               "val_acc": round(m["accuracy"], 4),
               "val_auc": round(m["roc_auc_macro_ovr"], 4),
               "val_mal_sens": round(m.get("malignant_sensitivity", float('nan')), 4),
               "lr": round(sched.get_last_lr()[0], 6),
               "sec": round(time.time() - t0, 1)}
        history.append(rec)
        star = ""
        if m["f1_macro"] > best["f1"]:
            best.update(f1=m["f1_macro"], state={k: v.detach().cpu().clone()
                                                 for k, v in model.state_dict().items()},
                        epoch=len(history), metrics=m)
            star = "  <-- best"
        print(f"[{tag}] ep{ep+1:02d}/{epochs} loss={rec['train_loss']:.4f} "
              f"valF1={rec['val_f1_macro']:.4f} balAcc={rec['val_bal_acc']:.4f} "
              f"AUC={rec['val_auc']:.4f} sens={rec['val_mal_sens']:.4f} "
              f"({rec['sec']}s){star}", flush=True)

        if len(history) - best["epoch"] >= EARLY_STOP_PATIENCE:
            print(f"early stop: no val improvement for {EARLY_STOP_PATIENCE} epochs")
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default=BACKBONE)
    ap.add_argument("--epochs-head", type=int, default=EPOCHS_HEAD)
    ap.add_argument("--epochs-full", type=int, default=EPOCHS_FULL)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--splits", default="splits.csv",
                    help="splits manifest in data/splits/ (e.g. splits_merged.csv)")
    ap.add_argument("--init-from", default=None,
                    help="tag of a checkpoint to initialise weights from. Enables "
                         "two-stage transfer: pretrain on the merged multi-domain set, "
                         "then fine-tune on the target domain only.")
    ap.add_argument("--lr-scale", type=float, default=1.0,
                    help="multiplier on both learning rates; use <1 when fine-tuning "
                         "an already-trained checkpoint so it is not clobbered.")
    args = ap.parse_args()

    tag = args.tag or args.backbone
    set_seed(args.seed)
    hw = hardware_summary()
    device = torch.device(hw["device"])
    print(json.dumps(hw, indent=1))

    (tr, va, te), (tl, vl, tel) = make_loaders(args.batch_size, num_workers=0,
                                               balanced=True, splits_file=args.splits)
    print(f"splits manifest: {args.splits}")
    print(f"train={len(tr)} val={len(va)} test={len(te)}  classes={CLASSES}")
    print(f"train class counts={tr.class_counts().tolist()}")

    # When --init-from is given we start from our own checkpoint rather than
    # ImageNet, so the ImageNet download is skipped.
    model = build_model(args.backbone, pretrained=args.init_from is None).to(device)
    if args.init_from:
        src = MODELS / f"cnn_{args.init_from}.pt"
        if not src.exists():
            raise SystemExit(f"--init-from checkpoint not found: {src}")
        prev = torch.load(src, map_location=device, weights_only=False)
        model.load_state_dict(prev["state_dict"])
        print(f"initialised from {src.name} (val F1 was {prev.get('val_f1', float('nan')):.4f})")
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"backbone={args.backbone} feat_dim={model.feat_dim} params={n_par/1e6:.2f}M")

    # Both a balanced sampler AND class weights would double-correct, so the
    # sampler handles balance and focal loss handles example difficulty.
    criterion = FocalLoss(weight=None, gamma=1.5,
                          label_smoothing=LABEL_SMOOTHING).to(device)
    _dt = _amp_dtype(device)
    # GradScaler is only needed for fp16; bf16 does not require loss scaling.
    scaler = torch.amp.GradScaler("cuda") if _dt == torch.float16 else None
    print(f"autocast dtype: {_dt}  grad_scaler: {scaler is not None}")

    history, best = [], {"f1": -1.0, "state": None, "epoch": 0, "metrics": None}
    t0 = time.time()
    stopped = run_stage(model, tl, vl, device, args.epochs_head,
                        LR_HEAD * args.lr_scale,
                        criterion, False, scaler, "head", history, best)
    if not stopped:
        run_stage(model, tl, vl, device, args.epochs_full,
                  LR_FULL * args.lr_scale,
                  criterion, True, scaler, "full", history, best)
    train_time = time.time() - t0

    model.load_state_dict(best["state"])
    print(f"\nrestored best checkpoint (val F1={best['f1']:.4f})")

    yv, pv, ev = evaluate(model, vl, device)
    yt, pt, et = evaluate(model, tel, device)
    ytr, ptr, etr = evaluate(model, tl, device)   # embeddings for downstream branches
    val_m, test_m = compute_metrics(yv, pv), compute_metrics(yt, pt)

    print(f"\nVAL  f1={val_m['f1_macro']:.4f} balAcc={val_m['balanced_accuracy']:.4f} "
          f"auc={val_m['roc_auc_macro_ovr']:.4f}")
    print(f"TEST f1={test_m['f1_macro']:.4f} balAcc={test_m['balanced_accuracy']:.4f} "
          f"auc={test_m['roc_auc_macro_ovr']:.4f} acc={test_m['accuracy']:.4f}")
    print("TEST per-class:")
    for c, v in test_m["per_class"].items():
        print(f"   {c:10s} sens={v['recall_sensitivity']:.3f} spec={v['specificity']:.3f} "
              f"prec={v['precision']:.3f} f1={v['f1']:.3f} auc={v['roc_auc_ovr']:.3f} n={v['support']}")
    print("TEST confusion (rows=true):", test_m["confusion_matrix"])

    ckpt = MODELS / f"cnn_{tag}.pt"
    torch.save({"state_dict": model.state_dict(), "backbone": args.backbone,
                "classes": CLASSES, "embed_dim": model.embed_dim,
                "feat_dim": model.feat_dim, "val_f1": best["f1"],
                "seed": args.seed}, ckpt)
    torch.save({"state_dict": model.state_dict(), "backbone": args.backbone},
               MODELS / f"cnn_{tag}_backup.pt")

    # Cache embeddings: every downstream branch (XGBoost/RF/SVM, PCA->quantum,
    # fusion) consumes these, so they are computed exactly once.
    np.savez_compressed(MODELS / f"embeddings_{tag}.npz",
                        train_X=etr, train_y=ytr, train_p=ptr,
                        val_X=ev, val_y=yv, val_p=pv,
                        test_X=et, test_y=yt, test_p=pt)

    exp = {"experiment": f"cnn_{tag}", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
           "backbone": args.backbone, "params_millions": round(n_par / 1e6, 2),
           "hardware": hw, "seed": args.seed,
           "hyperparameters": {"batch_size": args.batch_size, "epochs_head": args.epochs_head,
                               "epochs_full": args.epochs_full, "lr_head": LR_HEAD,
                               "lr_full": LR_FULL, "weight_decay": WEIGHT_DECAY,
                               "loss": "FocalLoss(gamma=1.5)", "label_smoothing": LABEL_SMOOTHING,
                               "sampler": "WeightedRandomSampler(balanced)",
                               "amp_dtype": str(_amp_dtype(device)),
                               "grad_scaler": bool(scaler)},
           "dataset": {"train": len(tr), "val": len(va), "test": len(te),
                       "splits_file": args.splits,
                       "split": "group-aware pseudo-patient, seed 42"},
           "train_time_sec": round(train_time, 1),
           "best_epoch": best["epoch"],
           "val_metrics": val_m, "test_metrics": test_m, "history": history}
    with open(EXPERIMENTS / f"cnn_{tag}.json", "w") as fh:
        json.dump(exp, fh, indent=1)
    print(f"\nsaved {ckpt.name}, embeddings_{tag}.npz, experiments/cnn_{tag}.json")


if __name__ == "__main__":
    main()
