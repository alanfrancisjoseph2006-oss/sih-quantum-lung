"""
Phase 13d - merge LIDC into the TRAINING split, then probe for a dataset shortcut.

MERGE RULE
  IQ-OTH train + LIDC slices  -> train
  IQ-OTH val                  -> val   (unchanged, byte-identical)
  IQ-OTH test                 -> test  (unchanged, byte-identical)

Validation and test stay pure IQ-OTH so every number remains directly comparable
to the pre-LIDC results, and so a dataset shortcut cannot inflate them.

GROUPING
  LIDC has REAL patient IDs, unlike IQ-OTH. Each LIDC patient becomes one group,
  offset by 100000 so it cannot collide with an IQ-OTH pseudo-patient group id.
  All LIDC rows are in train, so no cross-split group leak is possible - but the
  ids are kept correct anyway so grouped CV downstream stays valid.

DATASET-IDENTITY PROBE
  Adding a second domain risks the model learning "this looks like LIDC" instead
  of lesion morphology. The probe trains a classifier to predict SOURCE from the
  CNN embedding. A near-perfect probe means the shortcut is available; combined
  with the class balance of the added data it tells us how dangerous that is.
  Reported honestly either way.
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA, SPLITS, REPORTS, SEED, CLASSES

LIDC = DATA / "external" / "lidc"
GROUP_OFFSET = 100_000


def build_merged(max_per_patient: int) -> dict:
    src = SPLITS / "splits.csv"
    rows = list(csv.DictReader(open(src, newline="")))
    for r in rows:
        r.setdefault("source", "IQ-OTH")

    lidc_csv = LIDC / "lidc_slices.csv"
    if not lidc_csv.exists():
        raise SystemExit(f"missing {lidc_csv} - run scripts/11_lidc_extract_slices.py")
    lidc = list(csv.DictReader(open(lidc_csv, newline="")))
    print(f"IQ-OTH rows: {len(rows)}   LIDC slices available: {len(lidc)}")

    # Cap per patient so one heavily-annotated scan cannot dominate.
    rng = np.random.default_rng(SEED)
    by_pat = {}
    for r in lidc:
        by_pat.setdefault(r["patient_id"], []).append(r)
    kept = []
    pid_to_group = {}
    for i, (pid, items) in enumerate(sorted(by_pat.items())):
        pid_to_group[pid] = GROUP_OFFSET + i
        if len(items) > max_per_patient:
            idx = rng.choice(len(items), max_per_patient, replace=False)
            items = [items[j] for j in sorted(idx)]
        kept.extend(items)
    print(f"after per-patient cap ({max_per_patient}): {len(kept)} slices "
          f"from {len(by_pat)} patients")

    merged = list(rows)
    for r in kept:
        merged.append({"split": "train", "path": r["path"], "file": r["file"],
                       "label": r["label"], "group_id": str(pid_to_group[r["patient_id"]]),
                       "source": "LIDC"})

    out = SPLITS / "splits_merged.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "path", "file", "label",
                                           "group_id", "source"])
        w.writeheader()
        for r in merged:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    # ---- integrity checks -------------------------------------------------
    g = {s: set() for s in ("train", "val", "test")}
    for r in merged:
        g[r["split"]].add(r["group_id"])
    assert not (g["train"] & g["val"]), "group leak train/val"
    assert not (g["train"] & g["test"]), "group leak train/test"
    assert not (g["val"] & g["test"]), "group leak val/test"

    orig = [r for r in rows if r["split"] in ("val", "test")]
    new = [r for r in merged if r["split"] in ("val", "test")]
    assert len(orig) == len(new), "val/test size changed - must stay identical"
    assert all(r["source"] == "IQ-OTH" for r in new), "external data leaked into val/test"

    stats = {}
    for sp in ("train", "val", "test"):
        sub = [r for r in merged if r["split"] == sp]
        stats[sp] = {"n": len(sub),
                     "by_label": dict(Counter(r["label"] for r in sub)),
                     "by_source": dict(Counter(r.get("source", "IQ-OTH") for r in sub))}
    print("\n=== MERGED SPLITS ===")
    for sp, s in stats.items():
        print(f"  {sp:5s} n={s['n']:5d}  {s['by_label']}  sources={s['by_source']}")

    before = dict(Counter(r["label"] for r in rows if r["split"] == "train"))
    after = stats["train"]["by_label"]
    print("\n  train class counts:")
    for c in CLASSES:
        b, a = before.get(c, 0), after.get(c, 0)
        print(f"    {c:10s} {b:4d} -> {a:4d}  ({a/max(b,1):.1f}x)")
    print("\n  group overlap train/val/test = 0 (verified)")
    print("  val + test unchanged and 100% IQ-OTH (verified)")
    return {"stats": stats, "before_train": before, "path": str(out)}


def probe(tag: str) -> dict:
    """Train a source classifier on CNN embeddings of the merged training set."""
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score
    from src.data.dataset import LungCTDataset
    from src.models.cnn import LungCNN
    from src.config import MODELS

    ck_path = MODELS / f"cnn_{tag}.pt"
    if not ck_path.exists():
        return {"skipped": f"no checkpoint {ck_path.name}"}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    m = LungCNN(backbone=ck["backbone"], pretrained=False).to(dev)
    m.load_state_dict(ck["state_dict"]); m.eval()

    ds = LungCTDataset("train", augment=False, splits_file="splits_merged.csv")
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=32, shuffle=False)
    E = []
    with torch.no_grad():
        for x, _ in dl:
            E.append(m.forward_embedding(x.to(dev)).float().cpu().numpy())
    E = np.concatenate(E)
    src = np.array([0 if s == "IQ-OTH" else 1 for s in ds.sources()])
    groups = ds.groups
    print(f"\n=== DATASET-IDENTITY PROBE ===")
    print(f"  embeddings {E.shape}  IQ-OTH={int((src==0).sum())} LIDC={int((src==1).sum())}")

    aucs = []
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for tr, te in sgkf.split(E, src, groups=groups):
        clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                                 random_state=SEED).fit(E[tr], src[tr])
        if len(np.unique(src[te])) < 2:
            continue
        aucs.append(roc_auc_score(src[te], clf.predict_proba(E[te])[:, 1]))
    auc = float(np.mean(aucs)) if aucs else float("nan")

    # How much does source tell you about label? Normalised mutual information.
    lab = ds.labels
    from sklearn.metrics import normalized_mutual_info_score
    nmi = float(normalized_mutual_info_score(src, lab))

    print(f"  source-prediction ROC-AUC (grouped 5-fold): {auc:.4f}")
    print(f"  normalised mutual information(source; label): {nmi:.4f}")
    verdict = ("Source is trivially separable, but that only becomes a shortcut if source "
               "also predicts the label. NMI quantifies exactly that." if auc > 0.9 else
               "Source is not trivially separable.")
    print(f"  -> {verdict}")
    print("  -> Decisive safeguard: val and test are 100% IQ-OTH, so a source shortcut "
          "cannot inflate the reported metrics.")
    return {"source_auc": round(auc, 4), "nmi_source_label": round(nmi, 4),
            "n_iqoth": int((src == 0).sum()), "n_lidc": int((src == 1).sum()),
            "interpretation": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-patient", type=int, default=12)
    ap.add_argument("--tag", default="efficientnet_b3")
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args()

    res = build_merged(args.max_per_patient)
    if not args.no_probe:
        res["probe"] = probe(args.tag)
    json.dump(res, open(REPORTS / "lidc_merge.json", "w"), indent=1, default=str)
    print(f"\nwrote {res['path']} and reports/lidc_merge.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
