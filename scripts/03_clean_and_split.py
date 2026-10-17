"""
Phase 2+3 - Cleaning (quarantine, never delete) + leakage-safe group-aware splitting.
Raw data at OneDrive is READ-ONLY; nothing there is modified or deleted.
"""
import os, json, shutil, collections, csv
import numpy as np

ROOT = r"C:\Users\alanf\sih-quantum-lung"
REP  = os.path.join(ROOT, "reports")
QUAR = os.path.join(ROOT, "data", "quarantine")
SPL  = os.path.join(ROOT, "data", "splits")
os.makedirs(QUAR, exist_ok=True); os.makedirs(SPL, exist_ok=True)
SEED = 42

recs   = json.load(open(os.path.join(REP, "audit_raw.json")))
groups = json.load(open(os.path.join(REP, "pseudo_patient_groups.json")))
iq = [r for r in recs if r["ok"] and r["source"] == "IQ-OTH/NCCD"]

# ---------------- CLEANING: flag, don't delete ----------------
flags = collections.defaultdict(list)

# 1) exact duplicates -> keep first (lexicographic), flag the rest
by_md5 = collections.defaultdict(list)
for r in sorted(iq, key=lambda x: x["file"]): by_md5[r["md5"]].append(r)
for md5, v in by_md5.items():
    for r in v[1:]:
        flags[r["file"]].append(f"EXACT_DUPLICATE_of:{v[0]['file']}")

# 2) label-ambiguous: member of a pseudo-patient group with mixed labels
for r in iq:
    g = groups.get(r["file"])
    if g and len(g["group_label_mix"]) > 1:
        flags[r["file"]].append(f"LABEL_AMBIGUOUS_group_mix:{g['group_label_mix']}")

# 3) geometry outliers (non-512x512)
for r in iq:
    if (r["width"], r["height"]) != (512, 512):
        flags[r["file"]].append(f"NON_STANDARD_RESOLUTION:{r['width']}x{r['height']}")

# 4) quality: near-blank / low dynamic range
for r in iq:
    if r["std"] < 10: flags[r["file"]].append(f"LOW_CONTRAST_std:{r['std']:.1f}")
    if r["nonzero_frac"] < 0.05: flags[r["file"]].append(f"MOSTLY_EMPTY:{r['nonzero_frac']:.3f}")

# Decide exclusion policy: only EXACT_DUPLICATE and LABEL_AMBIGUOUS are excluded from
# modelling. Resolution outliers are kept (resizing handles them) but recorded.
EXCLUDE_PREFIX = ("EXACT_DUPLICATE", "LABEL_AMBIGUOUS")
excluded = {f for f, rs in flags.items() if any(r.startswith(EXCLUDE_PREFIX) for r in rs)}

with open(os.path.join(REP, "cleaning_log.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["file", "label", "action", "reasons"])
    for r in iq:
        f = r["file"]
        if f in flags:
            act = "EXCLUDED_FROM_MODELLING" if f in excluded else "KEPT_WITH_NOTE"
            w.writerow([f, r["label"], act, " | ".join(flags[f])])

# quarantine = COPY of excluded files (originals untouched)
copied = 0
for r in iq:
    if r["file"] in excluded:
        d = os.path.join(QUAR, r["label"]); os.makedirs(d, exist_ok=True)
        try:
            shutil.copy2(r["path"], os.path.join(d, r["file"])); copied += 1
        except Exception as e:
            print("quarantine copy failed", r["file"], e)

print(f"flagged files: {len(flags)}   excluded from modelling: {len(excluded)}   "
      f"copied to quarantine: {copied}")
print("flag reason histogram:",
      collections.Counter(rr.split(':')[0].split('_of')[0] for rs in flags.values() for rr in rs))

# ---------------- SPLIT: group-aware, stratified ----------------
clean = [r for r in iq if r["file"] not in excluded]
print(f"\nclean modelling pool: {len(clean)}  {collections.Counter(r['label'] for r in clean)}")

gid_of = {r["file"]: groups[r["file"]]["group_id"] for r in clean}
g_items = collections.defaultdict(list)
for r in clean: g_items[gid_of[r["file"]]].append(r)

# group label = majority label of the group
g_label = {g: collections.Counter(r["label"] for r in v).most_common(1)[0][0]
           for g, v in g_items.items()}

rng = np.random.default_rng(SEED)
train, val, test = [], [], []
for lab in ["benign", "malignant", "normal"]:
    gs = sorted([g for g in g_items if g_label[g] == lab])
    rng.shuffle(gs)
    # split by cumulative IMAGE count so class proportions are honoured
    tot = sum(len(g_items[g]) for g in gs)
    acc, bucket = 0, []
    for g in gs:
        frac = acc / tot
        dest = train if frac < 0.70 else (val if frac < 0.85 else test)
        dest.extend(g_items[g]); acc += len(g_items[g])

def dist(s): return dict(collections.Counter(r["label"] for r in s))
print(f"\nSEED={SEED}  group-aware split (pseudo-patient groups kept intact)")
for nm, s in [("train", train), ("val", val), ("test", test)]:
    print(f"  {nm:5s} n={len(s):4d} ({100*len(s)/len(clean):4.1f}%)  {dist(s)}")

# verify zero group overlap
gt, gv, ge = ({gid_of[r['file']] for r in s} for s in (train, val, test))
print(f"  group overlap train/val={len(gt&gv)}  train/test={len(gt&ge)}  val/test={len(gv&ge)}  (must all be 0)")

with open(os.path.join(SPL, "splits.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["split", "path", "file", "label", "group_id"])
    for nm, s in [("train", train), ("val", val), ("test", test)]:
        for r in s: w.writerow([nm, r["path"], r["file"], r["label"], gid_of[r["file"]]])

json.dump({"seed": SEED, "grouping": "phash16 connected components @ hamming<=12",
           "n_clean": len(clean), "n_excluded": len(excluded),
           "counts": {"train": len(train), "val": len(val), "test": len(test)},
           "class_dist": {"train": dist(train), "val": dist(val), "test": dist(test)},
           "group_overlap": {"train_val": len(gt&gv), "train_test": len(gt&ge), "val_test": len(gv&ge)}},
          open(os.path.join(SPL, "split_manifest.json"), "w"), indent=1)
print("\nwrote data/splits/splits.csv + split_manifest.json")
