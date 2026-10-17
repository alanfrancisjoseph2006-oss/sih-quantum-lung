"""
Phase 1 - Forensic dataset audit for SIH 2026 PS-26139 (Team CodeNova).
Runs with plain numpy/PIL/imagehash - no torch required.
Writes: reports/audit_raw.json
"""
import os, sys, json, hashlib, glob, collections
import numpy as np
from PIL import Image
import imagehash

BASE = r"C:\Users\alanf\OneDrive\Desktop\sih dataset"
IQ = os.path.join(BASE, "The IQ-OTHNCCD lung cancer dataset", "The IQ-OTHNCCD lung cancer dataset")
TEST = os.path.join(BASE, "Test cases")
OUT = r"C:\Users\alanf\sih-quantum-lung\reports"
os.makedirs(OUT, exist_ok=True)

CLASSES = {"Bengin cases": "benign", "Malignant cases": "malignant", "Normal cases": "normal"}

def probe(path, label, source):
    rec = {"path": path, "file": os.path.basename(path), "label": label, "source": source}
    try:
        rec["bytes"] = os.path.getsize(path)
        with open(path, "rb") as fh:
            raw = fh.read()
        rec["md5"] = hashlib.md5(raw).hexdigest()
        im = Image.open(path)
        im.verify()                      # integrity check
        im = Image.open(path)            # reopen after verify
        rec["mode"], rec["format"] = im.mode, im.format
        rec["width"], rec["height"] = im.size
        g = np.asarray(im.convert("L"), dtype=np.uint8)
        rec["mean"] = float(g.mean()); rec["std"] = float(g.std())
        rec["min"] = int(g.min()); rec["max"] = int(g.max())
        rec["p01"] = float(np.percentile(g, 1)); rec["p99"] = float(np.percentile(g, 99))
        rec["nonzero_frac"] = float((g > 10).mean())
        # is the RGB actually colour, or grayscale replicated across 3 channels?
        if im.mode in ("RGB", "RGBA"):
            a = np.asarray(im.convert("RGB"), dtype=np.int16)
            rec["is_true_gray"] = bool(
                np.abs(a[..., 0] - a[..., 1]).max() == 0 and np.abs(a[..., 1] - a[..., 2]).max() == 0)
        else:
            rec["is_true_gray"] = True
        rec["phash"] = str(imagehash.phash(im.convert("L"), hash_size=16))
        rec["dhash"] = str(imagehash.dhash(im.convert("L"), hash_size=8))
        rec["ok"] = True
    except Exception as e:
        rec["ok"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec

records = []
for folder, label in CLASSES.items():
    for f in sorted(glob.glob(os.path.join(IQ, folder, "*"))):
        if os.path.isfile(f) and not f.lower().endswith(".txt"):
            records.append(probe(f, label, "IQ-OTH/NCCD"))
for f in sorted(glob.glob(os.path.join(TEST, "*"))):
    if os.path.isfile(f):
        records.append(probe(f, None, "TestCases"))

with open(os.path.join(OUT, "audit_raw.json"), "w") as fh:
    json.dump(records, fh, indent=1)

ok = [r for r in records if r["ok"]]
bad = [r for r in records if not r["ok"]]
print(f"probed={len(records)}  readable={len(ok)}  corrupt={len(bad)}")
for r in bad: print("  CORRUPT:", r["file"], r.get("error"))

print("\n--- class counts (IQ-OTH/NCCD) ---")
print(collections.Counter(r["label"] for r in ok if r["source"] == "IQ-OTH/NCCD"))
print("--- resolutions ---")
print(collections.Counter((r["width"], r["height"]) for r in ok).most_common(15))
print("--- modes ---", collections.Counter(r["mode"] for r in ok))
print("--- true-grayscale? ---", collections.Counter(r["is_true_gray"] for r in ok))

print("\n--- exact duplicates (md5) ---")
by_md5 = collections.defaultdict(list)
for r in ok: by_md5[r["md5"]].append(r)
dups = {k: v for k, v in by_md5.items() if len(v) > 1}
print(f"md5 groups with >1 member: {len(dups)}; extra files: {sum(len(v)-1 for v in dups.values())}")
cross = 0
for v in dups.values():
    labs = {x["label"] for x in v}
    if len(labs) > 1:
        cross += 1
        print("  CROSS-LABEL DUP:", labs, [x["file"] for x in v][:4])
print(f"cross-label duplicate groups: {cross}")

print("\n--- near duplicates (phash hamming<=4, within IQ) ---")
iq = [r for r in ok if r["source"] == "IQ-OTH/NCCD"]
hs = [imagehash.hex_to_hash(r["phash"]) for r in iq]
near = 0; near_cross = 0; pairs = []
for i in range(len(iq)):
    for j in range(i+1, len(iq)):
        d = hs[i] - hs[j]
        if d <= 4:
            near += 1
            if iq[i]["label"] != iq[j]["label"]:
                near_cross += 1
                if len(pairs) < 10: pairs.append((iq[i]["file"], iq[j]["file"], d))
print(f"near-dup pairs: {near}  cross-label near-dup pairs: {near_cross}")
for p in pairs: print("   ", p)

print("\n--- suspect quality ---")
blank = [r for r in ok if r["std"] < 5]
dark  = [r for r in ok if r["nonzero_frac"] < 0.02]
print(f"near-blank (std<5): {len(blank)}  mostly-empty(nonzero<2%): {len(dark)}")
for r in (blank+dark)[:10]: print("   ", r["source"], r["file"], f"std={r['std']:.1f} nz={r['nonzero_frac']:.3f}")

print("\n--- Test cases filename parse (DeepLesion convention) ---")
tc = [r for r in ok if r["source"] == "TestCases"]
pat = collections.Counter()
pids = set()
for r in tc:
    parts = os.path.splitext(r["file"])[0].split("_")
    pat[len(parts)] += 1
    if len(parts) == 4: pids.add(parts[0])
print("token counts:", dict(pat), " unique patient tokens:", len(pids))
print("sample patients:", sorted(pids)[:10])
