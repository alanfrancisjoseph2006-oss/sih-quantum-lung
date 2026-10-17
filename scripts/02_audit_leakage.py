"""
Phase 1b - leakage / grouping forensics + annotation extraction.
The IQ-OTH/NCCD README states 110 patients, but filenames carry NO patient ID.
We quantify how severe the resulting leakage risk is, and build a
perceptual-similarity connected-component grouping as a surrogate "pseudo-patient"
grouping so that splits can at least be made group-aware.
"""
import os, json, glob, collections
import numpy as np
from PIL import Image
import imagehash

OUT = r"C:\Users\alanf\sih-quantum-lung\reports"
recs = json.load(open(os.path.join(OUT, "audit_raw.json")))
iq = [r for r in recs if r["ok"] and r["source"] == "IQ-OTH/NCCD"]
print(f"IQ images: {len(iq)}")

hs = [imagehash.hex_to_hash(r["phash"]) for r in iq]
n = len(iq)

# ---- connected components over phash similarity graph at several thresholds ----
def components(thresh):
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    for i in range(n):
        for j in range(i+1, n):
            if hs[i] - hs[j] <= thresh: union(i, j)
    groups = collections.defaultdict(list)
    for i in range(n): groups[find(i)].append(i)
    return list(groups.values())

for t in (4, 8, 12, 16, 20):
    g = components(t)
    sizes = sorted((len(x) for x in g), reverse=True)
    impure = sum(1 for x in g if len({iq[i]["label"] for i in x}) > 1)
    print(f"phash<= {t:2d} -> {len(g):4d} groups | largest {sizes[:6]} | label-impure groups: {impure}")

# choose t=12 as pseudo-patient grouping (documented in dataset_report.md)
T = 12
groups = components(T)
groups.sort(key=len, reverse=True)
gmap = {}
for gi, members in enumerate(groups):
    labs = collections.Counter(iq[i]["label"] for i in members)
    for i in members:
        gmap[iq[i]["file"]] = {"group_id": int(gi), "group_size": len(members),
                               "group_label_mix": dict(labs)}
print(f"\nchosen threshold {T}: {len(groups)} pseudo-patient groups for {n} images "
      f"(README says 110 real patients)")
print("group size distribution:", collections.Counter(len(g) for g in groups).most_common(10))

impure = [g for g in groups if len({iq[i]['label'] for i in g}) > 1]
print(f"label-impure groups: {len(impure)}")
for g in impure[:5]:
    print("   ", collections.Counter(iq[i]['label'] for i in g), [iq[i]['file'] for i in g][:5])

json.dump(gmap, open(os.path.join(OUT, "pseudo_patient_groups.json"), "w"), indent=1)

# ---- estimate leakage magnitude if a naive random split were used ----
rng = np.random.default_rng(42)
idx = rng.permutation(n); cut = int(0.8*n)
tr, te = set(idx[:cut].tolist()), set(idx[cut:].tolist())
leaked = 0
for g in groups:
    if any(i in tr for i in g) and any(i in te for i in g): leaked += 1
n_te_leaked = sum(1 for i in te if any(i in g and any(j in tr for j in g) for g in groups if i in g))
print(f"\nNAIVE random 80/20 split -> {leaked}/{len(groups)} groups straddle train&test; "
      f"{n_te_leaked}/{len(te)} test images ({100*n_te_leaked/len(te):.1f}%) have a "
      f"near-duplicate sibling in train.  <-- this is the leakage the report must flag.")

# ---- extract green bounding boxes from DeepLesion 'Test cases' ----
TC = r"C:\Users\alanf\OneDrive\Desktop\sih dataset\Test cases"
boxes = []
for f in sorted(glob.glob(os.path.join(TC, "*.png"))):
    a = np.asarray(Image.open(f).convert("RGB")).astype(np.int16)
    green = (a[...,1] > 150) & (a[...,0] < 100) & (a[...,2] < 100)
    ys, xs = np.nonzero(green)
    if len(xs) == 0: continue
    stem = os.path.splitext(os.path.basename(f))[0]
    pid, study, series, sl = stem.split("_")
    boxes.append({"file": os.path.basename(f), "patient_id": pid, "study": study,
                  "series": series, "slice": sl,
                  "x1": int(xs.min()), "y1": int(ys.min()),
                  "x2": int(xs.max()), "y2": int(ys.max()),
                  "w": int(xs.max()-xs.min()), "h": int(ys.max()-ys.min()),
                  "img_w": int(a.shape[1]), "img_h": int(a.shape[0])})
json.dump(boxes, open(os.path.join(OUT, "deeplesion_boxes.json"), "w"), indent=1)
ws = np.array([b["w"] for b in boxes]); hsz = np.array([b["h"] for b in boxes])
print(f"\nDeepLesion overlay boxes extracted: {len(boxes)} from "
      f"{len({b['patient_id'] for b in boxes})} patients")
print(f"box width  px: min={ws.min()} med={np.median(ws):.0f} max={ws.max()}")
print(f"box height px: min={hsz.min()} med={np.median(hsz):.0f} max={hsz.max()}")
print(f"median lesion area fraction of image: "
      f"{np.median(ws*hsz/(512*512))*100:.3f}%")
