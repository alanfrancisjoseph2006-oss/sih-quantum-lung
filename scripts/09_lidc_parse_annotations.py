"""
Phase 13a - parse LIDC-IDRI XML annotations (no DICOM download needed yet).

The XML package (9 MB) contains every radiologist reading for all 1,018 LIDC
series. Parsing it FIRST lets us rank series by how many benign nodules they
contain, so a 50-series download targets exactly the class we are short of
instead of sampling blindly.

LABEL SEMANTICS (documented, because this differs from IQ-OTH/NCCD)
-------------------------------------------------------------------
LIDC malignancy is a radiologist IMPRESSION on a 1-5 scale, given independently
by up to 4 readers per nodule. It is NOT pathology-confirmed.
  1 = highly unlikely malignant .. 5 = highly suspicious
We cluster annotations of the same physical nodule by 3D centroid proximity,
average the readers' ratings, and map:
    mean < 3   -> benign
    mean > 3   -> malignant
    mean == 3  -> INDETERMINATE, excluded entirely (never silently assigned)
Only nodules with a <characteristics> block are usable; LIDC omits that block
for sub-3mm nodules, which therefore carry no malignancy rating and are skipped.
"""
from __future__ import annotations
import json, re, sys, zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA, REPORTS

LIDC = DATA / "external" / "lidc"
CLUSTER_MM = 8.0          # annotations within this 3D distance = same nodule


def strip_ns(t: str) -> str:
    return t.split("}")[-1] if "}" in t else t


def parse_xml(raw: bytes) -> tuple[str | None, list[dict]]:
    """Return (series_uid, [nodule annotations]) for one XML reading file."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return None, []

    series = None
    for el in root.iter():
        if strip_ns(el.tag) == "SeriesInstanceUid" and el.text:
            series = el.text.strip()
            break

    out = []
    for sess in root.iter():
        if strip_ns(sess.tag) != "readingSession":
            continue
        for nod in sess:
            if strip_ns(nod.tag) != "unblindedReadNodule":
                continue
            mal, rois = None, []
            for child in nod:
                tag = strip_ns(child.tag)
                if tag == "characteristics":
                    for c in child:
                        if strip_ns(c.tag) == "malignancy" and c.text:
                            try:
                                mal = int(c.text.strip())
                            except ValueError:
                                pass
                elif tag == "roi":
                    z, xs, ys, sop = None, [], [], None
                    for r in child:
                        rt = strip_ns(r.tag)
                        if rt == "imageZposition" and r.text:
                            z = float(r.text)
                        elif rt == "imageSOP_UID" and r.text:
                            sop = r.text.strip()
                        elif rt == "edgeMap":
                            xv = yv = None
                            for e in r:
                                if strip_ns(e.tag) == "xCoord" and e.text:
                                    xv = int(float(e.text))
                                elif strip_ns(e.tag) == "yCoord" and e.text:
                                    yv = int(float(e.text))
                            if xv is not None and yv is not None:
                                xs.append(xv); ys.append(yv)
                    if z is not None and xs:
                        rois.append({"z": z, "sop": sop,
                                     "x1": min(xs), "x2": max(xs),
                                     "y1": min(ys), "y2": max(ys),
                                     "cx": float(np.mean(xs)), "cy": float(np.mean(ys)),
                                     "n_pts": len(xs)})
            # Only >=3mm nodules carry a malignancy rating.
            if mal is not None and rois:
                zs = [r["z"] for r in rois]
                out.append({"malignancy": mal, "rois": rois,
                            "cz": float(np.mean(zs)),
                            "cx": float(np.mean([r["cx"] for r in rois])),
                            "cy": float(np.mean([r["cy"] for r in rois]))})
    return series, out


def cluster(anns: list[dict]) -> list[list[dict]]:
    """Greedy 3D clustering: annotations of the same physical nodule from
    different readers land within a few mm of each other."""
    clusters = []
    for a in anns:
        placed = False
        for c in clusters:
            # pixel coords are ~0.7mm; z is already mm. Compare loosely.
            d = np.hypot((a["cx"] - c[0]["cx"]) * 0.7, (a["cy"] - c[0]["cy"]) * 0.7)
            dz = abs(a["cz"] - c[0]["cz"])
            if d <= CLUSTER_MM * 1.5 and dz <= CLUSTER_MM:
                c.append(a); placed = True; break
        if not placed:
            clusters.append([a])
    return clusters


def main():
    zp = LIDC / "lidc_xml.zip"
    if not zp.exists():
        print(f"missing {zp} - download LIDC-XML-only.zip first"); return 1
    z = zipfile.ZipFile(zp)
    files = [f for f in z.namelist() if f.lower().endswith(".xml")]
    print(f"parsing {len(files)} XML reading files ...")

    by_series: dict[str, list[dict]] = defaultdict(list)
    bad = 0
    for f in files:
        s, anns = parse_xml(z.read(f))
        if s is None:
            bad += 1; continue
        by_series[s].extend(anns)
    print(f"series with annotations: {len(by_series)}   unparseable files: {bad}")

    series_info = {}
    tot = {"benign": 0, "malignant": 0, "indeterminate": 0}
    for s, anns in by_series.items():
        nods = []
        for c in cluster(anns):
            ratings = [a["malignancy"] for a in c]
            m = float(np.mean(ratings))
            lab = "benign" if m < 3 else ("malignant" if m > 3 else "indeterminate")
            # union of all readers' boxes for this nodule, per z-slice
            rois = [r for a in c for r in a["rois"]]
            nods.append({"label": lab, "mean_malignancy": round(m, 2),
                         "n_readers": len(ratings), "ratings": ratings,
                         "cz": float(np.mean([a["cz"] for a in c])),
                         "rois": rois})
            tot[lab] += 1
        series_info[s] = {
            "n_nodules": len(nods),
            "n_benign": sum(1 for n in nods if n["label"] == "benign"),
            "n_malignant": sum(1 for n in nods if n["label"] == "malignant"),
            "n_indeterminate": sum(1 for n in nods if n["label"] == "indeterminate"),
            "nodules": nods}

    print(f"\nnodule clusters: benign={tot['benign']} malignant={tot['malignant']} "
          f"indeterminate(excluded)={tot['indeterminate']}")

    # rank series by benign yield -- that is the class we are short of
    cat = json.load(open(LIDC / "lidc_series.json"))
    have = {c["SeriesInstanceUID"]: c for c in cat}
    ranked = []
    for s, info in series_info.items():
        if s not in have:
            continue
        c = have[s]
        ranked.append({"SeriesInstanceUID": s, "PatientID": c["PatientID"],
                       "size_mb": round(float(c.get("FileSize", 0)) / 1e6, 1),
                       "image_count": int(c.get("ImageCount", 0)),
                       "manufacturer": c.get("Manufacturer", ""),
                       **{k: v for k, v in info.items() if k != "nodules"}})
    ranked.sort(key=lambda r: (-r["n_benign"], -r["n_malignant"], r["size_mb"]))

    print(f"\nseries matched to the download catalogue: {len(ranked)}")
    print("top 10 by benign yield:")
    for r in ranked[:10]:
        print(f"  {r['PatientID']}  benign={r['n_benign']:2d} malignant={r['n_malignant']:2d} "
              f"{r['size_mb']:6.1f} MB  {r['image_count']} slices")

    for n in (50, 100, 150):
        sel = ranked[:n]
        print(f"  top {n:3d} series -> benign nodules {sum(r['n_benign'] for r in sel):4d} | "
              f"malignant {sum(r['n_malignant'] for r in sel):4d} | "
              f"{sum(r['size_mb'] for r in sel)/1000:.1f} GB")

    json.dump(series_info, open(LIDC / "lidc_annotations.json", "w"))
    json.dump(ranked, open(LIDC / "lidc_series_ranked.json", "w"), indent=1)
    print(f"\nwrote lidc_annotations.json + lidc_series_ranked.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
