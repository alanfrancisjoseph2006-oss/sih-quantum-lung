"""
Phase 14a - build pixel-level nodule masks from LIDC-IDRI radiologist contours.

The LIDC XML stores each reader's nodule outline as an <edgeMap> polygon in
image pixel coordinates, per slice. That is genuine pixel-level ground truth -
unlike IQ-OTH/NCCD, which has no masks at all and is why the platform has so far
refused to output segmentation.

CONSENSUS RULE
--------------
Up to 4 radiologists outline the same nodule and they disagree at the margins.
We rasterise every reader's polygon and keep pixels where **at least 2 readers
agree** (or all of them, when fewer than 2 outlined that slice). This is the
standard 50%-consensus convention and avoids training on one reader's opinion.

Masks are written aligned 1:1 with the slices produced by
scripts/11_lidc_extract_slices.py, matched on SOPInstanceUID.
"""
from __future__ import annotations
import csv, json, sys, zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA, REPORTS

LIDC = DATA / "external" / "lidc"
MASKS = LIDC / "masks"
MIN_READERS = 2


def strip_ns(t: str) -> str:
    return t.split("}")[-1] if "}" in t else t


def parse_polygons(raw: bytes):
    """-> {sop_uid: [ {pts, malignancy, reader} ]} for one XML reading file."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return {}
    out = defaultdict(list)
    for ri, sess in enumerate(e for e in root.iter() if strip_ns(e.tag) == "readingSession"):
        for nod in sess:
            if strip_ns(nod.tag) != "unblindedReadNodule":
                continue
            mal = None
            rois = []
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
                    sop, pts, incl = None, [], True
                    for r in child:
                        rt = strip_ns(r.tag)
                        if rt == "imageSOP_UID" and r.text:
                            sop = r.text.strip()
                        elif rt == "inclusion" and r.text:
                            incl = r.text.strip().upper() == "TRUE"
                        elif rt == "edgeMap":
                            x = y = None
                            for e in r:
                                if strip_ns(e.tag) == "xCoord" and e.text:
                                    x = int(float(e.text))
                                elif strip_ns(e.tag) == "yCoord" and e.text:
                                    y = int(float(e.text))
                            if x is not None and y is not None:
                                pts.append([x, y])
                    # `inclusion=FALSE` marks a hole (e.g. cavitation) - skipped,
                    # since holes are rare and mishandling them would corrupt masks.
                    if sop and len(pts) >= 3 and incl:
                        rois.append((sop, pts))
            if mal is None:
                continue                      # sub-3mm nodule, no rating -> skip
            for sop, pts in rois:
                out[sop].append({"pts": pts, "malignancy": mal, "reader": ri})
    return out


def main():
    MASKS.mkdir(parents=True, exist_ok=True)
    slices_csv = LIDC / "lidc_slices.csv"
    if not slices_csv.exists():
        print("run scripts/11_lidc_extract_slices.py first"); return 1
    rows = list(csv.DictReader(open(slices_csv, newline="")))
    want = {r["sop_uid"]: r for r in rows}
    print(f"slices needing masks: {len(want)}")

    z = zipfile.ZipFile(LIDC / "lidc_xml.zip")
    files = [f for f in z.namelist() if f.lower().endswith(".xml")]
    per_sop = defaultdict(list)
    for f in files:
        for sop, items in parse_polygons(z.read(f)).items():
            if sop in want:
                per_sop[sop].extend(items)
    print(f"slices with at least one contour: {len(per_sop)}")

    out_rows, stats = [], defaultdict(int)
    for sop, items in per_sop.items():
        row = want[sop]
        img = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            stats["image_missing"] += 1
            continue
        h, w = img.shape

        # Rasterise each reader separately, then vote.
        by_reader = defaultdict(list)
        for it in items:
            by_reader[it["reader"]].append(it["pts"])
        votes = np.zeros((h, w), np.uint8)
        for reader, polys in by_reader.items():
            m = np.zeros((h, w), np.uint8)
            for pts in polys:
                p = np.array(pts, np.int32).reshape(-1, 1, 2)
                p[:, :, 0] = np.clip(p[:, :, 0], 0, w - 1)
                p[:, :, 1] = np.clip(p[:, :, 1], 0, h - 1)
                cv2.fillPoly(m, [p], 1)
            votes += m
        need = min(MIN_READERS, len(by_reader))
        mask = ((votes >= need).astype(np.uint8)) * 255
        area = int((mask > 0).sum())
        if area < 4:
            stats["mask_too_small"] += 1
            continue

        mp = MASKS / f"{Path(row['file']).stem}_mask.png"
        cv2.imwrite(str(mp), mask)
        out_rows.append({**row, "mask_path": str(mp),
                         "mask_area_px": area,
                         "mask_area_frac": round(area / (h * w), 6),
                         "n_readers": len(by_reader)})
        stats["masks_written"] += 1
        stats[f"label_{row['label']}"] += 1

    with open(LIDC / "lidc_masks.csv", "w", newline="") as fh:
        w_ = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w_.writeheader(); w_.writerows(out_rows)

    areas = np.array([r["mask_area_frac"] for r in out_rows])
    print("\n=== MASK SUMMARY ===")
    for k in sorted(stats):
        print(f"  {k:24s} {stats[k]}")
    print(f"  {'unique patients':24s} {len({r['patient_id'] for r in out_rows})}")
    print(f"  nodule area fraction: median={np.median(areas)*100:.3f}% "
          f"p90={np.percentile(areas,90)*100:.3f}% max={areas.max()*100:.3f}%")
    print(f"  -> nodules occupy ~{np.median(areas)*100:.2f}% of the image: "
          f"this is a severe foreground/background imbalance, so the loss must "
          f"be Dice-based, not plain BCE.")
    json.dump({k: int(v) for k, v in stats.items()},
              open(REPORTS / "lidc_mask_stats.json", "w"), indent=1)
    print(f"\nwrote {LIDC/'lidc_masks.csv'} and {stats['masks_written']} mask PNGs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
