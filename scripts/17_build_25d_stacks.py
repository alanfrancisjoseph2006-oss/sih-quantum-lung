"""
Phase 15a - build 2.5D inputs (adjacent-slice stacks) to attack the detection failure.

WHY
---
D23 measured that the segmenter localises accurately when it fires (centre within
~0.17 mm) but detects only 55% of nodules, and that the misses carry essentially
zero response (median peak probability 0.0003) rather than a weak one. Threshold
tuning moved detection by <3 points, so the information simply is not reaching
the model.

The most likely reason is that it sees ONE 2D slice. In a single axial slice a
small nodule and a vessel seen end-on are nearly identical discs. What separates
them is continuity through the volume: a vessel persists as a tube across
adjacent slices, a nodule does not. Radiologists resolve this by scrolling.

So we give the model the same cue: stack slice z-1, z, z+1 as three channels.
This is standard "2.5D" and is far cheaper than a full 3D U-Net, which would not
fit this dataset size or the 6 GB GPU.

The mask is unchanged - it still belongs to the CENTRE slice only.

Also raises the per-nodule slice cap from 3 to 5, since more annotated slices per
nodule is free extra supervision now that we are re-reading the volumes anyway.
"""
from __future__ import annotations
import argparse, csv, io, json, sys, zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA, REPORTS

LIDC = DATA / "external" / "lidc"
ZIPS = LIDC / "zips"
OUT = LIDC / "stacks25d"
LUNG_C, LUNG_W = -600.0, 1500.0


def window_hu(hu: np.ndarray) -> np.ndarray:
    lo, hi = LUNG_C - LUNG_W / 2, LUNG_C + LUNG_W / 2
    return (np.clip(hu, lo, hi) - lo) / (hi - lo) * 255.0


def read_series_sorted(zp: Path):
    """-> (sop_order list sorted by z, {sop: hu array}, {sop: z})"""
    import pydicom
    imgs, zs = {}, {}
    try:
        z = zipfile.ZipFile(zp)
    except zipfile.BadZipFile:
        return [], {}, {}
    for name in z.namelist():
        if not name.lower().endswith(".dcm"):
            continue
        try:
            ds = pydicom.dcmread(io.BytesIO(z.read(name)), force=True)
            if not hasattr(ds, "pixel_array"):
                continue
            arr = ds.pixel_array.astype(np.float32)
            if arr.ndim != 2:
                continue
            slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
            inter = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
            hu = arr * slope + inter
            if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                hu = hu.max() - hu
            sop = str(getattr(ds, "SOPInstanceUID", name))
            imgs[sop] = hu
            try:
                zs[sop] = float(ds.ImagePositionPatient[2])
            except Exception:
                zs[sop] = float(getattr(ds, "SliceLocation", 0.0) or 0.0)
        except Exception:
            continue
    order = sorted(imgs.keys(), key=lambda s: zs.get(s, 0.0))
    return order, imgs, zs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=int, default=1,
                    help="how many slices away the neighbour channels sit")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    masks = list(csv.DictReader(open(LIDC / "lidc_masks.csv", newline="")))
    by_series = defaultdict(list)
    for r in masks:
        by_series[r["series_uid"]].append(r)
    print(f"masked slices: {len(masks)} across {len(by_series)} series")

    zips = {}
    for zp in ZIPS.glob("*.zip"):
        zips[zp.stem.split("__")[-1]] = zp

    out_rows, stats = [], defaultdict(int)
    for si, (uid, rows) in enumerate(sorted(by_series.items()), 1):
        zp = next((v for k, v in zips.items() if uid.endswith(k)), None)
        if zp is None:
            stats["series_zip_missing"] += 1
            continue
        order, imgs, zs = read_series_sorted(zp)
        if not order:
            stats["series_unreadable"] += 1
            continue
        idx = {s: i for i, s in enumerate(order)}

        for r in rows:
            sop = r["sop_uid"]
            if sop not in idx:
                stats["centre_slice_missing"] += 1
                continue
            i = idx[sop]
            lo = order[max(0, i - args.gap)]
            hi = order[min(len(order) - 1, i + args.gap)]
            if lo == sop or hi == sop:
                stats["edge_slice_neighbour_duplicated"] += 1

            stack = np.stack([window_hu(imgs[lo]),
                              window_hu(imgs[sop]),
                              window_hu(imgs[hi])], axis=-1).astype(np.uint8)
            # BGR order on disk; channel 1 (green) is always the centre slice.
            fn = f"{Path(r['file']).stem}_25d.png"
            fp = OUT / fn
            cv2.imwrite(str(fp), stack)
            out_rows.append({**r, "stack_path": str(fp)})
            stats["stacks_written"] += 1
        if si % 10 == 0:
            print(f"  [{si}/{len(by_series)}] series processed", flush=True)

    with open(LIDC / "lidc_stacks25d.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    print("\n=== 2.5D STACK SUMMARY ===")
    for k in sorted(stats):
        print(f"  {k:34s} {stats[k]}")
    print(f"  {'unique patients':34s} {len({r['patient_id'] for r in out_rows})}")
    json.dump({k: int(v) for k, v in stats.items()},
              open(REPORTS / "lidc_25d_stats.json", "w"), indent=1)
    print(f"\nwrote {LIDC/'lidc_stacks25d.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
