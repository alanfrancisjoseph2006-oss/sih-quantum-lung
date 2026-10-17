"""
Phase 13c - turn downloaded LIDC DICOM series into labelled 2D slices.

EXPERIMENTAL DESIGN (this is the part that keeps the results honest)
--------------------------------------------------------------------
LIDC slices go into the TRAINING SET ONLY. The validation and test splits stay
100% IQ-OTH/NCCD, byte-identical to the ones used for every earlier result.

That matters for two reasons:
  1. Results remain directly comparable to the pre-LIDC numbers.
  2. It removes the incentive to learn a dataset shortcut. If the model learned
     "looks like LIDC -> benign", that knowledge is worthless on a pure IQ-OTH
     test set, so any measured improvement on that set is a real improvement.

LABELLING
  - Slice label comes from the nodules annotated on it.
  - If a slice carries both benign and malignant nodules, it is labelled
    MALIGNANT (worst-case), which is the clinically safe direction.
  - Indeterminate nodules (mean malignancy exactly 3) are excluded, and any
    slice whose only annotation is indeterminate is dropped entirely.
  - NO "normal" slices are taken from LIDC. Every LIDC patient was scanned
    because of suspected pathology, so an unannotated slice is not evidence of
    a normal lung. Inventing normals here would be fabricating labels.

PREPROCESSING
  Real DICOM means real Hounsfield Units, so unlike IQ-OTH we can apply a true
  lung window (C -600 / W 1500) before the shared 8-bit pipeline. This is the
  correct treatment and is documented as a deliberate difference.
"""
from __future__ import annotations
import argparse, csv, io, json, sys, zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA, REPORTS, SEED

LIDC = DATA / "external" / "lidc"
ZIPS = LIDC / "zips"
OUT = LIDC / "slices"
LUNG_C, LUNG_W = -600.0, 1500.0
MAX_SLICES_PER_NODULE = 3      # avoid one big nodule dominating the training set


def window_hu(hu: np.ndarray, c: float = LUNG_C, w: float = LUNG_W) -> np.ndarray:
    lo, hi = c - w / 2, c + w / 2
    return (np.clip(hu, lo, hi) - lo) / (hi - lo) * 255.0


def read_series(zp: Path):
    """Read every DICOM in a series zip -> {SOPInstanceUID: (hu_array, z)}."""
    import pydicom
    out, zpos = {}, {}
    try:
        z = zipfile.ZipFile(zp)
    except zipfile.BadZipFile:
        return out, zpos
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
            out[sop] = hu
            try:
                zpos[sop] = float(ds.ImagePositionPatient[2])
            except Exception:
                zpos[sop] = float(getattr(ds, "SliceLocation", 0.0) or 0.0)
        except Exception:
            continue
    return out, zpos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-nodule", type=int, default=MAX_SLICES_PER_NODULE)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ann = json.load(open(LIDC / "lidc_annotations.json"))
    sel = json.load(open(LIDC / "lidc_selected.json"))
    by_uid = {r["SeriesInstanceUID"]: r for r in sel}

    zips = sorted(ZIPS.glob("*.zip"))
    print(f"series zips found: {len(zips)}")
    if not zips:
        print("nothing downloaded yet - run scripts/10_lidc_download.py"); return 1

    rows, stats = [], defaultdict(int)
    for zi, zp in enumerate(zips, 1):
        uid_tail = zp.stem.split("__")[-1]
        match = [u for u in by_uid if u.endswith(uid_tail)]
        if not match:
            stats["series_no_match"] += 1; continue
        uid = match[0]
        pid = by_uid[uid]["PatientID"]
        info = ann.get(uid)
        if not info:
            stats["series_no_annotations"] += 1; continue

        slices, zpos = read_series(zp)
        if not slices:
            stats["series_unreadable"] += 1
            print(f"[{zi}/{len(zips)}] {pid}: unreadable"); continue

        # Collect, per SOP UID, which labels and boxes appear on that slice.
        per_slice = defaultdict(lambda: {"labels": set(), "boxes": []})
        for nod in info["nodules"]:
            if nod["label"] == "indeterminate":
                stats["nodules_indeterminate"] += 1
                continue
            # take the ROIs closest to the nodule centre, capped
            rois = sorted(nod["rois"], key=lambda r: abs(r["z"] - nod["cz"]))
            seen_sop = []
            for r in rois:
                if r["sop"] in seen_sop:
                    continue
                seen_sop.append(r["sop"])
                if len(seen_sop) > args.max_per_nodule:
                    break
                if r["sop"] not in slices:
                    continue
                per_slice[r["sop"]]["labels"].add(nod["label"])
                per_slice[r["sop"]]["boxes"].append(
                    {"x1": r["x1"], "y1": r["y1"], "x2": r["x2"], "y2": r["y2"],
                     "label": nod["label"], "mean_malignancy": nod["mean_malignancy"]})
            stats[f"nodules_{nod['label']}"] += 1

        n_written = 0
        for sop, d in per_slice.items():
            if not d["labels"]:
                continue
            # worst-case labelling: malignant wins on a mixed slice
            label = "malignant" if "malignant" in d["labels"] else "benign"
            if len(d["labels"]) > 1:
                stats["slices_mixed_labelled_malignant"] += 1
            img8 = window_hu(slices[sop]).astype(np.uint8)
            fname = f"{pid}_{sop[-12:]}_{label}.png"
            sub = OUT / label
            sub.mkdir(exist_ok=True)
            cv2.imwrite(str(sub / fname), img8)
            rows.append({"file": fname, "path": str(sub / fname), "label": label,
                         "patient_id": pid, "series_uid": uid, "sop_uid": sop,
                         "z": round(zpos.get(sop, 0.0), 2),
                         "n_boxes": len(d["boxes"]),
                         "boxes": json.dumps(d["boxes"])})
            n_written += 1
            stats[f"slices_{label}"] += 1
        print(f"[{zi}/{len(zips)}] {pid}: {n_written} slices "
              f"(benign nod {info['n_benign']}, malignant nod {info['n_malignant']})",
              flush=True)

    with open(LIDC / "lidc_slices.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                           ["file", "path", "label", "patient_id", "series_uid",
                            "sop_uid", "z", "n_boxes", "boxes"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== EXTRACTION SUMMARY ===")
    for k in sorted(stats):
        print(f"  {k:38s} {stats[k]}")
    print(f"  {'total slices written':38s} {len(rows)}")
    print(f"  {'unique patients':38s} {len({r['patient_id'] for r in rows})}")
    json.dump(dict(stats), open(REPORTS / "lidc_extraction_stats.json", "w"), indent=1)
    print(f"\nwrote {LIDC/'lidc_slices.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
