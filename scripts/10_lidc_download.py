"""
Phase 13b - download a targeted LIDC-IDRI subset from the TCIA public API.

SELECTION STRATEGY (important)
------------------------------
Ranking purely by benign yield would give ~347 benign vs ~37 malignant nodules.
Every LIDC image added would then be benign, and the model could learn
"this image looks like LIDC -> benign" instead of learning lesion morphology -
a dataset shortcut worse than the imbalance we set out to fix.

So the subset is deliberately built from BOTH ends:
    - N_BENIGN series ranked by benign-nodule count
    - N_MALIGNANT series ranked by malignant-nodule count
This keeps both classes present in the new domain, so dataset identity carries
little information about the label. `scripts/12_*` verifies that empirically
with a dataset-identity probe.

Resumable: already-downloaded zips are skipped, so the run can be interrupted.
"""
from __future__ import annotations
import argparse, json, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import DATA

LIDC = DATA / "external" / "lidc"
ZIPS = LIDC / "zips"
API = ("https://services.cancerimagingarchive.net/nbia-api/services/v1/"
       "getImage?SeriesInstanceUID=")


def select(ranked, n_benign: int, n_malignant: int):
    """Take the top series by benign yield and by malignant yield, without overlap."""
    chosen, seen = [], set()
    for r in sorted(ranked, key=lambda r: (-r["n_benign"], r["size_mb"])):
        if len(chosen) >= n_benign:
            break
        if r["n_benign"] > 0 and r["SeriesInstanceUID"] not in seen:
            chosen.append(r); seen.add(r["SeriesInstanceUID"])
    mal = 0
    for r in sorted(ranked, key=lambda r: (-r["n_malignant"], r["size_mb"])):
        if mal >= n_malignant:
            break
        if r["n_malignant"] > 0 and r["SeriesInstanceUID"] not in seen:
            chosen.append(r); seen.add(r["SeriesInstanceUID"]); mal += 1
    return chosen


def fetch(r, retries: int = 3) -> tuple[str, int, str]:
    """Download one series zip. Returns (patient, bytes, status)."""
    out = ZIPS / f"{r['PatientID']}__{r['SeriesInstanceUID'][-16:]}.zip"
    if out.exists() and out.stat().st_size > 100_000:
        return r["PatientID"], out.stat().st_size, "cached"
    last = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API + r["SeriesInstanceUID"],
                                         headers={"User-Agent": "sih-quantumlung/1.0"})
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = resp.read()
            if len(data) < 100_000:
                last = f"too small ({len(data)}B)"
                continue
            tmp = out.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(out)
            return r["PatientID"], len(data), "ok"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 + 3 * attempt)
    return r["PatientID"], 0, f"FAILED {last}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-benign", type=int, default=32)
    ap.add_argument("--n-malignant", type=int, default=18)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    ZIPS.mkdir(parents=True, exist_ok=True)
    ranked = json.load(open(LIDC / "lidc_series_ranked.json"))
    sel = select(ranked, args.n_benign, args.n_malignant)

    gb = sum(r["size_mb"] for r in sel) / 1000
    nb = sum(r["n_benign"] for r in sel)
    nm = sum(r["n_malignant"] for r in sel)
    print(f"selected {len(sel)} series | {gb:.1f} GB | "
          f"benign nodules {nb} | malignant nodules {nm}")
    json.dump(sel, open(LIDC / "lidc_selected.json", "w"), indent=1)

    done = sum(1 for r in sel
               if (ZIPS / f"{r['PatientID']}__{r['SeriesInstanceUID'][-16:]}.zip").exists())
    print(f"already present: {done}/{len(sel)}   starting with {args.workers} workers\n")

    t0, total, okc, fail = time.time(), 0, 0, []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(fetch, r): r for r in sel}
        for i, f in enumerate(as_completed(futs), 1):
            pid, n, status = f.result()
            total += n
            if status.startswith("FAILED"):
                fail.append((pid, status))
            else:
                okc += 1
            el = time.time() - t0
            rate = total / max(el, 1) / 1e3
            eta = (len(sel) - i) * (el / max(i, 1)) / 60
            print(f"[{i:3d}/{len(sel)}] {pid:16s} {n/1e6:6.1f} MB {status:8s} | "
                  f"{total/1e9:.2f} GB {rate:.0f} KB/s ETA {eta:.0f} min", flush=True)

    print(f"\ncomplete: {okc}/{len(sel)} series, {total/1e9:.2f} GB in "
          f"{(time.time()-t0)/60:.1f} min")
    if fail:
        print(f"failed ({len(fail)}):")
        for p, s in fail[:10]:
            print("  ", p, s[:100])
    return 0 if okc else 1


if __name__ == "__main__":
    sys.exit(main())
