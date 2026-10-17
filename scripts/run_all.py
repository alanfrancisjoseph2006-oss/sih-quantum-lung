"""
Run the entire pipeline end to end: audit -> clean -> train -> select -> validate.
Each stage is skipped if its output already exists, unless --force is passed.
"""
import argparse, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

STAGES = [
    ("01_audit_dataset.py",        [], "reports/audit_raw.json"),
    ("02_audit_leakage.py",        [], "reports/deeplesion_boxes.json"),
    ("03_clean_and_split.py",      [], "data/splits/splits.csv"),
    ("04_train_cnn.py",            ["--backbone", "efficientnet_b3"],
                                   "models/cnn_efficientnet_b3.pt"),
    ("04_train_cnn.py",            ["--backbone", "densenet121", "--tag", "densenet121"],
                                   "models/cnn_densenet121.pt"),
    ("05_train_branches.py",       ["--tag", "efficientnet_b3"],
                                   "models/branches_efficientnet_b3.pkl"),
    ("05_train_branches.py",       ["--tag", "densenet121"],
                                   "models/branches_densenet121.pkl"),
    ("06_model_selection.py",      [], "reports/model_comparison.json"),
    ("07_validate_localization.py", [], "reports/localization_validation.json"),
    ("08_improve_benign.py",        [], "reports/experiments/benign_improvement.json"),
]

# Optional external-data stages. They need ~3 GB of download, so they are only
# run with --with-external. See decisions D21 (classification: negative result)
# and D22 (segmentation: deployed).
EXTERNAL_STAGES = [
    ("09_lidc_parse_annotations.py", [], "data/external/lidc/lidc_series_ranked.json"),
    ("10_lidc_download.py",          [], "data/external/lidc/lidc_selected.json"),
    ("11_lidc_extract_slices.py",    [], "data/external/lidc/lidc_slices.csv"),
    ("12_merge_and_probe.py",        [], "data/splits/splits_merged.csv"),
    ("14_build_lidc_masks.py",       [], "data/external/lidc/lidc_masks.csv"),
    ("15_train_unet.py",             [], "models/unet_nodule.pt"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-run stages even if outputs exist")
    ap.add_argument("--with-external", action="store_true",
                    help="also run the LIDC-IDRI stages (~3 GB download, ~25 min, "
                         "plus ~20 min to train the segmentation U-Net)")
    args = ap.parse_args()

    stages = STAGES + (EXTERNAL_STAGES if args.with_external else [])

    t0 = time.time()
    for i, (script, extra, out) in enumerate(stages, 1):
        target = ROOT / out
        label = f"{script} {' '.join(extra)}".strip()
        if target.exists() and not args.force:
            print(f"[{i}/{len(stages)}] SKIP  {label}  (found {out})")
            continue
        print(f"\n[{i}/{len(STAGES)}] RUN   {label}")
        print("-" * 66)
        r = subprocess.run([PY, str(ROOT / "scripts" / script), *extra], cwd=str(ROOT))
        if r.returncode != 0:
            print(f"\nFAILED at {label} (exit {r.returncode})")
            return r.returncode
    print(f"\nAll stages complete in {(time.time()-t0)/60:.1f} min.")
    print("Start the web app with:  python scripts/serve.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
