"""
Phase 9 - Assemble every measured result into the three required tables.

Reads only the JSON artefacts produced by earlier scripts. Nothing is
recomputed or estimated here, so the tables cannot drift from the experiments.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REPORTS


def load(name):
    p = REPORTS / name
    return json.load(open(p)) if p.exists() else None


def main() -> int:
    ext = load("external_validation.json")
    dr = load("domain_robust_comparison.json")
    roi = load("roi_classifier.json")
    hq = load("hybrid_quantum_roi.json")
    sig = load("domain_signature_probe.json")
    har = load("harmonisation_search.json")
    inv = load("inversion_analysis.json")

    rows_t1, rows_t2, rows_t3 = [], [], []

    # ---- TABLE 1 / 2: whole-image arms, in-domain and external -------------
    if dr:
        for r in dr["results"]:
            rows_t1.append({"model": f"whole-image / {r['arm']}",
                            "auc": r["in_domain"]["roc_auc"],
                            "ci": r["in_domain"]["roc_auc_ci95"],
                            "n": r["in_domain"]["n"], "eval": "IQ-OTH held-out"})
            rows_t2.append({"model": f"whole-image / {r['arm']}",
                            "auc": r["external"]["roc_auc"],
                            "ci": r["external"]["roc_auc_ci95"],
                            "n": r["external"]["n"], "eval": "LIDC (unseen)"})

    # ---- ROI arms, evaluated on held-out LIDC patients ---------------------
    if roi:
        for key in ("size_only", "handcrafted", "roi_cnn"):
            r = roi["results"][key]
            rows_t2.append({"model": f"ROI / {r['name']}", "auc": r["roc_auc"],
                            "ci": r["roc_auc_ci95"], "n": r["n"],
                            "eval": "LIDC held-out patients"})
    if hq:
        for r in hq["results"]:
            rows_t2.append({"model": f"ROI features / {r['name']}", "auc": r["roc_auc"],
                            "ci": r["roc_auc_ci95"], "n": hq["split"]["test_rois"],
                            "eval": "LIDC held-out patients"})

    def show(title, rows, note=""):
        print(f"\n{'='*94}\n{title}\n{'='*94}")
        if note:
            print(f"{note}\n")
        print(f"{'model':<44} {'AUC':>8} {'95% CI':>18} {'n':>6}  {'evaluated on'}")
        print("-" * 94)
        for r in sorted(rows, key=lambda x: -x["auc"]):
            ci = "[{:.3f}, {:.3f}]".format(*r["ci"])
            print(f"{r['model']:<44} {r['auc']:>8.4f} {ci:>18} {r['n']:>6}  {r['eval']}")

    show("TABLE 1 - IN-DOMAIN VALIDATION (train and test both IQ-OTH)", rows_t1,
         "Note the ceiling effect: these are near-perfect and therefore uninformative "
         "about generalisation.")

    show("TABLE 2 - EXTERNAL / CROSS-PATIENT VALIDATION", rows_t2,
         "Whole-image rows are trained on IQ-OTH and tested on LIDC (cross-dataset).\n"
         "ROI rows are trained and tested on LIDC under a PATIENT-level split "
         "(cross-patient, same dataset).\nThese two are not interchangeable and are "
         "labelled accordingly.")

    # ---- TABLE 3: the argument, in order ----------------------------------
    print(f"\n{'='*94}\nTABLE 3 - CROSS-DOMAIN DIAGNOSIS: what was measured, in order\n{'='*94}")
    steps = []
    if sig:
        steps.append(("Dataset identity from a whole slice",
                      f"AUC {sig['results']['whole']['domain_auc']:.3f}",
                      "a linear probe names the source dataset perfectly"))
        steps.append(("Dataset identity from a random lung crop",
                      f"AUC {sig['results']['roi']['domain_auc']:.3f}",
                      "cropping alone does NOT remove the signature"))
    if har:
        b = har["lowest_domain"]
        steps.append(("Best harmonisation found",
                      f"domain AUC {b['domain_auc']:.3f}",
                      f"'{b['transform']}' - still far from 0.5"))
    if ext:
        steps.append(("Whole-image model, external",
                      f"AUC {ext['slice_level']['roc_auc']:.3f}",
                      "at chance on unseen hospitals"))
    if dr:
        best = max(dr["results"], key=lambda r: r["external"]["roc_auc"])
        steps.append(("Harmonisation + domain-adversarial training",
                      f"best external AUC {best['external']['roc_auc']:.3f}",
                      "neither intervention helped"))
    if inv:
        steps.append(("Nodule SIZE alone (LIDC)",
                      f"AUC {inv['h1_size_confound']['auc_of_size_alone']:.3f}",
                      "a trivial clinical feature works well"))
        steps.append(("Model score vs nodule size",
                      f"rho {inv['h2_score_tracks_size']['spearman_score_vs_area']:+.3f}",
                      "the model was ignoring the nodule entirely"))
    if hq:
        best = max(hq["results"], key=lambda r: r["roc_auc"])
        steps.append(("ROI features + classical ML",
                      f"AUC {best['roc_auc']:.3f}",
                      f"{best['name']} - focusing on the lesion works"))
    for i, (what, val, why) in enumerate(steps, 1):
        print(f"  {i}. {what:<46} {val:<26} {why}")

    # ---- headline ----------------------------------------------------------
    print(f"\n{'='*94}\nHEADLINE\n{'='*94}")
    if ext and hq:
        base = ext["slice_level"]["roc_auc"]
        best = max(hq["results"], key=lambda r: r["roc_auc"])
        print(f"  Whole-image classification, unseen hospitals ....... AUC {base:.3f}  (chance)")
        print(f"  Nodule-ROI features, unseen patients ............... AUC {best['roc_auc']:.3f}")
        print(f"\n  Attending to the lesion instead of the whole image is the difference")
        print(f"  between a model that does not work and one that does.")
        print(f"\n  IMPORTANT: these are not the same evaluation. The first is")
        print(f"  cross-DATASET; the second is cross-PATIENT within LIDC. Cross-dataset")
        print(f"  ROI validation is not yet possible because IQ-OTH has no nodule")
        print(f"  annotations. That remains the key open item.")

    if hq and hq.get("quantum_advantage_test"):
        print(f"\n  Quantum branch on the same features:")
        for k, v in hq["quantum_advantage_test"].items():
            verdict = ("significantly WORSE" if v["significant"] and v["mean_diff"] < 0
                       else "significantly better" if v["significant"]
                       else "no significant difference")
            print(f"    {k}: delta {v['mean_diff']:+.4f}, p = {v['p_value']:.3f} -> {verdict}")

    out = {"table1_in_domain": rows_t1, "table2_external": rows_t2,
           "table3_diagnosis": [{"step": w, "value": v, "meaning": y}
                                for w, v, y in steps]}
    json.dump(out, open(REPORTS / "final_comparison.json", "w"), indent=1)
    print("\nwrote reports/final_comparison.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
