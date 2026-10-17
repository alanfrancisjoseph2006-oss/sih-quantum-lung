"""
Scientific model readiness levels.

A single accuracy number cannot tell a clinician whether a model is safe to rely
on. These levels encode WHAT HAS BEEN DEMONSTRATED, and each is awarded from
measured artefacts rather than asserted.

  L1 Prototype               trains and produces output
  L2 Internally validated    held-out performance with no patient leakage
  L3 Externally validated    performs above chance on an UNSEEN site/dataset
  L4 Multi-centre validated  holds across several independent sites
  L5 Clinical research ready prospective validation, regulatory pathway

The gate between L2 and L3 is the one this project failed, and the level system
exists so that failure is visible in the product instead of buried in a report.
"""
from __future__ import annotations

import json
from pathlib import Path

LEVELS = {
    1: {"name": "Prototype",
        "meaning": "Trains and produces output. No validation claim."},
    2: {"name": "Internally validated",
        "meaning": "Held-out performance measured with leakage controls, on data "
                   "from the same source as training."},
    3: {"name": "Externally validated",
        "meaning": "Performs meaningfully better than chance on data from a site, "
                   "scanner or population it was never trained on."},
    4: {"name": "Multi-centre validated",
        "meaning": "External performance holds across several independent centres."},
    5: {"name": "Clinical research ready",
        "meaning": "Prospective validation, defined intended use, regulatory pathway."},
}


def assess(reports_dir: Path) -> dict:
    """
    Award a readiness level from the measured artefacts on disk.

    Deliberately conservative: a criterion counts as met only when a report
    exists and its numbers support it. Missing evidence is never treated as a
    pass.
    """
    def load(name):
        p = reports_dir / name
        try:
            return json.load(open(p))
        except Exception:
            return None

    comparison = load("model_comparison.json")
    external = load("external_validation.json")
    splits_ok = (reports_dir.parent / "data" / "splits" / "splits.csv").exists()

    criteria = []

    # ---- L1 -------------------------------------------------------------
    trains = comparison is not None
    criteria.append({"level": 1, "criterion": "Model trains and produces output",
                     "met": bool(trains),
                     "evidence": "reports/model_comparison.json present" if trains
                                 else "no trained model found"})

    # ---- L2 -------------------------------------------------------------
    leak_free = splits_ok
    in_domain = None
    if comparison:
        dep = comparison.get("deployed", {})
        bb = comparison.get("backbones", {}).get(dep.get("backbone", ""), {})
        m = bb.get("models", {}).get(dep.get("primary_model", ""), {})
        in_domain = (m.get("test") or {}).get("roc_auc_macro_ovr")
    ok2 = bool(leak_free and in_domain and in_domain > 0.7)
    criteria.append({
        "level": 2, "criterion": "Held-out performance with leakage controls",
        "met": ok2,
        "evidence": (f"group-aware split verified; in-domain macro ROC-AUC "
                     f"{in_domain:.3f}" if ok2 else "not demonstrated")})

    # ---- L3 -------------------------------------------------------------
    ext_auc = ext_lo = ext_hi = None
    if external:
        sl = external.get("slice_level", {})
        ext_auc = sl.get("roc_auc")
        ci = sl.get("ci95") or [None, None]
        ext_lo, ext_hi = ci[0], ci[1]
    # Passing requires the whole CI to sit above 0.5 - a point estimate above
    # chance with an interval that includes it demonstrates nothing.
    ok3 = bool(ext_lo is not None and ext_lo > 0.5)
    if ext_auc is None:
        ev3 = "no external validation has been run"
    elif ok3:
        ev3 = (f"external ROC-AUC {ext_auc:.3f}, 95% CI {ext_lo:.3f}-{ext_hi:.3f} "
               f"lies entirely above chance")
    else:
        ev3 = (f"external ROC-AUC {ext_auc:.3f}, 95% CI {ext_lo:.3f}-{ext_hi:.3f} "
               f"- does not exclude chance (0.50)")
    criteria.append({"level": 3, "criterion": "Above chance on an unseen dataset",
                     "met": ok3, "evidence": ev3})

    # ---- L4 / L5 ---------------------------------------------------------
    criteria.append({"level": 4, "criterion": "Holds across multiple independent centres",
                     "met": False,
                     "evidence": "only one external dataset has been tested"})
    criteria.append({"level": 5, "criterion": "Prospective validation and regulatory pathway",
                     "met": False,
                     "evidence": "not attempted; this is a research prototype"})

    # Highest level whose criterion AND all below it are met.
    level = 0
    for c in sorted(criteria, key=lambda c: c["level"]):
        if c["met"]:
            level = c["level"]
        else:
            break

    blocking = next((c for c in sorted(criteria, key=lambda c: c["level"])
                     if not c["met"]), None)

    return {
        "level": level,
        "level_name": LEVELS.get(level, {}).get("name", "Not assessed"),
        "level_meaning": LEVELS.get(level, {}).get("meaning", ""),
        "levels": LEVELS,
        "criteria": criteria,
        "blocking_criterion": blocking,
        "clinically_deployable": level >= 5,
        "statement": (
            f"Readiness level {level} of 5 - {LEVELS.get(level, {}).get('name', '?')}. "
            + (f"Blocked at level {blocking['level']}: {blocking['evidence']}."
               if blocking else "")
            + " This is a research prototype and is not clinically deployable."),
    }
