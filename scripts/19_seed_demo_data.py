"""
Seed synthetic demonstration data.

Every row created here is flagged `is_demo=True` at the database level, so the
UI can label it and no synthetic record can be mistaken for a real patient.

Patient identifiers are opaque (DEMO-0001 style). No names, dates of birth or
contact details are generated, because the schema has nowhere to put them.

Images are drawn from the real IQ-OTH/NCCD set so the analyses are genuine model
outputs - the *patients* are fictional, the *inference* is real. That keeps the
dashboard honest: every risk score shown was actually computed.
"""
from __future__ import annotations

import argparse
import glob
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.db.models import FollowUp, FollowUpStatus
from api.db.session import SessionLocal, init_db
from api.services import analysis_service as svc
from src.config import IQ_DIR

SEED = 42


def pick_images(n_per_class: int, rng: random.Random) -> list[tuple[str, str]]:
    out = []
    for folder, label in (("Bengin cases", "benign"),
                          ("Malignant cases", "malignant"),
                          ("Normal cases", "normal")):
        files = sorted(glob.glob(str(IQ_DIR / folder / "*.jpg")))
        if not files:
            continue
        for f in rng.sample(files, min(n_per_class, len(files))):
            out.append((f, label))
    rng.shuffle(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=14)
    ap.add_argument("--per-class", type=int, default=7,
                    help="images sampled per class to distribute across patients")
    ap.add_argument("--reset", action="store_true", help="delete existing demo rows first")
    args = ap.parse_args()

    init_db()
    rng = random.Random(SEED)
    db = SessionLocal()

    if args.reset:
        from api.db.models import (Analysis, AuditLog, DoctorFeedback, Finding,
                                   Patient, Report, Scan)
        n = 0
        for pat in db.query(Patient).filter(Patient.is_demo.is_(True)).all():
            db.delete(pat); n += 1
        db.query(AuditLog).delete()
        db.commit()
        print(f"removed {n} demo patients and their cascaded records")

    from src.inference.engine import get_engine
    engine = get_engine()
    actor = svc.get_or_create_default_user(db)

    images = pick_images(args.per_class, rng)
    if not images:
        print("No source images found - check IQ_DIR in src/config.py")
        return 1
    print(f"seeding {args.patients} demo patients from {len(images)} real CT images")

    now = datetime.now(timezone.utc)
    created = 0
    for i in range(args.patients):
        ext_id = f"DEMO-{i + 1:04d}"
        patient = svc.get_or_create_patient(
            db, ext_id, age=rng.randint(41, 79),
            sex=rng.choice(["M", "F"]), is_demo=True)
        patient.is_demo = True

        # 1-3 scans per patient so timelines and comparison have real material
        n_scans = rng.choices([1, 2, 3], weights=[5, 3, 2])[0]
        for k in range(n_scans):
            path, _ = images[(i * 3 + k) % len(images)]
            with open(path, "rb") as fh:
                data = fh.read()
            result = engine.analyze(data, Path(path).name)

            from src.inference.clinical_info import CANCER_EDUCATION, clinical_package
            result["clinical_information"] = clinical_package(
                result["final_prediction"]["predicted_class"],
                result["final_prediction"]["confidence"])
            result["ai_limitations"] = CANCER_EDUCATION["ai_limitations"]

            analysis = svc.record_analysis(
                db, patient=patient, filename=Path(path).name, result=result,
                is_demo=True, actor=actor,
                acquired_at=now - timedelta(days=(n_scans - k) * rng.randint(45, 200)))
            # backdate so timelines are not all "today"
            analysis.created_at = now - timedelta(
                days=(n_scans - k - 1) * rng.randint(40, 190),
                hours=rng.randint(0, 20))
            analysis.is_demo = True
            db.commit()
            created += 1

        # A follow-up for some higher-risk patients, with a clinician-set date.
        last = analysis
        if last.risk_band in ("High", "Intermediate") and rng.random() < 0.7:
            db.add(FollowUp(
                patient_id=patient.id, analysis_id=last.id, risk_band=last.risk_band,
                last_scan_at=last.created_at,
                review_due=now + timedelta(days=rng.choice([-20, -5, 7, 21, 60, 120])),
                notes="Interval set by reviewing clinician (demonstration record).",
                set_by=actor.id, is_demo=True,
            ))
        db.commit()
        print(f"  {ext_id}: {n_scans} scan(s), latest risk {last.risk_band}", flush=True)

    svc.refresh_followup_statuses(db)
    stats = svc.overview_stats(db)
    print(f"\nseeded {created} analyses across {args.patients} demo patients")
    print(f"  high priority: {stats['high_priority']}  pending review: {stats['pending_review']}")
    print(f"  by risk: {stats['by_risk']}")
    print(f"  follow-ups due/overdue: {stats['follow_ups_due']}")
    print("\nAll rows flagged is_demo=True and surfaced as 'Demo Data' in the UI.")
    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
