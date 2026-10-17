"""
Business logic for scans, analyses, reports and triage.

Sits between the HTTP routers and both the ML engine and the database. The
routers do transport and validation; this module owns the rules.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.db.models import (
    Analysis, AuditLog, DoctorFeedback, Finding, FollowUp, FollowUpStatus,
    Organization, Patient, Priority, Report, ReviewStatus, Scan, User, UserRole,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------- audit

def audit(db: Session, action: str, entity_type: str, entity_id: str,
          actor: User | None = None, **detail: Any) -> None:
    db.add(AuditLog(
        actor_id=actor.id if actor else None,
        actor_role=actor.role.value if actor else "system",
        action=action, entity_type=entity_type, entity_id=entity_id,
        detail_json=detail or {},
    ))


# ------------------------------------------------------------------ triage

def derive_priority(predicted_class: str, confidence: float, agreement: float) -> Priority:
    """
    Presentation-layer triage ordering, not a clinical algorithm.

    Deliberately simple and inspectable so a clinician can predict its
    behaviour. Low branch agreement escalates a 'normal' because disagreement
    between independently trained branches marks an ambiguous case - exactly
    the sort a human should see sooner.
    """
    if predicted_class == "malignant":
        return Priority.high
    if predicted_class == "benign":
        return Priority.attention
    if agreement < 0.75 or confidence < 0.70:
        return Priority.attention
    return Priority.routine


TRIAGE_RULE_DESCRIPTION = (
    "Malignant -> High priority. Benign -> Attention required. Normal with branch "
    "agreement below 75% or confidence below 70% -> Attention required, because "
    "disagreement between independently trained branches marks an ambiguous case. "
    "Everything else -> Routine. This is a transparent ordering rule for the work "
    "queue, not a clinical severity score."
)


# ------------------------------------------------------------- persistence

def get_or_create_default_org(db: Session) -> Organization:
    org = db.scalar(select(Organization).where(Organization.short_code == "demo"))
    if org is None:
        org = Organization(name="Demonstration Hospital", short_code="demo", is_demo=True)
        db.add(org)
        db.flush()
    return org


def get_or_create_default_user(db: Session) -> User:
    """
    The prototype resolves a single reviewer identity.

    Real authentication is designed for (see api/security.py and the User model)
    but not enforced; this function is the single place that assumption lives,
    so wiring in a real session later touches one call site.
    """
    user = db.scalar(select(User).where(User.email == "reviewer@demo.local"))
    if user is None:
        org = get_or_create_default_org(db)
        user = User(organization_id=org.id, email="reviewer@demo.local",
                    full_name="Demo Reviewer", role=UserRole.radiologist)
        db.add(user)
        db.flush()
    return user


def get_or_create_patient(db: Session, external_id: str, *, age: int | None = None,
                          sex: str | None = None, is_demo: bool = False) -> Patient:
    org = get_or_create_default_org(db)
    p = db.scalar(select(Patient).where(
        Patient.external_id == external_id, Patient.organization_id == org.id))
    if p is None:
        p = Patient(organization_id=org.id, external_id=external_id,
                    age=age, sex=sex, is_demo=is_demo)
        db.add(p)
        db.flush()
    return p


def record_analysis(db: Session, *, patient: Patient, filename: str,
                    result: dict, is_demo: bool = False,
                    acquired_at: datetime | None = None,
                    actor: User | None = None) -> Analysis:
    """Persist one engine result as Scan + Analysis + Findings + draft Report."""
    fp = result["final_prediction"]
    inp = result.get("input", {})
    seg = result.get("segmentation", {}) or {}
    perf = result.get("model_performance", {}) or {}

    scan = Scan(
        patient_id=patient.id, filename=filename,
        source_format=str(inp.get("source_format", "")),
        is_dicom=bool(inp.get("is_dicom", False)),
        width=(inp.get("original_size") or [None, None])[0],
        height=(inp.get("original_size") or [None, None])[1],
        acquired_at=acquired_at, uploaded_by=actor.id if actor else None,
        is_demo=is_demo,
    )
    db.add(scan)
    db.flush()

    priority = derive_priority(fp["predicted_class"], float(fp["confidence"]),
                               float(fp.get("branch_agreement", 1.0)))

    analysis = Analysis(
        scan_id=scan.id, patient_id=patient.id,
        predicted_class=fp["predicted_class"], risk_band=fp["risk_band"],
        confidence=float(fp["confidence"]),
        branch_agreement=float(fp.get("branch_agreement", 1.0)),
        priority=priority, review_status=ReviewStatus.ai_generated,
        n_lesions=int(seg.get("n_lesions") or 0),
        runtime_seconds=float(result.get("runtime_seconds", 0.0)),
        model_version=f"{perf.get('backbone', '')}/{perf.get('primary_model', '')}",
        result_json=result, is_demo=is_demo,
    )
    db.add(analysis)
    db.flush()

    for les in (seg.get("lesions") or []):
        db.add(Finding(
            analysis_id=analysis.id, source="unet_segmentation",
            area_px=int(les.get("area_px", 0)),
            equivalent_diameter_px=float(les.get("equivalent_diameter_px", 0.0)),
            area_fraction_of_lung=les.get("area_fraction_of_lung"),
            laterality=str(les.get("laterality", "")), zone=str(les.get("zone", "")),
            mean_confidence=float(les.get("mean_confidence", 0.0)),
            bbox_json=list(les.get("bbox_xyxy") or []),
        ))

    db.add(Report(analysis_id=analysis.id, status=ReviewStatus.ai_generated,
                  body_json=build_report_body(result, patient, scan)))

    audit(db, "analysis.created", "analysis", analysis.id, actor,
          patient=patient.external_id, priority=priority.value,
          predicted=fp["predicted_class"])
    db.commit()
    return analysis


# -------------------------------------------------------------- reporting

def build_report_body(result: dict, patient: Patient, scan: Scan) -> dict:
    """
    Structured preliminary report.

    Every section is derived from measured output. Nothing is inferred beyond
    what the engine returned, and the unsupported sections are present with an
    explicit reason rather than omitted (an absent section reads as an
    oversight; a stated one reads as a boundary).
    """
    fp = result["final_prediction"]
    ex = result.get("explainability", {}) or {}
    reg = ex.get("region", {}) or {}
    seg = result.get("segmentation", {}) or {}
    mp = result.get("measured_parameters", {}) or {}
    perf = result.get("model_performance", {}) or {}
    ci = result.get("clinical_information", {}) or {}
    inp = result.get("input", {}) or {}

    idp = seg.get("in_domain_performance") or {}

    return {
        "patient_information": {
            "identifier": patient.external_id,
            "age": patient.age, "sex": patient.sex,
            "note": "Opaque identifier only. No personally identifying information is stored.",
        },
        "scan_information": {
            "filename": scan.filename,
            "format": inp.get("source_format"),
            "dicom": inp.get("is_dicom"),
            "original_size": inp.get("original_size"),
            "analysis_resolution": inp.get("analysis_resolution"),
            "windowing": inp.get("windowing_applied"),
            "hu_available": inp.get("hu_available"),
            "acquisition_notes": inp.get("notes", []),
        },
        "image_quality": {
            "lung_field_detected": mp.get("lung_field_detected"),
            "lung_field_area_fraction": mp.get("lung_field_area_fraction"),
            "sharpness_laplacian_var": mp.get("sharpness_laplacian_var"),
            "contrast_rms": mp.get("contrast_rms"),
            "intensity_mean": mp.get("intensity_mean"),
            "intensity_std": mp.get("intensity_std"),
            "note": ("Automated technical checks only. This is not a diagnostic "
                     "image-quality certification."),
        },
        "ai_findings": {
            "classification": fp["predicted_class"],
            "display": fp["display"],
            "confidence": fp["confidence"],
            "probabilities": fp["probabilities"],
            "selected_model": fp["selected_model"],
            "branch_agreement": fp.get("branch_agreement"),
            "decision_policy_applied": fp.get("decision_policy_applied"),
        },
        "suspected_region": {
            "segmentation_regions": seg.get("n_lesions", 0),
            "lesions": seg.get("lesions", []),
            "attention_location": reg.get("location_description"),
            "attention_caveat": ex.get("caveat"),
        },
        "risk_assessment": {
            "band": fp["risk_band"],
            "confidence": fp["confidence"],
            "caveat": ci.get("confidence_caveat"),
        },
        "explainability_summary": {
            "method": ex.get("method"),
            "label": ex.get("label"),
            "caveat": ex.get("caveat"),
            "segmentation_model": seg.get("model"),
            "segmentation_detection_rate": idp.get("detection_rate"),
            "segmentation_dice_when_detected": idp.get("dice_when_detected"),
        },
        "model_information": {
            "backbone": perf.get("backbone"),
            "primary_model": perf.get("primary_model"),
            "test_metrics": perf.get("test_metrics", {}),
            "quantum_circuit": perf.get("quantum_circuit", {}),
        },
        "limitations": result.get("ai_limitations", []),
        "unsupported": {
            "clinical_staging": result.get("staging", {}).get("message"),
            **{k: v for k, v in (result.get("not_supported") or {}).items()},
        },
        "clinical_review": {
            "status": "ai_generated",
            "reviewer": None, "reviewed_at": None, "clinician_notes": "",
        },
        "disclaimer": result.get("disclaimer"),
    }


# ------------------------------------------------------------------ queries

def queue_items(db: Session, *, priority: str | None = None,
                review_status: str | None = None, search: str | None = None,
                limit: int = 100, offset: int = 0) -> tuple[list[dict], int]:
    stmt = (select(Analysis, Patient, Scan)
            .join(Patient, Analysis.patient_id == Patient.id)
            .join(Scan, Analysis.scan_id == Scan.id))
    if priority:
        stmt = stmt.where(Analysis.priority == Priority(priority))
    if review_status:
        stmt = stmt.where(Analysis.review_status == ReviewStatus(review_status))
    if search:
        stmt = stmt.where(Patient.external_id.ilike(f"%{search}%"))

    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0

    order = {Priority.high: 0, Priority.attention: 1, Priority.routine: 2}
    rows = db.execute(stmt.order_by(Analysis.created_at.desc()).limit(500)).all()
    items = [{
        "id": a.id, "scan_id": a.scan_id, "patient_id": a.patient_id,
        "patient_external_id": p.external_id, "scan_filename": s.filename,
        "patient_age": p.age, "patient_sex": p.sex,
        "predicted_class": a.predicted_class, "risk_band": a.risk_band,
        "confidence": a.confidence, "branch_agreement": a.branch_agreement,
        "priority": a.priority.value, "review_status": a.review_status.value,
        "n_lesions": a.n_lesions, "runtime_seconds": a.runtime_seconds,
        "model_version": a.model_version,
        "created_at": a.created_at.isoformat(), "is_demo": a.is_demo,
    } for a, p, s in rows]
    items.sort(key=lambda i: (order[Priority(i["priority"])], i["created_at"]), reverse=False)
    items.sort(key=lambda i: order[Priority(i["priority"])])
    return items[offset:offset + limit], total


def overview_stats(db: Session) -> dict:
    today = _now().replace(hour=0, minute=0, second=0, microsecond=0)
    total = db.scalar(select(func.count(Analysis.id))) or 0
    today_n = db.scalar(select(func.count(Analysis.id)).where(Analysis.created_at >= today)) or 0
    high = db.scalar(select(func.count(Analysis.id))
                     .where(Analysis.priority == Priority.high)) or 0
    pending = db.scalar(select(func.count(Analysis.id)).where(
        Analysis.review_status.in_([ReviewStatus.ai_generated, ReviewStatus.in_review]))) or 0
    approved = db.scalar(select(func.count(Analysis.id))
                         .where(Analysis.review_status == ReviewStatus.approved)) or 0
    demo_n = db.scalar(select(func.count(Analysis.id)).where(Analysis.is_demo.is_(True))) or 0
    patients = db.scalar(select(func.count(Patient.id))) or 0

    by_risk = dict(db.execute(
        select(Analysis.risk_band, func.count(Analysis.id)).group_by(Analysis.risk_band)).all())
    followups_due = db.scalar(select(func.count(FollowUp.id)).where(
        FollowUp.status.in_([FollowUpStatus.due, FollowUpStatus.overdue]))) or 0

    return {
        "total_analyses": total, "today_analyses": today_n,
        "high_priority": high, "pending_review": pending, "completed": approved,
        "patients": patients, "follow_ups_due": followups_due,
        "by_risk": by_risk,
        "demo_analyses": demo_n,
        "contains_demo_data": demo_n > 0,
    }


def recent_activity(db: Session, limit: int = 12) -> list[dict]:
    rows = db.execute(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)).scalars().all()
    return [{"id": r.id, "action": r.action, "entity_type": r.entity_type,
             "entity_id": r.entity_id, "actor_role": r.actor_role,
             "detail": r.detail_json, "created_at": r.created_at.isoformat()} for r in rows]


def refresh_followup_statuses(db: Session) -> None:
    """Recompute due/overdue from clinician-set dates. No interval is invented."""
    now = _now()
    for fu in db.execute(select(FollowUp)).scalars().all():
        if fu.status == FollowUpStatus.completed:
            continue
        due = fu.review_due
        if due.tzinfo is None:
            due = due.replace(tzinfo=timezone.utc)
        if due < now:
            fu.status = FollowUpStatus.overdue
        elif due <= now + timedelta(days=14):
            fu.status = FollowUpStatus.due
        else:
            fu.status = FollowUpStatus.upcoming
    db.commit()
