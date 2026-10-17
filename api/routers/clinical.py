"""
Clinical workflow endpoints: patients, scans, analyses, queue, reports,
feedback, follow-ups and comparison.

Transport and validation only - the rules live in api/services.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.db.models import (
    Analysis, AgreementStatus, DoctorFeedback, FollowUp, FollowUpStatus,
    Patient, Report, ReviewStatus, Scan,
)
from api.db.session import get_session
from api.services import analysis_service as svc

router = APIRouter(prefix="/api", tags=["clinical"])

MAX_UPLOAD_MB = 40
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom", ".ima"}


# ------------------------------------------------------------------ schemas

class FeedbackIn(BaseModel):
    doctor_assessment: str = Field(..., description="normal | benign | malignant | uncertain")
    agreement: str = Field(..., description="agree | modify | reject")
    comments: str = ""


class ReportUpdateIn(BaseModel):
    status: str | None = None
    clinician_notes: str | None = None
    edits: dict[str, Any] | None = None


class FollowUpIn(BaseModel):
    patient_id: str
    analysis_id: str | None = None
    review_due: str = Field(..., description="ISO date set by the clinician")
    notes: str = ""


# ------------------------------------------------------------------ analyse

@router.post("/analyze")
async def analyze_and_store(
    file: UploadFile = File(...),
    patient_external_id: str | None = Form(default=None),
    patient_age: int | None = Form(default=None),
    patient_sex: str | None = Form(default=None),
    persist: bool = Form(default=True),
    db: Session = Depends(get_session),
):
    """
    Run inference and (optionally) persist it against a patient.

    The response shape is a superset of the original stateless `/api/analyze`,
    so existing consumers keep working; `analysis_id` and `patient_id` are added
    when the result is stored.
    """
    from src.inference.engine import ModelNotAvailable, get_engine
    from src.inference.clinical_info import CANCER_EDUCATION, clinical_package
    from src.preprocessing.medical_image import ImageLoadError
    from pathlib import Path

    name = file.filename or "upload"
    ext = Path(name).suffix.lower()
    if ext and ext not in ALLOWED_EXT:
        raise HTTPException(415, f"Unsupported file type '{ext}'. "
                                 f"Accepted: {', '.join(sorted(ALLOWED_EXT))}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_MB} MB limit.")

    try:
        engine = get_engine()
    except ModelNotAvailable as e:
        raise HTTPException(503, f"Models are not available: {e}")

    try:
        result = engine.analyze(data, name)
    except ImageLoadError as e:
        raise HTTPException(422, str(e))
    except MemoryError:
        raise HTTPException(507, "Ran out of memory processing this image.")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise HTTPException(507, "GPU memory exhausted. Retry shortly.")
        raise HTTPException(500, f"Analysis failed: {e}")

    result["clinical_information"] = clinical_package(
        result["final_prediction"]["predicted_class"],
        result["final_prediction"]["confidence"])
    result["ai_limitations"] = CANCER_EDUCATION["ai_limitations"]
    result["triage"] = {
        "priority": svc.derive_priority(
            result["final_prediction"]["predicted_class"],
            float(result["final_prediction"]["confidence"]),
            float(result["final_prediction"].get("branch_agreement", 1.0))).value,
        "rule": svc.TRIAGE_RULE_DESCRIPTION,
    }

    if persist:
        actor = svc.get_or_create_default_user(db)
        pid = patient_external_id or f"ANON-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        patient = svc.get_or_create_patient(db, pid, age=patient_age, sex=patient_sex)
        analysis = svc.record_analysis(db, patient=patient, filename=name,
                                       result=result, actor=actor)
        result["analysis_id"] = analysis.id
        result["patient_id"] = patient.id
        result["patient_external_id"] = patient.external_id
        result["persisted"] = True
    else:
        result["persisted"] = False

    return result


# -------------------------------------------------------------- dashboards

@router.get("/overview")
def overview(db: Session = Depends(get_session)):
    svc.refresh_followup_statuses(db)
    return {"stats": svc.overview_stats(db),
            "recent_activity": svc.recent_activity(db),
            "triage_rule": svc.TRIAGE_RULE_DESCRIPTION}


@router.get("/priority-queue")
def priority_queue(
    priority: str | None = Query(default=None),
    review_status: str | None = Query(default=None),
    search: str | None = Query(default=None),
    limit: int = Query(default=100, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_session),
):
    items, total = svc.queue_items(db, priority=priority, review_status=review_status,
                                   search=search, limit=limit, offset=offset)
    return {"items": items, "total": total, "rule": svc.TRIAGE_RULE_DESCRIPTION}


# ---------------------------------------------------------------- patients

@router.get("/patients")
def list_patients(search: str | None = Query(default=None),
                  limit: int = Query(default=100, le=500),
                  db: Session = Depends(get_session)):
    stmt = select(Patient)
    if search:
        stmt = stmt.where(Patient.external_id.ilike(f"%{search}%"))
    patients = db.execute(stmt.order_by(Patient.created_at.desc()).limit(limit)).scalars().all()

    out = []
    for p in patients:
        rows = db.execute(select(Analysis).where(Analysis.patient_id == p.id)
                          .order_by(Analysis.created_at.desc())).scalars().all()
        out.append({
            "id": p.id, "external_id": p.external_id, "age": p.age, "sex": p.sex,
            "is_demo": p.is_demo, "created_at": p.created_at.isoformat(),
            "scan_count": len(rows),
            "latest_risk": rows[0].risk_band if rows else None,
            "latest_scan_at": rows[0].created_at.isoformat() if rows else None,
        })
    return {"items": out, "total": len(out)}


@router.get("/patients/{patient_id}")
def patient_detail(patient_id: str, db: Session = Depends(get_session)):
    p = db.get(Patient, patient_id)
    if p is None:
        raise HTTPException(404, "Patient not found")

    rows = db.execute(
        select(Analysis, Scan).join(Scan, Analysis.scan_id == Scan.id)
        .where(Analysis.patient_id == p.id).order_by(Analysis.created_at.asc())
    ).all()

    timeline = [{
        "analysis_id": a.id, "scan_id": s.id, "filename": s.filename,
        "predicted_class": a.predicted_class, "risk_band": a.risk_band,
        "confidence": a.confidence, "priority": a.priority.value,
        "review_status": a.review_status.value, "n_lesions": a.n_lesions,
        "created_at": a.created_at.isoformat(),
        "acquired_at": s.acquired_at.isoformat() if s.acquired_at else None,
        "is_demo": a.is_demo,
    } for a, s in rows]

    fus = db.execute(select(FollowUp).where(FollowUp.patient_id == p.id)
                     .order_by(FollowUp.review_due.asc())).scalars().all()

    return {
        "patient": {"id": p.id, "external_id": p.external_id, "age": p.age,
                    "sex": p.sex, "notes": p.notes, "is_demo": p.is_demo,
                    "created_at": p.created_at.isoformat()},
        "timeline": timeline,
        "follow_ups": [{
            "id": f.id, "risk_band": f.risk_band,
            "review_due": f.review_due.isoformat(),
            "status": f.status.value, "notes": f.notes,
            "last_scan_at": f.last_scan_at.isoformat() if f.last_scan_at else None,
        } for f in fus],
        "comparison_available": len(timeline) >= 2,
    }


# ---------------------------------------------------------------- analyses

@router.get("/analysis/{analysis_id}")
def get_analysis(analysis_id: str, db: Session = Depends(get_session)):
    a = db.get(Analysis, analysis_id)
    if a is None:
        raise HTTPException(404, "Analysis not found")
    p = db.get(Patient, a.patient_id)
    s = db.get(Scan, a.scan_id)
    rep = db.scalar(select(Report).where(Report.analysis_id == a.id))
    fb = db.execute(select(DoctorFeedback).where(DoctorFeedback.analysis_id == a.id)
                    .order_by(DoctorFeedback.created_at.desc())).scalars().all()
    return {
        "analysis": {
            "id": a.id, "patient_id": a.patient_id, "scan_id": a.scan_id,
            "predicted_class": a.predicted_class, "risk_band": a.risk_band,
            "confidence": a.confidence, "branch_agreement": a.branch_agreement,
            "priority": a.priority.value, "review_status": a.review_status.value,
            "n_lesions": a.n_lesions, "model_version": a.model_version,
            "created_at": a.created_at.isoformat(), "is_demo": a.is_demo,
        },
        "patient": {"id": p.id, "external_id": p.external_id, "age": p.age, "sex": p.sex} if p else None,
        "scan": {"id": s.id, "filename": s.filename, "source_format": s.source_format,
                 "is_dicom": s.is_dicom} if s else None,
        "result": a.result_json,
        "report": {"id": rep.id, "status": rep.status.value, "body": rep.body_json,
                   "clinician_edits": rep.clinician_edits,
                   "updated_at": rep.updated_at.isoformat()} if rep else None,
        "feedback": [{
            "id": f.id, "ai_prediction": f.ai_prediction,
            "doctor_assessment": f.doctor_assessment, "agreement": f.agreement.value,
            "comments": f.comments, "reviewer_role": f.reviewer_role.value,
            "created_at": f.created_at.isoformat(),
        } for f in fb],
    }


# ----------------------------------------------------------------- reports

@router.get("/reports/{analysis_id}")
def get_report(analysis_id: str, db: Session = Depends(get_session)):
    rep = db.scalar(select(Report).where(Report.analysis_id == analysis_id))
    if rep is None:
        raise HTTPException(404, "Report not found")
    return {"id": rep.id, "analysis_id": rep.analysis_id, "status": rep.status.value,
            "body": rep.body_json, "clinician_edits": rep.clinician_edits,
            "reviewed_at": rep.reviewed_at.isoformat() if rep.reviewed_at else None,
            "updated_at": rep.updated_at.isoformat()}


@router.patch("/reports/{analysis_id}")
def update_report(analysis_id: str, payload: ReportUpdateIn,
                  db: Session = Depends(get_session)):
    """
    Move a report through review.

    The AI-generated body is never overwritten - clinician changes accumulate in
    `clinician_edits` so the original model output stays auditable.
    """
    rep = db.scalar(select(Report).where(Report.analysis_id == analysis_id))
    if rep is None:
        raise HTTPException(404, "Report not found")
    a = db.get(Analysis, analysis_id)
    actor = svc.get_or_create_default_user(db)

    if payload.status:
        try:
            new_status = ReviewStatus(payload.status)
        except ValueError:
            raise HTTPException(400, f"Unknown status '{payload.status}'")
        rep.status = new_status
        if a:
            a.review_status = new_status
        if new_status in (ReviewStatus.approved, ReviewStatus.rejected):
            rep.reviewed_by = actor.id
            rep.reviewed_at = datetime.now(timezone.utc)

    edits = dict(rep.clinician_edits or {})
    if payload.edits:
        edits.update(payload.edits)
    if payload.clinician_notes is not None:
        edits["clinician_notes"] = payload.clinician_notes
    rep.clinician_edits = edits

    svc.audit(db, "report.updated", "report", rep.id, actor,
              status=rep.status.value, has_edits=bool(edits))
    db.commit()
    return {"id": rep.id, "status": rep.status.value, "clinician_edits": rep.clinician_edits}


# ---------------------------------------------------------------- feedback

@router.post("/doctor-feedback")
def submit_feedback(analysis_id: str, payload: FeedbackIn,
                    db: Session = Depends(get_session)):
    """
    Store a clinician assessment.

    Explicitly NOT used to retrain any production model. Rows are retained for a
    future validated training pipeline; `used_for_training` stays False here.
    """
    a = db.get(Analysis, analysis_id)
    if a is None:
        raise HTTPException(404, "Analysis not found")
    try:
        agreement = AgreementStatus(payload.agreement)
    except ValueError:
        raise HTTPException(400, "agreement must be agree | modify | reject")

    actor = svc.get_or_create_default_user(db)
    fb = DoctorFeedback(
        analysis_id=a.id, ai_prediction=a.predicted_class,
        doctor_assessment=payload.doctor_assessment, agreement=agreement,
        comments=payload.comments, reviewer_id=actor.id, reviewer_role=actor.role,
        used_for_training=False,
    )
    db.add(fb)

    if agreement == AgreementStatus.reject:
        a.review_status = ReviewStatus.rejected
    elif agreement == AgreementStatus.modify:
        a.review_status = ReviewStatus.edited

    svc.audit(db, "feedback.submitted", "analysis", a.id, actor,
              agreement=agreement.value, assessment=payload.doctor_assessment)
    db.commit()
    return {"id": fb.id, "stored": True,
            "note": ("Stored for future validated training pipelines. This platform "
                     "does not retrain production models from clinician feedback.")}


# -------------------------------------------------------------- follow-ups

@router.get("/follow-ups")
def list_followups(status: str | None = Query(default=None),
                   db: Session = Depends(get_session)):
    svc.refresh_followup_statuses(db)
    stmt = select(FollowUp, Patient).join(Patient, FollowUp.patient_id == Patient.id)
    if status:
        stmt = stmt.where(FollowUp.status == FollowUpStatus(status))
    rows = db.execute(stmt.order_by(FollowUp.review_due.asc())).all()
    return {"items": [{
        "id": f.id, "patient_id": f.patient_id, "patient_external_id": p.external_id,
        "analysis_id": f.analysis_id, "risk_band": f.risk_band,
        "last_scan_at": f.last_scan_at.isoformat() if f.last_scan_at else None,
        "review_due": f.review_due.isoformat(), "status": f.status.value,
        "notes": f.notes, "is_demo": f.is_demo,
    } for f, p in rows],
        "note": ("Review dates are set by clinicians. The platform does not derive "
                 "follow-up intervals autonomously.")}


@router.post("/follow-ups")
def create_followup(payload: FollowUpIn, db: Session = Depends(get_session)):
    p = db.get(Patient, payload.patient_id)
    if p is None:
        raise HTTPException(404, "Patient not found")
    try:
        due = datetime.fromisoformat(payload.review_due.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(400, "review_due must be an ISO date")

    last = db.execute(select(Analysis).where(Analysis.patient_id == p.id)
                      .order_by(Analysis.created_at.desc())).scalars().first()
    actor = svc.get_or_create_default_user(db)
    fu = FollowUp(patient_id=p.id, analysis_id=payload.analysis_id or (last.id if last else None),
                  risk_band=last.risk_band if last else "",
                  last_scan_at=last.created_at if last else None,
                  review_due=due, notes=payload.notes, set_by=actor.id,
                  is_demo=p.is_demo)
    db.add(fu)
    svc.audit(db, "followup.created", "patient", p.id, actor, review_due=payload.review_due)
    db.commit()
    svc.refresh_followup_statuses(db)
    return {"id": fu.id, "status": fu.status.value}


# ------------------------------------------------------------- comparison

@router.get("/scan-comparison")
def scan_comparison(analysis_a: str, analysis_b: str, db: Session = Depends(get_session)):
    """
    Compare two analyses of the same patient.

    Reports OBSERVED IMAGING AND MODEL DIFFERENCES only. It deliberately makes
    no statement about disease progression - that requires clinical correlation
    and validation this platform does not have.
    """
    a = db.get(Analysis, analysis_a)
    b = db.get(Analysis, analysis_b)
    if a is None or b is None:
        raise HTTPException(404, "One or both analyses not found")
    if a.patient_id != b.patient_id:
        raise HTTPException(400, "Analyses belong to different patients")

    ra, rb = a.result_json or {}, b.result_json or {}
    sa = (ra.get("segmentation") or {})
    sb = (rb.get("segmentation") or {})
    ma = (ra.get("measured_parameters") or {})
    mb = (rb.get("measured_parameters") or {})

    def area(seg):
        return sum(l.get("area_px", 0) for l in (seg.get("lesions") or []))

    return {
        "patient_id": a.patient_id,
        "earlier": {"analysis_id": a.id, "created_at": a.created_at.isoformat(),
                    "predicted_class": a.predicted_class, "risk_band": a.risk_band,
                    "confidence": a.confidence, "n_lesions": a.n_lesions,
                    "total_lesion_area_px": area(sa)},
        "later": {"analysis_id": b.id, "created_at": b.created_at.isoformat(),
                  "predicted_class": b.predicted_class, "risk_band": b.risk_band,
                  "confidence": b.confidence, "n_lesions": b.n_lesions,
                  "total_lesion_area_px": area(sb)},
        "observed_differences": {
            "risk_band_changed": a.risk_band != b.risk_band,
            "class_changed": a.predicted_class != b.predicted_class,
            "confidence_delta": round(b.confidence - a.confidence, 4),
            "lesion_count_delta": b.n_lesions - a.n_lesions,
            "lesion_area_delta_px": area(sb) - area(sa),
            "lung_field_fraction_delta": round(
                float(mb.get("lung_field_area_fraction", 0)) -
                float(ma.get("lung_field_area_fraction", 0)), 4),
        },
        "interpretation_note": (
            "These are observed imaging and model differences between two independent "
            "analyses. They are NOT a measurement of disease progression. Scans may "
            "differ in slice position, acquisition parameters and patient positioning, "
            "any of which changes these numbers without any change in the patient. "
            "Clinical correlation by a radiologist is required."),
        "images": {
            "earlier_overlay": ((ra.get("explainability") or {}).get("images") or {}).get("overlay"),
            "later_overlay": ((rb.get("explainability") or {}).get("images") or {}).get("overlay"),
        },
    }
