"""
Database entities for the QuantumLung clinical platform.

Design notes
------------
* SQLite via SQLAlchemy 2.0. Chosen because the platform must run from a single
  clone with no service dependencies; the schema is portable to Postgres
  unchanged (no SQLite-specific column types are used).

* Every clinically meaningful row carries `is_demo`. Synthetic records are
  labelled at the database level, not just in the UI, so a demo record can never
  be silently presented as real patient data.

* `AuditLog` is append-only by convention and records who did what to which
  entity. Real healthcare software needs this; building it now means the
  security model is structural rather than bolted on later.

* Patient identifiers are opaque external IDs. No names, addresses, or dates of
  birth are modelled at all - the safest way to avoid handling PHI is to have
  nowhere to put it.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import (
    Boolean, DateTime, Enum, Float, ForeignKey, Integer, JSON, String, Text, Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# --------------------------------------------------------------------- enums

class UserRole(str, PyEnum):
    administrator = "administrator"
    radiologist = "radiologist"
    doctor = "doctor"
    technician = "technician"
    researcher = "researcher"


class ReviewStatus(str, PyEnum):
    ai_generated = "ai_generated"
    in_review = "in_review"
    edited = "edited"
    approved = "approved"
    further_review = "further_review"
    rejected = "rejected"


class Priority(str, PyEnum):
    high = "high"
    attention = "attention"
    routine = "routine"


class AgreementStatus(str, PyEnum):
    agree = "agree"
    modify = "modify"
    reject = "reject"


class FollowUpStatus(str, PyEnum):
    due = "due"
    upcoming = "upcoming"
    completed = "completed"
    overdue = "overdue"


# --------------------------------------------------------------------- core

class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))
    short_code: Mapped[str] = mapped_column(String(32), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=True)

    users: Mapped[list["User"]] = relationship(back_populates="organization")
    patients: Mapped[list["Patient"]] = relationship(back_populates="organization")


class User(Base):
    """
    Authentication is designed for but not enforced in the prototype.

    `password_hash` is present so a real auth flow can be added without a
    migration; the prototype never stores a credential and the API currently
    resolves a default reviewer. This is stated in the security documentation
    rather than implied to be complete.
    """
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"))
    email: Mapped[str] = mapped_column(String(255), unique=True)
    full_name: Mapped[str] = mapped_column(String(200))
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.radiologist)
    password_hash: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    organization: Mapped[Organization] = relationship(back_populates="users")


class Patient(Base):
    """Opaque identifier only. No name, DOB, address or contact is modelled."""
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"))
    external_id: Mapped[str] = mapped_column(String(64), index=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sex: Mapped[str | None] = mapped_column(String(1), nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    is_demo: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    organization: Mapped[Organization] = relationship(back_populates="patients")
    scans: Mapped[list["Scan"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    follow_ups: Mapped[list["FollowUp"]] = relationship(back_populates="patient", cascade="all, delete-orphan")


class Scan(Base):
    __tablename__ = "scans"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    source_format: Mapped[str] = mapped_column(String(32), default="")
    is_dicom: Mapped[bool] = mapped_column(Boolean, default=False)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    stored_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    acquired_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)
    uploaded_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    patient: Mapped[Patient] = relationship(back_populates="scans")
    analyses: Mapped[list["Analysis"]] = relationship(back_populates="scan", cascade="all, delete-orphan")


class Analysis(Base):
    """
    One inference run. The full engine payload is kept in `result_json` so the
    report and workspace can be re-rendered exactly, while the columns above it
    are the denormalised fields the queue and dashboards filter and sort on.
    """
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    scan_id: Mapped[str] = mapped_column(ForeignKey("scans.id"), index=True)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)

    predicted_class: Mapped[str] = mapped_column(String(16), index=True)
    risk_band: Mapped[str] = mapped_column(String(16), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    branch_agreement: Mapped[float] = mapped_column(Float, default=1.0)
    priority: Mapped[Priority] = mapped_column(Enum(Priority), default=Priority.routine, index=True)
    review_status: Mapped[ReviewStatus] = mapped_column(
        Enum(ReviewStatus), default=ReviewStatus.ai_generated, index=True)

    n_lesions: Mapped[int] = mapped_column(Integer, default=0)
    runtime_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    model_version: Mapped[str] = mapped_column(String(120), default="")

    result_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    scan: Mapped[Scan] = relationship(back_populates="analyses")
    findings: Mapped[list["Finding"]] = relationship(back_populates="analysis", cascade="all, delete-orphan")
    report: Mapped["Report | None"] = relationship(back_populates="analysis", uselist=False,
                                                   cascade="all, delete-orphan")
    feedback: Mapped[list["DoctorFeedback"]] = relationship(back_populates="analysis",
                                                            cascade="all, delete-orphan")


Index("ix_analyses_queue", Analysis.priority, Analysis.created_at)


class Finding(Base):
    """One segmented region. Never described as a confirmed tumour boundary."""
    __tablename__ = "findings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id"), index=True)
    source: Mapped[str] = mapped_column(String(32), default="unet_segmentation")
    area_px: Mapped[int] = mapped_column(Integer, default=0)
    equivalent_diameter_px: Mapped[float] = mapped_column(Float, default=0.0)
    area_fraction_of_lung: Mapped[float | None] = mapped_column(Float, nullable=True)
    laterality: Mapped[str] = mapped_column(String(64), default="")
    zone: Mapped[str] = mapped_column(String(64), default="")
    mean_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    bbox_json: Mapped[list] = mapped_column(JSON, default=list)

    analysis: Mapped[Analysis] = relationship(back_populates="findings")


class Report(Base):
    """
    Preliminary AI-assisted report moving through clinician review.

    `body_json` holds the structured sections. `clinician_edits` records what a
    reviewer changed, so the AI-generated original is never overwritten.
    """
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id"), unique=True, index=True)
    status: Mapped[ReviewStatus] = mapped_column(Enum(ReviewStatus), default=ReviewStatus.ai_generated)
    body_json: Mapped[dict] = mapped_column(JSON, default=dict)
    clinician_edits: Mapped[dict] = mapped_column(JSON, default=dict)
    reviewed_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    analysis: Mapped[Analysis] = relationship(back_populates="report")


class DoctorFeedback(Base):
    """
    Captured clinician assessment.

    Stored for future *validated* training pipelines only. Nothing in this
    application retrains a production model from these rows - that would be an
    unvalidated feedback loop on a clinical model.
    """
    __tablename__ = "doctor_feedback"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    analysis_id: Mapped[str] = mapped_column(ForeignKey("analyses.id"), index=True)
    ai_prediction: Mapped[str] = mapped_column(String(16))
    doctor_assessment: Mapped[str] = mapped_column(String(16))
    agreement: Mapped[AgreementStatus] = mapped_column(Enum(AgreementStatus))
    comments: Mapped[str] = mapped_column(Text, default="")
    reviewer_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    reviewer_role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.radiologist)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    used_for_training: Mapped[bool] = mapped_column(Boolean, default=False)

    analysis: Mapped[Analysis] = relationship(back_populates="feedback")


class FollowUp(Base):
    """Review dates are set by clinicians. The platform proposes no intervals."""
    __tablename__ = "follow_ups"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    patient_id: Mapped[str] = mapped_column(ForeignKey("patients.id"), index=True)
    analysis_id: Mapped[str | None] = mapped_column(ForeignKey("analyses.id"), nullable=True)
    risk_band: Mapped[str] = mapped_column(String(16), default="")
    last_scan_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    review_due: Mapped[datetime] = mapped_column(DateTime, index=True)
    status: Mapped[FollowUpStatus] = mapped_column(Enum(FollowUpStatus),
                                                   default=FollowUpStatus.upcoming, index=True)
    notes: Mapped[str] = mapped_column(Text, default="")
    set_by: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    patient: Mapped[Patient] = relationship(back_populates="follow_ups")


class ModelVersion(Base):
    """Which artefacts produced a given analysis, for traceability."""
    __tablename__ = "model_versions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    backbone: Mapped[str] = mapped_column(String(64), default="")
    primary_model: Mapped[str] = mapped_column(String(64), default="")
    n_qubits: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class AuditLog(Base):
    """Append-only. Who did what, to which entity, when."""
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    actor_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    actor_role: Mapped[str] = mapped_column(String(32), default="")
    action: Mapped[str] = mapped_column(String(64), index=True)
    entity_type: Mapped[str] = mapped_column(String(64))
    entity_id: Mapped[str] = mapped_column(String(32), index=True)
    detail_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now, index=True)
