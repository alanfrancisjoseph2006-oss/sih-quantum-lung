"""
FastAPI backend for the Hybrid Quantum-Classical Lung Cancer Analysis Platform.
SIH 2026 - PS 26139 - Team CodeNova.

Run:  uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations
import json
import math
import sys
import traceback
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (DISCLAIMER, CLASSES, CLASS_DISPLAY, CLASS_RISK_BAND,
                        UNSUPPORTED_CAPABILITIES, REPORTS, hardware_summary,
                        N_QUBITS_SWEEP, Q_LAYERS)
from src.inference.engine import get_engine, engine_status, ModelNotAvailable
from src.inference.clinical_info import clinical_package, CANCER_EDUCATION, TREATMENT_DISCLAIMER
from src.preprocessing.medical_image import ImageLoadError

from api.db.session import init_db
from api.routers import clinical
from api.services import analysis_service as svc

MAX_UPLOAD_MB = 40
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom", ".ima"}

app = FastAPI(title="Hybrid Quantum-Classical Lung Cancer Analysis Platform",
              description=DISCLAIMER, version="1.0.0")
# The Next.js dev server runs on :3000; production serves same-origin.
app.add_middleware(CORSMiddleware,
                   allow_origins=["http://localhost:3000", "http://127.0.0.1:3000",
                                  "http://localhost:8000", "http://127.0.0.1:8000"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

init_db()
app.include_router(clinical.router)

FRONTEND = ROOT / "frontend"


def _sanitize(o):
    """
    Replace NaN / Infinity with None.

    Metric files legitimately contain NaN (e.g. an AUC that is undefined when a
    bootstrap resample happens to contain a single class). Python's json module
    writes those as bare `NaN`, which is not valid JSON and makes the HTTP
    response un-parseable for the browser. Serving null instead keeps the
    endpoint working and is honest about the value being absent.
    """
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_sanitize(v) for v in o]
    return o


def _json(path: Path, default=None):
    try:
        return _sanitize(json.load(open(path)))
    except Exception:
        return default if default is not None else {}


# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "engine": engine_status(), "hardware": hardware_summary(),
            "disclaimer": DISCLAIMER}


@app.get("/api/meta")
def meta():
    """Everything the front-end needs to render honest capability statements."""
    st = engine_status()
    comp = _json(REPORTS / "model_comparison.json")
    dep = comp.get("deployed", {})
    circuit = {}
    if st.get("ready"):
        try:
            circuit = get_engine().circuit_info
        except Exception:
            pass
    return {
        "project": {
            "name": "Hybrid Quantum-Classical AI Platform for Early Lung Cancer Detection",
            "problem_statement_id": "26139",
            "team": "CodeNova",
            "event": "Smart India Hackathon 2026",
        },
        "classes": CLASSES,
        "class_display": CLASS_DISPLAY,
        "class_risk_band": CLASS_RISK_BAND,
        "unsupported_capabilities": UNSUPPORTED_CAPABILITIES,
        "deployed": dep,
        "engine": st,
        "hardware": hardware_summary(),
        "quantum": {"circuit": circuit, "qubit_sweep": N_QUBITS_SWEEP,
                    "variational_layers": Q_LAYERS},
        "disclaimer": DISCLAIMER,
        "accepted_formats": sorted(ALLOWED_EXT),
        "triage_rule": svc.TRIAGE_RULE_DESCRIPTION,
        "security_posture": {
            "statement": ("Designed with healthcare privacy and security considerations. "
                          "This is a research prototype and carries no regulatory "
                          "clearance or compliance certification."),
            "implemented": [
                "Opaque patient identifiers only - no names, dates of birth or contact details are modelled",
                "DICOM patient-identifying tags are never read into API responses",
                "Append-only audit log of who changed which record",
                "Role model (administrator, radiologist, doctor, technician, researcher)",
                "Demo records flagged at the database level, not only in the interface",
            ],
            "designed_not_enforced": [
                "Authentication and session management",
                "Role-based access enforcement on endpoints",
                "Encryption at rest and in transit",
            ],
            "not_claimed": [
                "HIPAA / GDPR compliance certification",
                "Medical device approval (CE / FDA / CDSCO)",
                "Clinical validation",
            ],
        },
    }


@app.get("/api/model-comparison")
def model_comparison():
    c = _json(REPORTS / "model_comparison.json")
    if not c:
        raise HTTPException(503, "Model comparison not available - run scripts/06_model_selection.py")
    return c


@app.get("/api/dataset-report")
def dataset_report():
    p = REPORTS / "dataset_report.md"
    if not p.exists():
        raise HTTPException(404, "dataset_report.md not found")
    return {"markdown": p.read_text(encoding="utf-8")}


@app.get("/api/localization-validation")
def localization_validation():
    d = _json(REPORTS / "localization_validation.json")
    if not d:
        raise HTTPException(503, "Localization validation not available")
    # strip the long per-file arrays; the UI only needs the summary
    out = {"note": d.get("note"), "results": {}}
    for k, v in d.get("results", {}).items():
        out["results"][k] = {kk: vv for kk, vv in v.items() if kk != "per_file"}
    return out


@app.get("/api/lidc-experiment")
def lidc_experiment():
    d = _json(REPORTS / "lidc_experiment.json")
    if not d:
        raise HTTPException(503, "LIDC experiment record not available")
    return d


@app.get("/api/external-validation")
def external_validation():
    d = _json(REPORTS / "external_validation.json")
    if not d:
        raise HTTPException(503, "External validation not available - run scripts/21_external_validation.py")
    return d


@app.get("/api/readiness")
def readiness():
    """Scientific readiness level, awarded from measured artefacts."""
    from src.evaluation.readiness import assess
    return assess(REPORTS)


@app.get("/api/domain-shift")
def domain_shift():
    """Everything measured about the domain-shift investigation (D26/D27)."""
    out = {
        "signature_probe": _json(REPORTS / "domain_signature_probe.json"),
        "harmonisation": _json(REPORTS / "harmonisation_search.json"),
        "robustness_arms": _json(REPORTS / "domain_robust_comparison.json"),
        "inversion": _json(REPORTS / "inversion_analysis.json"),
        "roi": _json(REPORTS / "roi_classifier.json"),
        "hybrid_quantum_roi": _json(REPORTS / "hybrid_quantum_roi.json"),
        "final_comparison": _json(REPORTS / "final_comparison.json"),
        "overfitting_fixes": _json(REPORTS / "overfitting_fixes.json"),
        "external": _json(REPORTS / "external_validation.json"),
    }
    if not any(out.values()):
        raise HTTPException(503, "Domain-shift analysis not available - run scripts/22-28")
    return out


@app.get("/api/education")
def education():
    return {"cancer_education": CANCER_EDUCATION,
            "treatment_disclaimer": TREATMENT_DISCLAIMER,
            "disclaimer": DISCLAIMER}


@app.get("/api/experiments")
def experiments():
    out = {}
    for f in sorted((REPORTS / "experiments").glob("*.json")):
        out[f.stem] = _json(f)
    return out


# ---------------------------------------------------------------------------
@app.post("/api/analyze-stateless")
async def analyze_stateless(file: UploadFile = File(...)):
    """Analyze one uploaded CT image (PNG/JPG/BMP/TIFF/DICOM)."""
    name = file.filename or "upload"
    ext = Path(name).suffix.lower()
    if ext and ext not in ALLOWED_EXT:
        raise HTTPException(415, f"Unsupported file type '{ext}'. "
                                 f"Accepted: {', '.join(sorted(ALLOWED_EXT))}")
    try:
        data = await file.read()
    except Exception as e:
        raise HTTPException(400, f"Could not read upload: {e}")

    if not data:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_MB} MB limit.")

    try:
        engine = get_engine()
    except ModelNotAvailable as e:
        raise HTTPException(503, f"Models are not available: {e}")
    except Exception as e:
        raise HTTPException(503, f"Inference engine failed to initialise: {e}")

    try:
        result = engine.analyze(data, name)
    except ImageLoadError as e:
        raise HTTPException(422, str(e))
    except MemoryError:
        raise HTTPException(507, "Ran out of memory processing this image. "
                                 "Try a smaller or lower-resolution file.")
    except RuntimeError as e:
        msg = str(e)
        if "CUDA out of memory" in msg:
            raise HTTPException(507, "GPU memory exhausted. Retry shortly.")
        traceback.print_exc()
        raise HTTPException(500, f"Analysis failed: {msg}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Unexpected analysis error: {type(e).__name__}: {e}")

    result["clinical_information"] = clinical_package(
        result["final_prediction"]["predicted_class"],
        result["final_prediction"]["confidence"])
    result["ai_limitations"] = CANCER_EDUCATION["ai_limitations"]
    return JSONResponse(result)


# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled(request, exc):
    traceback.print_exc()
    return JSONResponse(status_code=500,
                        content={"detail": f"Internal error: {type(exc).__name__}"})


if FRONTEND.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND)), name="assets")

    @app.get("/")
    def index():
        f = FRONTEND / "index.html"
        if not f.exists():
            return JSONResponse({"detail": "frontend/index.html not built yet"}, 404)
        return FileResponse(str(f))
