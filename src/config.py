"""
Central configuration for the SIH 2026 PS-26139 hybrid quantum-classical platform.
Team CodeNova.

Every tunable lives here so experiments stay reproducible and the API,
training scripts and evaluation all agree on the same constants.
"""
from __future__ import annotations
import os
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_SOURCE = Path(r"C:\Users\alanf\OneDrive\Desktop\sih dataset")  # read-only
IQ_DIR = RAW_SOURCE / "The IQ-OTHNCCD lung cancer dataset" / "The IQ-OTHNCCD lung cancer dataset"
DEEPLESION_DIR = RAW_SOURCE / "Test cases"

PROCESSED = DATA / "processed"
QUARANTINE = DATA / "quarantine"
SPLITS = DATA / "splits"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
EXPERIMENTS = REPORTS / "experiments"

for _p in (PROCESSED, QUARANTINE, SPLITS, MODELS, REPORTS, FIGURES, EXPERIMENTS):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Task definition -- derived strictly from what the dataset actually contains.
# See reports/dataset_report.md section 3.
# ----------------------------------------------------------------------------
CLASSES = ["benign", "malignant", "normal"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASSES)

# Human-facing risk wording. Deliberately hedged: these are model findings,
# not diagnoses.
CLASS_DISPLAY = {
    "normal":    "No suspicious finding detected",
    "benign":    "Benign-appearing finding",
    "malignant": "Suspicious finding - malignancy features",
}
CLASS_RISK_BAND = {"normal": "Low", "benign": "Intermediate", "malignant": "High"}

# Capabilities the dataset does NOT support. The UI reads these to render
# honest "not available" states instead of fabricating output.
UNSUPPORTED_CAPABILITIES = {
    "clinical_staging": "No stage labels exist in the IQ-OTH/NCCD distribution.",
    "supervised_segmentation": "No tumor masks exist in the dataset.",
    "histological_subtype": "No subtype labels exist in the dataset.",
}

DISCLAIMER = ("AI-assisted research and clinical decision-support tool. "
              "Not a replacement for professional medical diagnosis.")

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
SEED = 42

# ----------------------------------------------------------------------------
# Image preprocessing
# ----------------------------------------------------------------------------
IMG_SIZE = 300              # EfficientNet-B3 native training resolution
BACKBONE = "efficientnet_b3"
BASELINE_BACKBONE = "densenet121"

# ImageNet statistics -- used because we transfer-learn from ImageNet weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CLAHE. Applied because dataset A is 8-bit pre-windowed JPEG: true HU-based
# lung windowing is impossible (see dataset_report.md limitation #2).
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
BATCH_SIZE = 16             # tuned for 6 GB RTX 4050 at 300x300 with AMP
EPOCHS_HEAD = 4             # warm-up: classifier head only
EPOCHS_FULL = 22            # then full fine-tune
LR_HEAD = 1e-3
LR_FULL = 2e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
EARLY_STOP_PATIENCE = 7
NUM_WORKERS = 4
USE_AMP = True

# ----------------------------------------------------------------------------
# Quantum branch (PennyLane)
# ----------------------------------------------------------------------------
QUANTUM_BACKEND = "default.qubit"
N_QUBITS_SWEEP = [4, 6, 8]  # experimental sweep required by the brief
N_QUBITS = 8                # default; overridden by whichever wins on val
Q_LAYERS = 3                # shallow, to stay trainable and simulable
Q_ENCODING = "angle_ry"

# Dimensionality reduction feeding the quantum circuit
PCA_COMPONENTS = 32         # intermediate reduction before per-qubit projection

# ----------------------------------------------------------------------------
# Explainability
# ----------------------------------------------------------------------------
GRADCAM_ALPHA = 0.45        # overlay blend weight
ATTENTION_THRESHOLD = 0.6   # fraction of max activation counted as "attended"

# Language rules -- enforced so the UI can never call attention a tumor boundary.
ATTENTION_LABEL = "Model attention / suspected abnormality region"
ATTENTION_NOT_LABEL = "Confirmed tumor boundary"


def hardware_summary() -> dict:
    """Detect available compute. Safe to call without torch installed."""
    info = {"torch": False, "cuda": False, "device": "cpu", "gpu_name": None,
            "vram_gb": None, "amp": False}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info.update(cuda=True, device="cuda", gpu_name=torch.cuda.get_device_name(0),
                        vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
                        amp=USE_AMP)
    except Exception:
        pass
    return info


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG we touch, for reproducible experiments."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
