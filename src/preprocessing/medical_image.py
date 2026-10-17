"""
Medical image preprocessing.

Design notes (documented decisions -- see reports/dataset_report.md):

1. Dataset A (IQ-OTH/NCCD) is 8-bit JPEG that has ALREADY been windowed by the
   original authors. Hounsfield Units are unrecoverable, so we deliberately do
   NOT pretend to apply HU lung-windowing to it. We apply CLAHE instead, which
   is a legitimate contrast operation on an already-windowed image.

2. Dataset B ("Test cases", DeepLesion) has green bounding boxes burned into the
   pixels. Any model shown those pixels could cheat off the annotation, so the
   green is inpainted out before inference. The coordinates are kept separately.

3. For genuine DICOM uploaded by a user, we DO have HU, so we apply the proper
   rescale slope/intercept and a real lung window.
"""
from __future__ import annotations
import io
import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src.config import IMG_SIZE, CLAHE_CLIP, CLAHE_GRID

# Standard clinical lung window (Hounsfield Units)
LUNG_WINDOW_CENTER = -600.0
LUNG_WINDOW_WIDTH = 1500.0


class ImageLoadError(ValueError):
    """Raised when an input cannot be decoded as a usable medical image."""


@dataclass
class LoadedImage:
    """A decoded image plus provenance describing exactly how it was obtained."""
    array: np.ndarray                      # uint8 grayscale HxW
    source_format: str                     # 'DICOM' | 'JPEG' | 'PNG' | ...
    original_size: tuple                   # (w, h) before any resize
    is_dicom: bool = False
    hu_available: bool = False
    windowing_applied: Optional[str] = None
    overlay_removed: bool = False
    dicom_meta: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def provenance(self) -> dict:
        d = asdict(self)
        d.pop("array")
        return d


# ---------------------------------------------------------------------------
# DICOM
# ---------------------------------------------------------------------------
def _apply_window(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    lo, hi = center - width / 2.0, center + width / 2.0
    out = np.clip(hu, lo, hi)
    return ((out - lo) / max(hi - lo, 1e-6) * 255.0).astype(np.uint8)


def load_dicom(data: bytes | str) -> LoadedImage:
    """
    Read a DICOM file and convert to display-ready uint8.

    Correctly applies RescaleSlope/RescaleIntercept to recover Hounsfield Units,
    then a lung window. If the file carries its own WindowCenter/WindowWidth and
    it looks like a chest window, we respect it; otherwise we use the standard
    lung window, and record which was used.
    """
    import pydicom

    try:
        src = io.BytesIO(data) if isinstance(data, (bytes, bytearray)) else data
        ds = pydicom.dcmread(src, force=True)
    except Exception as e:
        raise ImageLoadError(f"Unreadable DICOM: {e}") from e

    if not hasattr(ds, "pixel_array"):
        raise ImageLoadError("DICOM contains no pixel data (may be a metadata-only file).")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            arr = ds.pixel_array.astype(np.float32)
    except Exception as e:
        raise ImageLoadError(
            f"DICOM pixel data could not be decoded (missing/unsupported "
            f"transfer-syntax codec): {e}") from e

    notes = []
    # Multi-frame -> take the middle frame, and say so.
    if arr.ndim == 3:
        mid = arr.shape[0] // 2
        notes.append(f"Multi-frame DICOM ({arr.shape[0]} frames); middle frame {mid} used.")
        arr = arr[mid]
    if arr.ndim != 2:
        raise ImageLoadError(f"Unsupported DICOM pixel array shape {arr.shape}")

    original_size = (arr.shape[1], arr.shape[0])

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    modality = str(getattr(ds, "Modality", "") or "")
    hu_available = (modality == "CT") or ("RescaleIntercept" in ds)

    hu = arr * slope + intercept

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        hu = hu.max() - hu
        notes.append("MONOCHROME1 inverted to MONOCHROME2.")

    if hu_available:
        wc, ww = getattr(ds, "WindowCenter", None), getattr(ds, "WindowWidth", None)
        if isinstance(wc, (list, tuple, pydicom.multival.MultiValue)):
            wc = wc[0]
        if isinstance(ww, (list, tuple, pydicom.multival.MultiValue)):
            ww = ww[0]
        try:
            wc, ww = float(wc), float(ww)
        except (TypeError, ValueError):
            wc = ww = None
        # Only trust the embedded window if it is plausibly a lung/chest window.
        if wc is not None and ww is not None and ww > 0 and -900 <= wc <= 200 and ww >= 800:
            img = _apply_window(hu, wc, ww)
            win = f"DICOM-embedded window (C={wc:.0f}, W={ww:.0f})"
        else:
            img = _apply_window(hu, LUNG_WINDOW_CENTER, LUNG_WINDOW_WIDTH)
            win = f"Standard lung window (C={LUNG_WINDOW_CENTER:.0f}, W={LUNG_WINDOW_WIDTH:.0f})"
    else:
        lo, hi = np.percentile(hu, [1, 99])
        img = np.clip((hu - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255).astype(np.uint8)
        win = "Percentile normalisation (non-CT modality; no HU semantics)"
        notes.append(f"Modality reported as '{modality or 'unknown'}' - not CT.")

    meta = {}
    for tag in ("Modality", "Manufacturer", "SliceThickness", "KVP", "StudyDate",
                "SeriesDescription", "BodyPartExamined", "Rows", "Columns",
                "PixelSpacing", "ConvolutionKernel"):
        v = getattr(ds, tag, None)
        if v is not None:
            meta[tag] = str(v)
    meta["RescaleSlope"] = slope
    meta["RescaleIntercept"] = intercept
    # Deliberately never surfacing PatientName / PatientID / etc.

    return LoadedImage(array=img, source_format="DICOM", original_size=original_size,
                       is_dicom=True, hu_available=hu_available, windowing_applied=win,
                       dicom_meta=meta, notes=notes)


# ---------------------------------------------------------------------------
# Standard images
# ---------------------------------------------------------------------------
def remove_green_overlay(rgb: np.ndarray) -> tuple[np.ndarray, bool, np.ndarray | None]:
    """
    Detect and inpaint burned-in pure-green annotation boxes (DeepLesion 'Test cases').

    Returns (cleaned_rgb, was_removed, box_xyxy_or_None). Keeping the box
    coordinates lets us use them as localization ground truth while ensuring the
    model never sees the annotation itself.
    """
    a = rgb.astype(np.int16)
    green = (a[..., 1] > 140) & (a[..., 0] < 110) & (a[..., 2] < 110) & \
            ((a[..., 1] - a[..., 0]) > 40) & ((a[..., 1] - a[..., 2]) > 40)
    if green.sum() < 20:
        return rgb, False, None

    ys, xs = np.nonzero(green)
    box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=int)
    mask = cv2.dilate(green.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    cleaned = cv2.inpaint(rgb.astype(np.uint8), mask, 3, cv2.INPAINT_TELEA)
    return cleaned, True, box


def load_image(data: bytes | str, filename: str = "") -> LoadedImage:
    """
    Universal loader: routes DICOM vs standard raster, with graceful errors.
    Accepts raw bytes (API upload) or a path (training).
    """
    name = (filename or (data if isinstance(data, str) else "")).lower()
    is_dicom_ext = name.endswith((".dcm", ".dicom", ".ima"))

    raw = None
    if isinstance(data, (bytes, bytearray)):
        raw = bytes(data)
    else:
        try:
            with open(data, "rb") as fh:
                raw = fh.read()
        except OSError as e:
            raise ImageLoadError(f"Cannot open file: {e}") from e

    # DICOM files carry 'DICM' at byte offset 128.
    has_magic = len(raw) > 132 and raw[128:132] == b"DICM"
    if is_dicom_ext or has_magic:
        return load_dicom(raw)

    try:
        pil = Image.open(io.BytesIO(raw))
        pil.verify()
        pil = Image.open(io.BytesIO(raw))
        fmt = pil.format or "UNKNOWN"
        rgb = np.asarray(pil.convert("RGB"), dtype=np.uint8)
    except Exception as e:
        raise ImageLoadError(
            f"Not a readable image or DICOM file ({type(e).__name__}). "
            f"Supported: PNG, JPG, JPEG, BMP, TIFF, DCM.") from e

    original_size = (rgb.shape[1], rgb.shape[0])
    cleaned, removed, _box = remove_green_overlay(rgb)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_RGB2GRAY)

    notes = []
    if removed:
        notes.append("Burned-in green annotation overlay detected and inpainted "
                     "before analysis, so it cannot influence the prediction.")

    return LoadedImage(array=gray, source_format=fmt, original_size=original_size,
                       is_dicom=False, hu_available=False,
                       windowing_applied="None (input already 8-bit windowed)",
                       overlay_removed=removed, notes=notes)


# ---------------------------------------------------------------------------
# Normalisation pipeline shared by training and inference
# ---------------------------------------------------------------------------
def denoise(gray: np.ndarray) -> np.ndarray:
    """Mild edge-preserving denoise. Bilateral keeps nodule margins sharp,
    which a Gaussian blur would erode."""
    return cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    return clahe.apply(gray)


def resize_letterbox(gray: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """
    Aspect-preserving resize with zero padding.

    Chosen over a plain resize because 5.6% of dataset A is non-square
    (623x512, 801x512). A naive stretch would distort lesion morphology,
    and lesion shape is diagnostically meaningful.
    """
    h, w = gray.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    r = cv2.resize(gray, (nw, nh), interpolation=interp)
    canvas = np.zeros((size, size), dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = r
    return canvas


def preprocess(gray: np.ndarray, size: int = IMG_SIZE,
               use_clahe: bool = True, use_denoise: bool = True) -> np.ndarray:
    """Full deterministic preprocessing -> uint8 (size, size). Identical at
    train and inference time, which is what makes results reproducible."""
    g = gray
    if use_denoise:
        g = denoise(g)
    if use_clahe:
        g = apply_clahe(g)
    return resize_letterbox(g, size)


def to_model_tensor(proc: np.ndarray):
    """uint8 HxW -> normalised float tensor (1,3,H,W) for an ImageNet backbone."""
    import torch
    from src.config import IMAGENET_MEAN, IMAGENET_STD
    x = proc.astype(np.float32) / 255.0
    x = np.stack([x, x, x], axis=0)                      # gray -> 3 channels
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]
    x = (x - mean) / std
    return torch.from_numpy(x).unsqueeze(0)


# ---------------------------------------------------------------------------
# Measured (not predicted) image parameters, for the report UI
# ---------------------------------------------------------------------------
def lung_mask(gray: np.ndarray) -> np.ndarray:
    """
    Approximate lung-field mask by thresholding dark air-filled regions inside
    the body, then removing components touching the border (that is the outside
    air, not lung). Heuristic and labelled as such in the UI.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    h, w = gray.shape
    keep = np.zeros_like(dark)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        touches_border = (x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1)
        if not touches_border and area > 0.004 * h * w:
            keep[lab == i] = 255
    return cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def measured_parameters(gray: np.ndarray) -> dict:
    """
    Objectively MEASURED image statistics. These are computed from pixels and
    involve no model, so the UI can present them separately from predictions.
    """
    m = lung_mask(gray)
    frac = float((m > 0).mean())
    inside = gray[m > 0]
    return {
        "width_px": int(gray.shape[1]),
        "height_px": int(gray.shape[0]),
        "intensity_mean": round(float(gray.mean()), 2),
        "intensity_std": round(float(gray.std()), 2),
        "intensity_min": int(gray.min()),
        "intensity_max": int(gray.max()),
        "intensity_p05": round(float(np.percentile(gray, 5)), 1),
        "intensity_p95": round(float(np.percentile(gray, 95)), 1),
        "lung_field_detected": bool(frac > 0.02),
        "lung_field_area_fraction": round(frac, 4),
        "lung_field_mean_intensity": round(float(inside.mean()), 2) if inside.size else None,
        "contrast_rms": round(float(gray.std() / max(gray.mean(), 1e-6)), 3),
        "sharpness_laplacian_var": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
    }


# ---------------------------------------------------------------------------
# Domain harmonisation (added in the domain-shift redesign)
#
# Measured motivation, from scripts/22 and scripts/23:
#   whole slice        -> a linear probe names the source dataset at AUC 1.000
#   lung-masked        -> 0.939
#   nodule-scale ROI   -> 0.937   (so ROI cropping alone does NOT fix it)
#   lung + quantile    -> 0.857   and the benign/malignant signal actually RISES
#                                 from 0.559 to 0.679, because removing the
#                                 appearance confound leaves structure behind.
#
# No preprocessing tested reached 0.5, so this is a necessary but not sufficient
# step; it is paired with domain-adversarial training.
# ---------------------------------------------------------------------------

def quantile_harmonise(gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Map the lung-field intensity distribution onto a flat reference.

    Rank-transforming intensities inside the lung field discards absolute
    brightness, contrast and the shape of the intensity histogram - the scanner
    and compression signatures - while preserving the ORDERING of tissue
    densities, which is what distinguishes a nodule from parenchyma.

    Pixels outside the lung field are zeroed: they carry body habitus, table,
    and text overlays, none of which is diagnostic and all of which is
    site-specific.
    """
    if mask is None:
        mask = lung_mask(gray)
    out = gray.astype(np.float32)
    m = mask > 0
    if m.sum() < 50:                      # no usable lung field - leave as-is
        return gray
    v = out[m]
    ranks = (np.argsort(np.argsort(v)) + 0.5) / len(v)
    out[m] = np.clip(ranks * 255.0, 0, 255)
    out[~m] = 0
    return out.astype(np.uint8)


def preprocess_harmonised(gray: np.ndarray, size: int = IMG_SIZE,
                          use_denoise: bool = True) -> np.ndarray:
    """
    Domain-harmonised preprocessing.

    Deliberately does NOT apply CLAHE: CLAHE amplifies local contrast, and local
    contrast statistics are part of the scanner signature this path is trying to
    remove. The rank transform supersedes it.
    """
    g = denoise(gray) if use_denoise else gray
    g = resize_letterbox(g, size)
    return quantile_harmonise(g)
