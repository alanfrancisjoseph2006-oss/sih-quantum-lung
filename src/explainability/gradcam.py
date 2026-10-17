"""
Grad-CAM / Grad-CAM++ explainability.

LANGUAGE DISCIPLINE (Step 7 + 13 of the brief)
----------------------------------------------
The dataset has NO tumor masks, so nothing here is segmentation. Every output
is described as "model attention", never as a tumor boundary. The
`ATTENTION_LABEL` constant is the only phrasing the UI is allowed to use.

What makes this implementation more than decoration: `pointing_game` scores the
attention map against the REAL lesion boxes recovered from the DeepLesion
overlays, so the heatmaps are quantitatively validated rather than merely
eyeballed.
"""
from __future__ import annotations
import base64
import io

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.config import GRADCAM_ALPHA, ATTENTION_LABEL


class GradCAM:
    """
    Standard Grad-CAM with an optional Grad-CAM++ weighting.

    Hooks the last spatial conv layer, captures activations on the forward pass
    and gradients on the backward pass, then weights channels by pooled
    gradients.
    """

    def __init__(self, model, target_layer=None, plus_plus: bool = True):
        self.model = model
        self.model.eval()
        self.layer = target_layer or model.gradcam_target_layer()
        self.plus_plus = plus_plus
        self.acts = None
        self.grads = None
        self._handles = [
            self.layer.register_forward_hook(self._fwd),
            self.layer.register_full_backward_hook(self._bwd),
        ]

    def _fwd(self, _m, _i, out):
        self.acts = out.detach()

    def _bwd(self, _m, _gi, go):
        self.grads = go[0].detach()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """Returns (cam HxW in [0,1], predicted_class, probabilities)."""
        # Grad-CAM needs gradients, so autocast/no_grad must be off here.
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            probs = torch.softmax(logits, 1)
            if class_idx is None:
                class_idx = int(logits.argmax(1)[0])
            self.model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward(retain_graph=False)

        if self.acts is None or self.grads is None:
            raise RuntimeError("Grad-CAM hooks captured nothing - check the target layer.")

        acts, grads = self.acts[0], self.grads[0]          # (C,H,W)

        if self.plus_plus:
            # Grad-CAM++ weights: better localisation when several small regions
            # contribute, which suits multifocal lung findings.
            g2, g3 = grads ** 2, grads ** 3
            denom = 2.0 * g2 + (acts * g3).sum(dim=(1, 2), keepdim=True)
            alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
            w = (alpha * F.relu(grads)).sum(dim=(1, 2))
        else:
            w = grads.mean(dim=(1, 2))

        cam = F.relu((w[:, None, None] * acts).sum(0))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), class_idx, probs[0].detach().cpu().numpy()


def resize_cam(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize CAM to (w,h) and renormalise."""
    c = cv2.resize(cam.astype(np.float32), size, interpolation=cv2.INTER_CUBIC)
    c = np.clip(c, 0, None)
    return c / (c.max() + 1e-8)


def overlay_heatmap(gray: np.ndarray, cam: np.ndarray,
                    alpha: float = GRADCAM_ALPHA) -> tuple[np.ndarray, np.ndarray]:
    """Returns (heatmap_rgb, overlay_rgb) at the resolution of `gray`."""
    c = resize_cam(cam, (gray.shape[1], gray.shape[0]))
    hm = cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    # Blend proportionally to activation so low-attention areas stay readable
    # as anatomy rather than being washed out in blue.
    w = (c[..., None] * alpha)
    ov = (base * (1 - w) + hm * w).astype(np.uint8)
    return hm, ov


def attention_region(cam: np.ndarray, shape: tuple[int, int],
                     threshold: float = 0.6) -> dict:
    """
    Summarise where the model looked -- as geometry, not as a diagnosis.

    Returns the peak point, a bounding box of the largest high-attention blob,
    the fraction of the image attended, and a laterality/zone description.
    """
    h, w = shape
    c = resize_cam(cam, (w, h))
    mask = (c >= threshold).astype(np.uint8)
    if mask.sum() == 0:
        threshold = float(c.max() * 0.8)
        mask = (c >= threshold).astype(np.uint8)

    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        py, px = np.unravel_index(int(np.argmax(c)), c.shape)
        return {"available": False, "peak_xy": [int(px), int(py)],
                "attended_area_fraction": 0.0}

    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = stats[k]
    cx, cy = cents[k]
    py, px = np.unravel_index(int(np.argmax(c)), c.shape)

    # Laterality. In radiological convention the patient's LEFT appears on the
    # RIGHT of the image, so the mapping is deliberately inverted here.
    rel_x = cx / w
    if rel_x < 0.40:
        side = "Right lung region (patient's right)"
    elif rel_x > 0.60:
        side = "Left lung region (patient's left)"
    else:
        side = "Central / mediastinal region"

    rel_y = cy / h
    zone = ("Upper zone" if rel_y < 0.37 else
            "Mid zone" if rel_y < 0.67 else "Lower zone")

    return {
        "available": True,
        "peak_xy": [int(px), int(py)],
        "centroid_xy": [int(cx), int(cy)],
        "bbox_xyxy": [int(x), int(y), int(x + bw), int(y + bh)],
        "attended_area_px": int(area),
        "attended_area_fraction": round(float(area) / (h * w), 4),
        "laterality": side,
        "zone": zone,
        "location_description": f"{side}, {zone.lower()}",
        "threshold_used": round(float(threshold), 3),
        "n_attention_blobs": int(n - 1),
        "label": ATTENTION_LABEL,
        "caveat": ("Attention indicates image regions that most influenced the "
                   "model's output. It is NOT a segmented or confirmed tumor boundary."),
    }


def pointing_game(cam: np.ndarray, box_xyxy, shape: tuple[int, int]) -> dict:
    """
    Quantitative localization check against a REAL annotation box.

    'hit' = the single most-activated pixel falls inside the ground-truth box.
    This is the standard weakly-supervised localization metric, and it is what
    lets us report a measured number instead of claiming the heatmaps look good.
    """
    h, w = shape
    c = resize_cam(cam, (w, h))
    py, px = np.unravel_index(int(np.argmax(c)), c.shape)
    x1, y1, x2, y2 = box_xyxy
    hit = bool(x1 <= px <= x2 and y1 <= py <= y2)

    # Also report energy fraction: how much total attention mass lands in the box.
    total = float(c.sum())
    inside = float(c[max(0, y1):y2 + 1, max(0, x1):x2 + 1].sum())
    box_area_frac = ((x2 - x1 + 1) * (y2 - y1 + 1)) / (h * w)
    return {"hit": hit, "peak_xy": [int(px), int(py)],
            "energy_fraction_in_box": round(inside / max(total, 1e-8), 4),
            "box_area_fraction": round(box_area_frac, 4)}


def to_png_b64(rgb: np.ndarray) -> str:
    """Encode an RGB array as a base64 PNG data URI for the web dashboard."""
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def gray_to_png_b64(gray: np.ndarray) -> str:
    return to_png_b64(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
