"""
Nodule segmentation inference.

The U-Net is trained on LIDC-IDRI radiologist consensus contours, which is the
only pixel-level ground truth available to this project. IQ-OTH/NCCD has no
masks, so nothing here was trained or validated on that domain.

That distinction drives everything in this module: the output carries the
in-domain Dice/IoU it actually earned, plus the measured cross-domain transfer
verdict, so the UI can state exactly how much weight the result deserves for
the image in front of the user.

Detections outside the detected lung field are discarded before reporting -
a "nodule" in the chest wall or outside the body is a false positive by
construction, and filtering it is honest cleanup rather than cherry-picking.
"""
from __future__ import annotations
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config import MODELS, EXPERIMENTS, REPORTS
from src.models.unet import UNet
from src.preprocessing.medical_image import lung_mask

SEG_SIZE = 512
MIN_BLOB_PX = 12          # below this a detection is noise, not a nodule


class NoduleSegmenter:
    """Loads the U-Net once and segments single slices."""

    def __init__(self, device: torch.device | None = None):
        ck_path = MODELS / "unet_nodule.pt"
        if not ck_path.exists():
            raise FileNotFoundError("models/unet_nodule.pt missing "
                                    "- run scripts/15_train_unet.py")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ck_path, map_location=self.device, weights_only=False)
        self.model = UNet(1, base=ck.get("base", 24)).to(self.device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.threshold = float(ck.get("threshold", 0.5))
        self.size = int(ck.get("size", SEG_SIZE))
        self.test_dice = ck.get("test_dice")
        self.test_iou = ck.get("test_iou")

        # Decomposed error analysis. A single Dice number conflates "did it find
        # the nodule" with "how good is the outline", and for this model those
        # two answers are very different - so both are surfaced.
        self.error_analysis = {}
        ea = REPORTS / "segmentation_error_analysis.json"
        if ea.exists():
            try:
                self.error_analysis = json.load(open(ea))
            except Exception:
                pass

        # Measured transfer behaviour, so the UI can be honest per-domain.
        self.transfer = {}
        f = EXPERIMENTS / "unet_segmentation.json"
        if f.exists():
            try:
                rec = json.load(open(f))
                self.transfer = {
                    "verdict": rec.get("transfer_verdict"),
                    "malignant_to_normal_firing_ratio":
                        rec.get("malignant_to_normal_firing_ratio"),
                    "probe": rec.get("cross_domain_probe", {}),
                    "in_domain": rec.get("in_domain_test", {}),
                    "split": rec.get("split", {}),
                }
            except Exception:
                pass

    @torch.no_grad()
    def segment(self, gray: np.ndarray, restrict_to_lung: bool = True) -> dict:
        """
        gray : uint8 HxW (already preprocessed to the analysis resolution)
        Returns a structured result including the mask, per-lesion geometry and
        the honesty metadata the dashboard needs.
        """
        h0, w0 = gray.shape[:2]
        img = (cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
               if (h0, w0) != (self.size, self.size) else gray)

        x = torch.from_numpy(img.astype(np.float32) / 255.0)[None, None].to(self.device)
        prob = torch.sigmoid(self.model(x).float())[0, 0].cpu().numpy()

        det = (prob > self.threshold).astype(np.uint8)
        lung = lung_mask(img)
        lung_frac = float((lung > 0).mean())
        outside = int(((det > 0) & (lung == 0)).sum())
        if restrict_to_lung and lung_frac > 0.02:
            det = ((det > 0) & (lung > 0)).astype(np.uint8)

        n, lab, stats, cents = cv2.connectedComponentsWithStats(det, 8)
        lesions = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < MIN_BLOB_PX:
                continue
            x0, y0, bw, bh = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                              int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
            cx, cy = float(cents[i][0]), float(cents[i][1])
            comp = (lab == i)
            conf = float(prob[comp].mean())
            # laterality: radiological convention (patient's left = image right)
            rx = cx / self.size
            side = ("Right lung (patient's right)" if rx < 0.42 else
                    "Left lung (patient's left)" if rx > 0.58 else
                    "Central / mediastinal")
            ry = cy / self.size
            zone = "Upper zone" if ry < 0.37 else ("Mid zone" if ry < 0.67 else "Lower zone")
            lesions.append({
                "area_px": area,
                "area_fraction_of_image": round(area / (self.size ** 2), 6),
                "area_fraction_of_lung": (round(area / max((lung > 0).sum(), 1), 5)
                                          if lung_frac > 0.02 else None),
                "equivalent_diameter_px": round(float(2 * np.sqrt(area / np.pi)), 1),
                "bbox_xyxy": [x0, y0, x0 + bw, y0 + bh],
                "centroid_xy": [int(cx), int(cy)],
                "mean_confidence": round(conf, 4),
                "peak_confidence": round(float(prob[comp].max()), 4),
                "laterality": side, "zone": zone,
                "location": f"{side}, {zone.lower()}",
            })
        lesions.sort(key=lambda l: -l["area_px"])

        mask_full = (cv2.resize(det * 255, (w0, h0), interpolation=cv2.INTER_NEAREST)
                     if (h0, w0) != (self.size, self.size) else det * 255)

        return {
            "available": True,
            "model": "U-Net (base 24, depth 4)",
            "trained_on": "LIDC-IDRI radiologist consensus contours (>=2 of 4 readers)",
            "threshold": self.threshold,
            "n_lesions": len(lesions),
            "lesions": lesions,
            "total_area_fraction": round(sum(l["area_fraction_of_image"] for l in lesions), 6),
            "lung_field_detected": lung_frac > 0.02,
            "detections_outside_lung_discarded": outside > 0 and restrict_to_lung,
            "_mask": mask_full,
            "_prob": prob,
            "in_domain_performance": {
                "dice": self.test_dice, "iou": self.test_iou,
                "measured_on": "held-out LIDC patients (patient-level split)",
                "detection_rate": self.error_analysis.get("detection_rate"),
                "dice_when_detected": self.error_analysis.get("dice_over_detected_only"),
                "median_centroid_error_mm":
                    self.error_analysis.get("median_centroid_error_mm"),
                "false_positive_blobs_per_slice":
                    self.error_analysis.get("false_positive_blobs_per_slice"),
                "diagnosis": (self.error_analysis.get("diagnosis") or {}).get("headline"),
            },
            "transfer": self.transfer,
        }


def overlay_mask(gray: np.ndarray, mask: np.ndarray,
                 color=(255, 64, 64), alpha: float = 0.45) -> np.ndarray:
    """Contour + translucent fill over the CT, so anatomy stays visible."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    m = (mask > 127).astype(np.uint8)
    if m.sum() == 0:
        return rgb
    tint = rgb.copy()
    tint[m > 0] = color
    out = cv2.addWeighted(tint, alpha, rgb, 1 - alpha, 0)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2)
    return out


_SEG: NoduleSegmenter | None = None


def get_segmenter() -> NoduleSegmenter | None:
    """Lazy singleton. Returns None when the U-Net has not been trained, so the
    rest of the platform degrades to 'segmentation not available'."""
    global _SEG
    if _SEG is None:
        try:
            _SEG = NoduleSegmenter()
        except Exception:
            return None
    return _SEG
