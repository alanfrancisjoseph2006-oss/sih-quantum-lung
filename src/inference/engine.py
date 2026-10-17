"""
End-to-end inference engine.

Runs a single uploaded image through every branch and returns a structured
report the web dashboard renders directly:

    image -> preprocess -> CNN -> embedding -> PCA
                                   |-> classical heads (RF / SVM / XGB / LogReg)
                                   |-> quantum circuit (PennyLane)
                                   '-> hybrid fusion
                             + Grad-CAM++ explainability
                             + measured (non-model) image parameters

Everything the model cannot legitimately produce -- staging, segmentation,
subtype -- is returned as an explicit "not supported" block rather than being
invented.
"""
from __future__ import annotations
import json
import pickle
import threading
import time
from pathlib import Path

import numpy as np
import torch

from src.config import (MODELS, REPORTS, CLASSES, IMG_SIZE, DISCLAIMER,
                        CLASS_DISPLAY, CLASS_RISK_BAND, UNSUPPORTED_CAPABILITIES,
                        Q_LAYERS, ATTENTION_LABEL)
from src.models.cnn import LungCNN
from src.quantum.circuits import QuantumClassifier
from src.preprocessing.medical_image import (load_image, preprocess, to_model_tensor,
                                             measured_parameters, ImageLoadError)
from src.explainability.gradcam import (GradCAM, overlay_heatmap, attention_region,
                                        to_png_b64, gray_to_png_b64)
from src.inference.segmentation import get_segmenter, overlay_mask


# ---------------------------------------------------------------------------
# Input resolution gate.
#
# MEASURED, not assumed. The training images are 512x512 and are letterboxed to
# 300x300 for the model. An image that arrives SMALLER than the analysis
# resolution has to be upscaled, and the interpolation smoothing that introduces
# is read by the network as pathology.
#
# Sweep over 10 ground-truth NORMAL held-out images (scripts/20_resolution_guard.py):
#     512 px -> 9/10 correct        350 px -> 4/10 correct
#     400 px -> 8/10 correct        320 px and below -> collapses
#     250x189 -> 12/12 flipped to "malignant"
#
# So anything under ~400 px on its short side is outside the range this model was
# validated on, and the platform must say so instead of returning a confident
# number. This is the failure mode behind web-sourced images being called
# malignant.
MIN_VALIDATED_PX = 400      # at or above this, the model is inside its tested range
MIN_USABLE_PX = 256         # below this the output is not presented as actionable


def validate_input_resolution(width: int, height: int) -> dict:
    """Grade an upload against the resolutions this model was validated on."""
    short = min(int(width), int(height))
    if short >= MIN_VALIDATED_PX:
        status, headline = "ok", "Within validated resolution range."
        detail = (f"Short side {short} px meets the {MIN_VALIDATED_PX} px minimum "
                  "this model was validated at.")
    elif short >= MIN_USABLE_PX:
        status, headline = "degraded", "Below validated resolution - result unreliable."
        detail = (f"Short side is {short} px; this model was validated at "
                  f"{MIN_VALIDATED_PX} px and above. On held-out images that were "
                  "downscaled into this range, the classifier drifted toward "
                  "'benign' and 'malignant' even on normal lungs. Treat the "
                  "prediction below as unreliable.")
    else:
        status, headline = "unreliable", "Resolution too low for a meaningful result."
        detail = (f"Short side is only {short} px. Downscaled test images in this "
                  "range flipped to 'malignant' 100% of the time even when the "
                  "ground truth was normal. No risk assessment is presented for "
                  "this input.")
    return {
        "status": status,
        "headline": headline,
        "detail": detail,
        "short_side_px": short,
        "min_validated_px": MIN_VALIDATED_PX,
        "min_usable_px": MIN_USABLE_PX,
        "prediction_is_reliable": status == "ok",
        "evidence": ("Measured by resolution sweep on held-out ground-truth normal "
                     "images; see reports/decisions.md D25."),
    }


def validate_input_plausibility(gray, measured: dict,
                                ood_score: float | None = None,
                                ood_threshold: float | None = None) -> dict:
    """
    Decide whether the input plausibly IS a chest CT slice.

    This exists because the classifier has no reject option. It is a softmax over
    three classes, so it returns a confident answer for literally any input -
    measured behaviour on synthetic inputs at 512x512:

        pure black   -> "benign"    0.87
        pure white   -> "benign"    0.90
        random noise -> "benign"    0.82
        gradient     -> "malignant" 0.94

    None of those are chest CTs. Without a gate, the platform would present a
    risk band for a blank image. These checks are deliberately simple and
    inspectable rather than a learned detector.
    """
    checks = []

    lung = bool(measured.get("lung_field_detected"))
    lung_frac = float(measured.get("lung_field_area_fraction") or 0.0)
    checks.append({
        "id": "lung_field",
        "label": "Lung field detected",
        "passed": lung and 0.03 <= lung_frac <= 0.60,
        "detail": (f"{lung_frac*100:.1f}% of the frame" if lung
                   else "no air-filled lung region found inside the body outline"),
    })

    std = float(measured.get("intensity_std") or 0.0)
    checks.append({
        "id": "structure",
        "label": "Anatomical structure present",
        "passed": std >= 25.0,
        "detail": f"intensity spread {std:.1f} (blank or near-uniform below 25)",
    })

    sharp = float(measured.get("sharpness_laplacian_var") or 0.0)
    checks.append({
        "id": "detail",
        "label": "Sufficient image detail",
        "passed": sharp >= 80.0,
        "detail": f"Laplacian variance {sharp:.0f}",
    })

    # A Mahalanobis out-of-distribution check on the CNN embedding was built and
    # then REMOVED: measured on this model it does not discriminate. In-domain
    # CT scored a median of 19 (max 28) while pure black scored 20.3, white 19.2
    # and random noise 21.6 - complete overlap. It produced false alarms on real
    # held-out scans while passing blank images, so it was worse than no check.
    # The pixel-level checks above do separate these cases and are kept.

    failed = [c for c in checks if not c["passed"]]
    if not failed:
        status = "ok"
        headline = "Input is plausibly a chest CT slice."
    elif len(failed) == 1:
        status = "degraded"
        headline = f"One input check failed: {failed[0]['label'].lower()}."
    else:
        status = "unreliable"
        headline = f"{len(failed)} input checks failed - this may not be a chest CT slice."

    return {
        "status": status,
        "headline": headline,
        "checks": checks,
        "failed": [c["id"] for c in failed],
        "note": ("The classifier has no reject option: it returns a confident answer "
                 "for any image, including a blank one. These checks decide whether "
                 "the answer should be presented at all."),
        "ood_score_diagnostic_only": ood_score,
    }


class ModelNotAvailable(RuntimeError):
    """Raised when the trained artefacts are missing, so the API can 503 cleanly."""


class InferenceEngine:
    """Loads artefacts once and serves predictions. Thread-safe for FastAPI."""

    def __init__(self, backbone_tag: str | None = None):
        self.lock = threading.Lock()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        sel = REPORTS / "model_comparison.json"
        if not sel.exists():
            raise ModelNotAvailable(
                "reports/model_comparison.json missing - run scripts/06_model_selection.py")
        self.comparison = json.load(open(sel))
        dep = self.comparison["deployed"]
        self.tag = backbone_tag or dep["backbone"]
        self.primary_name = dep["primary_model"]

        ck_path = MODELS / f"cnn_{self.tag}.pt"
        br_path = MODELS / f"branches_{self.tag}.pkl"
        q_path = MODELS / f"quantum_{self.tag}.pt"
        for p in (ck_path, br_path, q_path):
            if not p.exists():
                raise ModelNotAvailable(f"missing artefact: {p.name}")

        ck = torch.load(ck_path, map_location=self.device, weights_only=False)
        self.backbone = ck["backbone"]
        self.cnn = LungCNN(backbone=self.backbone, pretrained=False).to(self.device)
        self.cnn.load_state_dict(ck["state_dict"])
        self.cnn.eval()

        with open(br_path, "rb") as fh:
            self.branches = pickle.load(fh)
        self.scaler = self.branches["scaler"]
        self.pca = self.branches["pca"]
        self.classical = self.branches["classical"]
        self.best_classical = self.branches["best_classical"]
        self.meta = self.branches["meta_classifier"]
        self.w_classical = self.branches["weighted_average_w"]

        qck = torch.load(q_path, map_location="cpu", weights_only=False)
        self.n_qubits = qck["n_qubits"]
        self.quantum = QuantumClassifier(qck["in_dim"], self.n_qubits,
                                         len(CLASSES), qck.get("n_layers", Q_LAYERS))
        self.quantum.load_state_dict(qck["state_dict"])
        self.quantum.eval()
        self.circuit_info = qck.get("circuit", {})

        # Validation-set performance of whichever model is deployed, so the UI
        # can state measured performance rather than a marketing number.
        models = self.comparison["backbones"][self.tag]["models"]
        self.primary_metrics = models.get(self.primary_name, {})
        self.localization = None
        loc = REPORTS / "localization_validation.json"
        if loc.exists():
            self.localization = json.load(open(loc))

        # Out-of-distribution reference, fitted from the cached TRAINING
        # embeddings. Mahalanobis distance in embedding space is a standard OOD
        # signal and costs nothing here because the embeddings already exist.
        self.ood_mu = None
        self.ood_inv = None
        self.ood_threshold = None
        emb_path = MODELS / f"embeddings_{self.tag}.npz"
        if emb_path.exists():
            try:
                E = np.load(emb_path)
                Xtr = E["train_X"]
                self.ood_mu = Xtr.mean(0)
                cov = np.cov(Xtr.T) + np.eye(Xtr.shape[1]) * 1e-3
                self.ood_inv = np.linalg.inv(cov)
                d = Xtr - self.ood_mu
                dists = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.ood_inv, d), 0))
                # 99th percentile of training distance: generous, so only clearly
                # foreign inputs trip it.
                self.ood_threshold = float(np.percentile(dists, 99))
            except Exception:
                self.ood_mu = self.ood_inv = self.ood_threshold = None

        # Nodule segmentation U-Net. Optional: if it has not been trained the
        # platform degrades to the honest "segmentation not available" state.
        self.segmenter = get_segmenter()

        # Cost-sensitive decision policy. The deployed model under-called benign
        # lesions as "normal"; a normal-class penalty tuned by grouped CV over
        # train+val fixes most of that. See reports/decision_policy.json.
        self.decision_policy = None
        self.decision_weights = np.ones(len(CLASSES), dtype=np.float64)
        dp = REPORTS / "decision_policy.json"
        if dp.exists():
            try:
                self.decision_policy = json.load(open(dp))
                w = self.decision_policy.get("weights", {})
                self.decision_weights = np.array(
                    [float(w.get(c, 1.0)) for c in CLASSES], dtype=np.float64)
            except Exception:
                self.decision_policy = None

    # ------------------------------------------------------------------
    def _branch_probabilities(self, emb: np.ndarray) -> dict:
        """Run every branch on the 256-d embedding."""
        Z = self.pca.transform(self.scaler.transform(emb))
        out = {}

        for name, mdl in self.classical.items():
            try:
                out[name] = mdl.predict_proba(Z)[0]
            except Exception:
                continue

        with torch.no_grad():
            qp = torch.softmax(self.quantum(torch.tensor(Z, dtype=torch.float32)), 1)
            _, qfeat = self.quantum(torch.tensor(Z, dtype=torch.float32), return_qfeat=True)
        out[f"Quantum_{self.n_qubits}q"] = qp.numpy()[0]
        return out, Z, qfeat.numpy()[0]

    def _apply_policy(self, probs: np.ndarray) -> np.ndarray:
        """Re-weight class posteriors, then renormalise so the reported numbers
        still sum to 1 and remain interpretable as a distribution."""
        w = self.decision_weights * probs
        t = w.sum()
        return w / t if t > 0 else probs

    @staticmethod
    def _add_concordance(seg: dict, predicted_class: str) -> dict:
        """
        Cross-check the two independent models against each other.

        The classifier and the segmenter were trained on different datasets for
        different tasks, so their agreement is genuinely informative rather than
        circular. Disagreement is the useful signal: a lesion segmented on an
        image the classifier called normal is exactly the case a radiologist
        should look at, and it is also how the segmenter's measured ~16%
        false-positive rate on normals surfaces to the user instead of being
        buried in a caveat.
        """
        if not seg.get("available"):
            return seg
        n = seg.get("n_lesions", 0)
        found = n > 0
        lesion_expected = predicted_class in ("benign", "malignant")

        if found and lesion_expected:
            state, note = "agree", (
                f"Both models agree: the classifier reported '{predicted_class}' and the "
                f"segmenter localised {n} candidate region(s).")
        elif not found and not lesion_expected:
            state, note = "agree", (
                "Both models agree: the classifier reported no suspicious finding and the "
                "segmenter localised no nodule-like region.")
        elif found and not lesion_expected:
            state, note = "disagree_seg_positive", (
                f"DISAGREEMENT: the classifier reported '{predicted_class}', but the segmenter "
                f"localised {n} candidate region(s). This is the case most worth a human look. "
                "Note the segmenter is known to fire on a minority of normal images, so this "
                "may equally be a false positive as a missed finding.")
        else:
            state, note = "disagree_seg_negative", (
                f"DISAGREEMENT: the classifier reported '{predicted_class}', but the segmenter "
                "localised no region. The lesion may be subtle, atypical, or outside this "
                "model's training distribution. A negative segmentation is not a clearance.")

        seg.setdefault("interpretation", {})["concordance"] = {
            "state": state, "note": note,
            "classifier_class": predicted_class,
            "segmenter_regions": n,
        }
        return seg

    def _segmentation_block(self, seg, proc) -> dict:
        """
        Build the segmentation payload with its honesty metadata attached.

        The U-Net is trained on LIDC contours; IQ-OTH and most user uploads are a
        DIFFERENT domain with no masks to validate against. So the block always
        carries (a) the in-domain Dice/IoU it genuinely earned and (b) the
        measured cross-domain transfer verdict, and it never presents a mask as
        a confirmed tumour boundary.
        """
        if seg is None:
            return {"available": False,
                    "message": ("Segmentation model not available. Train it with "
                                "scripts/15_train_unet.py; the platform otherwise "
                                "falls back to attention-only localisation.")}
        if not seg.get("available"):
            return seg

        mask = seg.pop("_mask", None)
        seg.pop("_prob", None)
        images = {}
        if mask is not None:
            images["mask"] = gray_to_png_b64(mask)
            images["overlay"] = to_png_b64(overlay_mask(proc, mask))
        seg["images"] = images

        tr = seg.get("transfer") or {}
        ratio = tr.get("malignant_to_normal_firing_ratio")
        probe = tr.get("probe") or {}
        idp = seg.get("in_domain_performance") or {}
        perf_det = idp.get("detection_rate")
        perf_dice = idp.get("dice_when_detected") or 0.0
        seg["interpretation"] = {
            "what_this_is": ("A supervised segmentation predicted by a U-Net trained on "
                             "radiologist-drawn nodule contours from LIDC-IDRI."),
            "what_this_is_not": ("Not a confirmed tumour boundary. It is a model "
                                 "prediction requiring radiologist review, and "
                                 "malignancy cannot be established from a contour."),
            "domain_caveat": (
                "This image is likely from a different scanner, protocol and population "
                "than the LIDC data the model was trained on. "
                + (f"Measured cross-domain behaviour: {tr.get('verdict')}"
                   if tr.get("verdict") else
                   "Cross-domain transfer has not been measured for this build.")),
            "firing_ratio_note": (
                f"On local test data the segmenter fired {ratio}x more often on "
                f"malignant-labelled images than on normal-labelled ones."
                if ratio else None),
            "concordance": None,   # filled in by _add_concordance once the class is known
            "sensitivity_note": (
                f"On held-out patients this model detected only "
                f"{perf_det*100:.0f}% of annotated nodules. When it does detect one it "
                f"outlines it well (Dice {perf_dice:.2f}, centre within ~0.2 mm), but a "
                f"clean result is NOT a clearance - it missed roughly half."
                if perf_det else None),
            "false_positive_note": (
                f"It still fired on {probe.get('normal', {}).get('fire_rate', 0) * 100:.0f}% "
                f"of images labelled normal, so a detection is not proof of a nodule."
                if probe.get("normal") else None),
        }
        return seg

    @staticmethod
    def _fmt(name: str, probs: np.ndarray) -> dict:
        i = int(np.argmax(probs))
        return {"model": name, "predicted_class": CLASSES[i],
                "display": CLASS_DISPLAY[CLASSES[i]],
                "confidence": round(float(probs[i]), 4),
                "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)}}

    # ------------------------------------------------------------------
    def analyze(self, data: bytes, filename: str = "") -> dict:
        """Full pipeline for one image. Raises ImageLoadError on bad input."""
        t0 = time.time()
        li = load_image(data, filename)
        # Grade the ORIGINAL upload, not the letterboxed analysis frame - the
        # analysis frame is always 300x300 and would always look acceptable.
        resolution = validate_input_resolution(*li.original_size)
        proc = preprocess(li.array, IMG_SIZE)
        x = to_model_tensor(proc).to(self.device)

        with self.lock:
            # --- CNN + embedding ---
            with torch.no_grad():
                logits, emb = self.cnn(x, return_embedding=True)
                cnn_probs = torch.softmax(logits.float(), 1).cpu().numpy()[0]
            emb_np = emb.float().cpu().numpy()

            ood_score = None
            if self.ood_mu is not None and self.ood_inv is not None:
                d = emb_np[0] - self.ood_mu
                ood_score = float(np.sqrt(max(d @ self.ood_inv @ d.T, 0)))

            branch_probs, Z, qfeat = self._branch_probabilities(emb_np)

            # --- hybrid fusion (meta-classifier over classical+quantum+qfeat) ---
            bc = branch_probs.get(self.best_classical)
            qp = branch_probs.get(f"Quantum_{self.n_qubits}q")
            hybrid_probs = None
            if bc is not None and qp is not None:
                feats = np.concatenate([bc, qp, qfeat])[None, :]
                try:
                    hybrid_probs = self.meta.predict_proba(feats)[0]
                except Exception:
                    hybrid_probs = (self.w_classical * bc + (1 - self.w_classical) * qp)

            # --- Grad-CAM++ ---
            cam_engine = GradCAM(self.cnn, plus_plus=True)
            try:
                cam, cam_cls, _ = cam_engine(x)
            finally:
                cam_engine.remove()

        heat, overlay = overlay_heatmap(proc, cam)
        region = attention_region(cam, proc.shape[:2])

        # --- nodule segmentation (independent of the classifier) ---
        seg = None
        if self.segmenter is not None:
            try:
                seg = self.segmenter.segment(proc)
            except Exception as e:
                seg = {"available": False,
                       "message": f"Segmentation failed: {type(e).__name__}: {e}"}

        # --- assemble branch results ---
        classical_results = [self._fmt(n, p) for n, p in branch_probs.items()
                             if not n.startswith("Quantum")]
        cnn_result = self._fmt("CNN_DeepLearning", cnn_probs)
        quantum_result = self._fmt(f"Quantum_{self.n_qubits}q",
                                   branch_probs[f"Quantum_{self.n_qubits}q"])
        hybrid_result = (self._fmt("Hybrid_meta_classifier", hybrid_probs)
                         if hybrid_probs is not None else None)

        # --- adaptive router: use the model chosen offline on validation ---
        pool = {r["model"]: r for r in classical_results}
        pool[cnn_result["model"]] = cnn_result
        pool[quantum_result["model"]] = quantum_result
        if hybrid_result:
            pool[hybrid_result["model"]] = hybrid_result
        routed = pool.get(self.primary_name) or cnn_result
        selected_name = routed["model"]

        # Apply the cost-sensitive decision policy to the routed model only.
        # Individual branch cards above keep their raw, unadjusted outputs so a
        # reviewer can always see what each model said before the policy.
        raw_probs = np.array([routed["probabilities"][c] for c in CLASSES])
        adj = self._apply_policy(raw_probs)
        final = self._fmt(selected_name, adj)
        policy_changed = final["predicted_class"] != routed["predicted_class"]

        cls = final["predicted_class"]
        conf = final["confidence"]

        # agreement across branches -- a useful honesty signal for the clinician
        votes = [r["predicted_class"] for r in
                 ([cnn_result, quantum_result] + classical_results +
                  ([hybrid_result] if hybrid_result else []))]
        agree = votes.count(cls) / len(votes)

        meas = measured_parameters(proc)

        plausibility = validate_input_plausibility(proc, meas, ood_score, self.ood_threshold)
        # The gate is the WORSE of the two checks -- a high-resolution picture of
        # a cat should fail just as surely as a 100px chest CT.
        rank = {"ok": 0, "degraded": 1, "unreliable": 2}
        gate_status = max([resolution["status"], plausibility["status"]], key=lambda k: rank[k])
        gate_reasons = []
        if resolution["status"] != "ok":
            gate_reasons.append(resolution["headline"])
        if plausibility["status"] != "ok":
            gate_reasons.append(plausibility["headline"])
        elapsed = time.time() - t0

        return {
            "disclaimer": DISCLAIMER,
            "input_validation": {
                "resolution": resolution,
                "plausibility": plausibility,
                "overall_status": gate_status,
                "prediction_is_reliable": gate_status == "ok",
                "reasons": gate_reasons,
                "guidance": (
                    "Result shown normally." if gate_status == "ok" else
                    "Treat this result as unreliable and re-acquire the image at a "
                    "higher resolution from the original study." if gate_status == "degraded" else
                    "No risk assessment should be acted on for this input. It falls "
                    "outside anything this model was validated on."),
                "note": ("Automated technical checks on the uploaded file. These "
                         "grade whether the input is inside the range this model was "
                         "validated on - they are not a diagnostic image-quality "
                         "certification."),
            },
            "input": {
                "filename": filename,
                **li.provenance(),
                "analysis_resolution": [IMG_SIZE, IMG_SIZE],
            },
            "final_prediction": {
                "selected_model": selected_name,
                "selection_reason": (
                    f"'{selected_name}' was chosen offline on the validation split using a "
                    "significance-aware rule (paired bootstrap, then clinical tie-break on "
                    "malignant sensitivity). The test split was never used for selection."),
                "predicted_class": cls,
                "display": CLASS_DISPLAY[cls],
                "risk_band": CLASS_RISK_BAND[cls],
                "confidence": conf,
                "probabilities": final["probabilities"],
                "branch_agreement": round(agree, 3),
                "branch_agreement_note": (
                    f"{int(agree*100)}% of the model branches agree on this class. "
                    "Low agreement indicates an ambiguous case warranting closer review."),
                "decision_policy_applied": bool(self.decision_policy),
                "decision_policy_changed_outcome": bool(policy_changed),
                "raw_model_class": routed["predicted_class"],
                "raw_model_probabilities": routed["probabilities"],
                "reliability": gate_status,
                "reliability_warning": (
                    None if gate_status == "ok" else " ".join(gate_reasons)),
                "decision_policy_note": (
                    "A cost-sensitive weighting (normal x"
                    f"{self.decision_weights[CLASSES.index('normal')]:.2f}) is applied to the "
                    "routed model, tuned by grouped cross-validation. It raises benign "
                    "sensitivity from 0.73 to 0.93 on the test split at the cost of ~5 points "
                    "of normal specificity - the right trade for screening support."
                    + (" This policy CHANGED the outcome for this image."
                       if policy_changed else "")
                ) if self.decision_policy else "No decision policy applied.",
            },
            "branch_results": {
                "classical_deep_learning": cnn_result,
                "classical_ml": classical_results,
                "quantum": quantum_result,
                "hybrid": hybrid_result,
            },
            "explainability": {
                "method": "Grad-CAM++",
                "target_layer": type(self.cnn.gradcam_target_layer()).__name__,
                "explained_class": CLASSES[cam_cls],
                "label": ATTENTION_LABEL,
                "caveat": ("Regions contributing most strongly to the model prediction. "
                           "This is NOT a segmented or confirmed tumor boundary."),
                "region": region,
                "images": {
                    "original": gray_to_png_b64(proc),
                    "heatmap": to_png_b64(heat),
                    "overlay": to_png_b64(overlay),
                },
            },
            "measured_parameters": meas,
            "measured_parameters_note": (
                "Computed directly from image pixels. No model is involved, so these "
                "are measurements, not predictions."),
            "not_supported": {
                k: v for k, v in UNSUPPORTED_CAPABILITIES.items()
                # Segmentation is no longer unsupported once the LIDC-trained
                # U-Net is available, so it is dropped from this list.
                if not (k == "supervised_segmentation" and self.segmenter is not None)
            },
            "staging": {
                "available": False,
                "message": ("Clinical stage cannot be reliably determined from this "
                            "dataset/model alone. The training data contains no stage "
                            "labels, so no stage is predicted."),
            },
            "segmentation": self._add_concordance(
                self._segmentation_block(seg, proc), cls),
            "model_performance": {
                "note": ("Measured on the held-out test split (n=156) with group-aware "
                         "splitting to prevent near-duplicate leakage."),
                "backbone": self.backbone,
                "primary_model": self.primary_name,
                "test_metrics": self.primary_metrics.get("test", {}),
                "decision_policy": self.decision_policy,
                "quantum_circuit": self.circuit_info,
            },
            "runtime_seconds": round(elapsed, 3),
        }


_ENGINE: InferenceEngine | None = None
_ENGINE_ERR: str | None = None


def get_engine() -> InferenceEngine:
    """Lazy singleton, so the API starts even if artefacts are still training."""
    global _ENGINE, _ENGINE_ERR
    if _ENGINE is None:
        try:
            _ENGINE = InferenceEngine()
            _ENGINE_ERR = None
        except Exception as e:
            _ENGINE_ERR = f"{type(e).__name__}: {e}"
            raise
    return _ENGINE


def engine_status() -> dict:
    try:
        e = get_engine()
        return {"ready": True, "backbone": e.backbone, "primary_model": e.primary_name,
                "device": str(e.device), "n_qubits": e.n_qubits}
    except Exception as e:
        return {"ready": False, "error": f"{type(e).__name__}: {e}"}
