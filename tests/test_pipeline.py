"""
Test suite for the SIH 2026 PS-26139 platform.

Run:  python -m pytest tests/ -v     (or)     python tests/test_pipeline.py

Deliberately covers the error paths and the scientific-integrity guarantees,
not just the happy path: the whole point of the project is that it must not
fabricate capabilities it does not have.
"""
from __future__ import annotations
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CLASSES, SPLITS, REPORTS, MODELS, UNSUPPORTED_CAPABILITIES
from src.preprocessing.medical_image import (load_image, preprocess, resize_letterbox,
                                             measured_parameters, remove_green_overlay,
                                             ImageLoadError, lung_mask)
from src.evaluation.metrics import compute_metrics, fast_f1_macro


def _png_bytes(arr: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG")
    return b.getvalue()


# ---------------------------------------------------------------- preprocessing
def test_letterbox_preserves_aspect_and_size():
    """Non-square input must not be stretched -- lesion shape is diagnostic."""
    img = np.zeros((512, 801), np.uint8)
    img[200:300, 300:400] = 255                     # a square blob
    out = resize_letterbox(img, 300)
    assert out.shape == (300, 300)
    ys, xs = np.nonzero(out > 40)
    h, w = ys.max() - ys.min(), xs.max() - xs.min()
    assert abs(h - w) <= 3, f"aspect distorted: blob became {w}x{h}"


def test_preprocess_is_deterministic():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (512, 512), dtype=np.uint8)
    assert np.array_equal(preprocess(img), preprocess(img))


def test_preprocess_output_contract():
    img = (np.ones((400, 400)) * 120).astype(np.uint8)
    out = preprocess(img)
    assert out.dtype == np.uint8 and out.shape == (300, 300)
    assert 0 <= out.min() and out.max() <= 255


def test_green_overlay_is_removed_and_box_recovered():
    """The DeepLesion annotation must never reach the model."""
    img = np.full((256, 256, 3), 90, np.uint8)
    img[100:140, 110:150] = [0, 255, 0]             # burned-in green box
    cleaned, removed, box = remove_green_overlay(img)
    assert removed is True
    assert box is not None
    a = cleaned.astype(np.int16)
    still_green = ((a[..., 1] > 140) & (a[..., 0] < 110) & (a[..., 2] < 110)).sum()
    assert still_green == 0, "green annotation survived inpainting"


def test_no_false_green_detection_on_grayscale():
    gray_rgb = np.stack([np.full((128, 128), 130, np.uint8)] * 3, -1)
    _, removed, box = remove_green_overlay(gray_rgb)
    assert removed is False and box is None


# ---------------------------------------------------------------- loading / errors
def test_load_rejects_garbage_with_clear_message():
    with pytest.raises(ImageLoadError) as e:
        load_image(b"this is definitely not an image", "bad.png")
    assert "readable" in str(e.value).lower() or "supported" in str(e.value).lower()


def test_load_rejects_truncated_dicom():
    fake = b"\x00" * 128 + b"DICM" + b"\x01\x02\x03"
    with pytest.raises(ImageLoadError):
        load_image(fake, "broken.dcm")


def test_load_valid_png_roundtrip():
    arr = (np.random.default_rng(1).integers(0, 255, (300, 300))).astype(np.uint8)
    li = load_image(_png_bytes(arr), "x.png")
    assert li.array.ndim == 2 and li.source_format == "PNG"
    assert li.is_dicom is False and li.hu_available is False


def test_measured_parameters_are_finite():
    img = (np.random.default_rng(2).integers(0, 255, (300, 300))).astype(np.uint8)
    m = measured_parameters(img)
    for k, v in m.items():
        if isinstance(v, (int, float)) and v is not None:
            assert np.isfinite(v), f"{k} not finite"
    assert 0.0 <= m["lung_field_area_fraction"] <= 1.0


def test_lung_mask_shape_and_binary():
    img = (np.random.default_rng(3).integers(0, 255, (256, 256))).astype(np.uint8)
    m = lung_mask(img)
    assert m.shape == img.shape
    assert set(np.unique(m)).issubset({0, 255})


# ---------------------------------------------------------------- metrics
def test_metrics_perfect_prediction():
    y = np.array([0, 1, 2, 0, 1, 2])
    p = np.eye(3)[y]
    m = compute_metrics(y, p)
    assert m["accuracy"] == 1.0 and m["f1_macro"] == 1.0


def test_metrics_survive_nan_probabilities():
    """A diverged model must produce a bad score, not a crash."""
    y = np.array([0, 1, 2, 0])
    p = np.full((4, 3), np.nan)
    m = compute_metrics(y, p)
    assert np.isfinite(m["accuracy"])


def test_fast_f1_matches_sklearn_version():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 3, 200)
    p = rng.random((200, 3))
    p /= p.sum(1, keepdims=True)
    assert abs(fast_f1_macro(y, p.argmax(1), 3) - compute_metrics(y, p)["f1_macro"]) < 1e-9


def test_specificity_and_sensitivity_present():
    y = np.array([0, 1, 1, 2, 2, 0])
    p = np.eye(3)[np.array([0, 1, 2, 2, 2, 0])]
    m = compute_metrics(y, p)
    for c in CLASSES:
        assert "specificity" in m["per_class"][c]
        assert "recall_sensitivity" in m["per_class"][c]


# ---------------------------------------------------------------- data integrity
@pytest.mark.skipif(not (SPLITS / "splits.csv").exists(), reason="splits not built")
def test_splits_have_zero_group_overlap():
    """The central anti-leakage guarantee of this project."""
    import csv
    groups = {"train": set(), "val": set(), "test": set()}
    with open(SPLITS / "splits.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            groups[r["split"]].add(r["group_id"])
    assert not (groups["train"] & groups["val"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["val"] & groups["test"])


@pytest.mark.skipif(not (SPLITS / "splits.csv").exists(), reason="splits not built")
def test_no_file_appears_in_two_splits():
    import csv
    seen = {}
    with open(SPLITS / "splits.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            assert r["file"] not in seen or seen[r["file"]] == r["split"], \
                f"{r['file']} in both {seen.get(r['file'])} and {r['split']}"
            seen[r["file"]] = r["split"]


@pytest.mark.skipif(not (SPLITS / "splits.csv").exists(), reason="splits not built")
def test_all_classes_present_in_every_split():
    import csv, collections
    d = collections.defaultdict(set)
    with open(SPLITS / "splits.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            d[r["split"]].add(r["label"])
    for s in ("train", "val", "test"):
        assert d[s] == set(CLASSES), f"{s} missing classes: {set(CLASSES) - d[s]}"


# ---------------------------------------------------------------- quantum
def test_quantum_circuit_runs_and_is_differentiable():
    import torch
    from src.quantum.circuits import QuantumClassifier
    m = QuantumClassifier(16, 4)
    m.train()
    x = torch.randn(8, 16)
    logits, qf = m(x, return_qfeat=True)
    assert logits.shape == (8, len(CLASSES)) and qf.shape == (8, 4)
    logits.sum().backward()
    assert m.q.weights.grad is not None
    assert torch.isfinite(m.q.weights.grad).all()


def test_quantum_measurements_are_valid_expectations():
    """Pauli-Z expectation values must lie in [-1, 1]."""
    import torch
    from src.quantum.circuits import QuantumLayer
    q = QuantumLayer(4, 2)
    out = q(torch.rand(6, 4) * np.pi)
    assert out.shape == (6, 4)
    assert out.min() >= -1.0001 and out.max() <= 1.0001


@pytest.mark.parametrize("nq", [4, 6, 8])
def test_qubit_counts_are_configurable(nq):
    import torch
    from src.quantum.circuits import QuantumLayer
    q = QuantumLayer(nq, 2)
    assert q(torch.zeros(2, nq)).shape == (2, nq)
    info = q.circuit_info()
    assert info["n_qubits"] == nq and info["circuit_depth"] > 0


# ---------------------------------------------------------------- integrity guarantees
@pytest.mark.skipif(not (REPORTS / "model_comparison.json").exists(), reason="no comparison")
def test_no_quantum_advantage_is_claimed_without_significance():
    """If a quantum/hybrid advantage is reported, it must carry a p-value."""
    c = json.load(open(REPORTS / "model_comparison.json"))
    for bb in c["backbones"].values():
        for name, qa in bb.get("quantum_advantage_test", {}).items():
            assert "p_value" in qa and "significant_at_0.05" in qa
            if qa["significant_at_0.05"]:
                assert qa["p_value"] < 0.05


@pytest.mark.skipif(not (REPORTS / "model_comparison.json").exists(), reason="no comparison")
def test_selection_never_uses_test_split():
    c = json.load(open(REPORTS / "model_comparison.json"))
    for bb in c["backbones"].values():
        assert "never" in bb["selection_policy"].lower()


def test_unsupported_capabilities_are_declared():
    for k in ("clinical_staging", "supervised_segmentation", "histological_subtype"):
        assert k in UNSUPPORTED_CAPABILITIES


@pytest.mark.skipif(not (MODELS / "cnn_efficientnet_b3.pt").exists(), reason="model not trained")
def test_inference_report_refuses_staging():
    """
    The report must decline what the data cannot support.

    Staging stays permanently unavailable (no stage labels exist anywhere).
    Segmentation is NOW available, because the LIDC-trained U-Net gives genuine
    pixel-level supervision -- but it must still never assert a confirmed tumour
    boundary, which the regex below enforces.
    """
    from src.inference.engine import get_engine
    e = get_engine()
    arr = (np.random.default_rng(5).integers(40, 200, (512, 512))).astype(np.uint8)
    r = e.analyze(_png_bytes(arr), "synthetic.png")
    assert r["staging"]["available"] is False
    assert "not a replacement" in r["disclaimer"].lower()
    blob = json.dumps(r).lower()
    # Word-boundary match for an actual staging claim. A plain "stage i" substring
    # would false-positive on the honest sentence "no stage is predicted".
    import re
    assert not re.search(r"stage\s+(i|ii|iii|iv|1|2|3|4)", blob),         "report appears to assert a clinical cancer stage"
    # The phrase may legitimately appear inside a negation ("NOT a segmented or
    # confirmed tumor boundary"). Only an UN-negated assertion is a violation.
    for m in re.finditer(r"confirmed tumou?r boundary", blob):
        prefix = blob[max(0, m.start() - 60):m.start()]
        assert "not" in prefix, f"un-negated tumour-boundary claim near: ...{prefix}"


@pytest.mark.skipif(not (MODELS / "cnn_efficientnet_b3.pt").exists(), reason="model not trained")
def test_inference_returns_all_branches():
    from src.inference.engine import get_engine
    e = get_engine()
    arr = (np.random.default_rng(6).integers(40, 200, (400, 400))).astype(np.uint8)
    r = e.analyze(_png_bytes(arr), "s.png")
    b = r["branch_results"]
    assert b["classical_deep_learning"] and b["quantum"] and b["classical_ml"]
    probs = r["final_prediction"]["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 0.02
    assert set(probs) == set(CLASSES)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))


# ---------------------------------------------------------------- external data (LIDC)
MERGED = SPLITS / "splits_merged.csv"


@pytest.mark.skipif(not MERGED.exists(), reason="merged splits not built")
def test_external_data_never_enters_val_or_test():
    """
    The core safeguard of the LIDC experiment (D21): external data goes into
    TRAIN only, so val/test stay pure target-domain and a dataset shortcut
    cannot inflate the reported metrics.
    """
    import csv
    with open(MERGED, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        if r["split"] in ("val", "test"):
            assert r.get("source", "IQ-OTH") == "IQ-OTH", \
                f"external row leaked into {r['split']}: {r['file']}"
    assert any(r.get("source") == "LIDC" for r in rows), "no LIDC rows present at all"


@pytest.mark.skipif(not MERGED.exists() or not (SPLITS / "splits.csv").exists(),
                    reason="splits not built")
def test_merged_val_test_identical_to_original():
    """Val/test must be byte-identical to the pre-LIDC manifest, otherwise the
    before/after comparison would not be apples-to-apples."""
    import csv
    orig = {(r["split"], r["file"]) for r in csv.DictReader(open(SPLITS / "splits.csv", newline=""))
            if r["split"] in ("val", "test")}
    merged = {(r["split"], r["file"]) for r in csv.DictReader(open(MERGED, newline=""))
              if r["split"] in ("val", "test")}
    assert orig == merged, "val/test membership changed between manifests"


@pytest.mark.skipif(not MERGED.exists(), reason="merged splits not built")
def test_merged_splits_have_zero_group_overlap():
    import csv
    g = {"train": set(), "val": set(), "test": set()}
    for r in csv.DictReader(open(MERGED, newline="")):
        g[r["split"]].add(r["group_id"])
    assert not (g["train"] & g["val"])
    assert not (g["train"] & g["test"])
    assert not (g["val"] & g["test"])


@pytest.mark.skipif(not (REPORTS / "lidc_experiment.json").exists(),
                    reason="LIDC experiment not run")
def test_lidc_negative_result_is_recorded_not_hidden():
    """A regression that removed the honest reporting of the negative result
    would be a scientific-integrity failure, so it is asserted."""
    d = json.load(open(REPORTS / "lidc_experiment.json"))
    assert "worse" in d["answer"].lower()
    assert "unchanged" in d["decision"].lower() or "not deploy" in d["decision"].lower()
    res = d["results_on_unchanged_iqoth_test"]
    assert res["Hybrid_meta"]["delta"] < 0
    assert res["Hybrid_meta"]["significant"] is True


@pytest.mark.skipif(not (REPORTS / "lidc_experiment.json").exists(),
                    reason="LIDC experiment not run")
def test_deployed_model_is_not_the_lidc_model():
    """The LIDC arm measured worse, so it must not be the deployed configuration."""
    comp = json.load(open(REPORTS / "model_comparison.json"))
    assert "lidc" not in comp["deployed"]["backbone"].lower()


# ---------------------------------------------------------------- segmentation (U-Net)
UNET = MODELS / "unet_nodule.pt"


def test_unet_forward_and_shape():
    import torch
    from src.models.unet import UNet, dice_loss, DiceBCELoss
    m = UNet(1, base=8, depth=3)
    x = torch.randn(2, 1, 128, 128)
    y = m(x)
    assert y.shape == (2, 1, 128, 128), f"U-Net changed spatial size: {y.shape}"
    loss = DiceBCELoss()(y, torch.zeros_like(y))
    loss.backward()
    assert torch.isfinite(loss)


def test_dice_is_one_for_perfect_prediction():
    import torch
    from src.models.unet import dice_score, iou_score
    t = torch.zeros(1, 1, 32, 32)
    t[0, 0, 10:20, 10:20] = 1
    perfect = t.clone()
    assert float(dice_score(perfect, t)[0]) > 0.99
    assert float(iou_score(perfect, t)[0]) > 0.99


def test_dice_is_nan_for_empty_ground_truth():
    """Empty-mask slices must be excluded from the average, not scored 0 or 1."""
    import torch
    import math
    from src.models.unet import dice_score
    t = torch.zeros(1, 1, 16, 16)
    p = torch.zeros(1, 1, 16, 16)
    assert math.isnan(float(dice_score(p, t)[0]))


@pytest.mark.skipif(not UNET.exists(), reason="U-Net not trained")
def test_segmenter_output_contract():
    from src.inference.segmentation import get_segmenter
    seg = get_segmenter()
    assert seg is not None
    arr = (np.random.default_rng(11).integers(40, 200, (512, 512))).astype(np.uint8)
    r = seg.segment(arr)
    assert r["available"] is True
    assert isinstance(r["n_lesions"], int) and r["n_lesions"] >= 0
    for l in r["lesions"]:
        assert l["area_px"] >= 12, "sub-threshold blob was not filtered"
        assert 0.0 <= l["mean_confidence"] <= 1.0
        assert len(l["bbox_xyxy"]) == 4


@pytest.mark.skipif(not UNET.exists(), reason="U-Net not trained")
def test_segmentation_reports_measured_performance_not_claims():
    """The block must carry the Dice/IoU it actually earned, plus caveats."""
    from src.inference.engine import get_engine
    e = get_engine()
    arr = (np.random.default_rng(12).integers(40, 200, (512, 512))).astype(np.uint8)
    r = e.analyze(_png_bytes(arr), "s.png")
    s = r["segmentation"]
    assert s["available"] is True
    perf = s["in_domain_performance"]
    assert perf["dice"] is not None and 0.0 < perf["dice"] < 1.0
    it = s["interpretation"]
    assert "not a confirmed tumour boundary" in it["what_this_is_not"].lower()
    assert it["domain_caveat"]


@pytest.mark.skipif(not (EXPERIMENTS := REPORTS / "experiments").joinpath(
    "unet_segmentation.json").exists(), reason="U-Net experiment not run")
def test_segmentation_metrics_are_recorded_honestly():
    rec = json.load(open(REPORTS / "experiments" / "unet_segmentation.json"))
    ind = rec["in_domain_test"]
    assert 0.0 < ind["dice"] < 1.0, "implausible Dice - would suggest a metric bug"
    assert ind["n_patients"] >= 1
    sp = rec["split"]
    assert sp["patients_train"] + sp["patients_val"] + sp["patients_test"] <= 50
    assert "cross_domain_probe" in rec and rec["transfer_verdict"]


# ------------------------------------------------------- input gate (D25)
def test_resolution_validator_grades_correctly():
    """Thresholds come from a measured sweep, so lock them in."""
    from src.inference.engine import validate_input_resolution
    assert validate_input_resolution(512, 512)["status"] == "ok"
    assert validate_input_resolution(552, 554)["status"] == "ok"
    assert validate_input_resolution(350, 350)["status"] == "degraded"
    assert validate_input_resolution(250, 189)["status"] == "unreliable"
    # graded on the SHORT side, not the long one
    assert validate_input_resolution(2000, 100)["status"] == "unreliable"
    assert validate_input_resolution(512, 512)["prediction_is_reliable"] is True
    assert validate_input_resolution(250, 189)["prediction_is_reliable"] is False


def test_plausibility_rejects_blank_images():
    from src.inference.engine import validate_input_plausibility
    blank = {"lung_field_detected": False, "lung_field_area_fraction": 0.0,
             "intensity_std": 0.0, "sharpness_laplacian_var": 0.0}
    r = validate_input_plausibility(None, blank)
    assert r["status"] == "unreliable"
    assert "lung_field" in r["failed"] and "structure" in r["failed"]

    good = {"lung_field_detected": True, "lung_field_area_fraction": 0.20,
            "intensity_std": 70.0, "sharpness_laplacian_var": 700.0}
    assert validate_input_plausibility(None, good)["status"] == "ok"


@pytest.mark.skipif(not (MODELS / "cnn_efficientnet_b3.pt").exists(), reason="model not trained")
def test_low_resolution_input_is_flagged_not_silently_answered():
    """
    The regression this guards: a 250x189 web image was returned as
    'malignant' at 92% confidence with no warning at all.
    """
    from PIL import Image
    from src.inference.engine import get_engine
    e = get_engine()
    arr = (np.random.default_rng(3).integers(40, 200, (189, 250))).astype(np.uint8)
    r = e.analyze(_png_bytes(arr), "tiny.png")
    iv = r["input_validation"]
    assert iv["overall_status"] == "unreliable"
    assert iv["prediction_is_reliable"] is False
    assert r["final_prediction"]["reliability_warning"]


@pytest.mark.skipif(not (MODELS / "cnn_efficientnet_b3.pt").exists(), reason="model not trained")
def test_blank_image_is_flagged_not_silently_answered():
    """A blank frame must never be presented as an actionable risk assessment."""
    from src.inference.engine import get_engine
    e = get_engine()
    r = e.analyze(_png_bytes(np.zeros((512, 512), np.uint8)), "black.png")
    assert r["input_validation"]["overall_status"] == "unreliable"
    assert r["final_prediction"]["reliability_warning"]


@pytest.mark.skipif(not (MODELS / "cnn_efficientnet_b3.pt").exists(), reason="model not trained")
def test_real_ct_passes_the_gate():
    """The gate must not cry wolf on genuine in-domain scans."""
    import csv
    from src.inference.engine import get_engine
    rows = [r for r in csv.DictReader(open(SPLITS / "splits.csv", newline=""))
            if r["split"] == "test"][:5]
    if not rows:
        pytest.skip("no splits")
    e = get_engine()
    for row in rows:
        with open(row["path"], "rb") as fh:
            r = e.analyze(fh.read(), row["file"])
        assert r["input_validation"]["overall_status"] == "ok", \
            f"false alarm on real CT {row['file']}"


# --------------------------------------------- domain-shift redesign (D27)
def test_quantile_harmonise_removes_absolute_intensity():
    """The rank transform must discard brightness while keeping tissue ordering."""
    from src.preprocessing.medical_image import quantile_harmonise
    rng = np.random.default_rng(0)
    base = rng.integers(40, 200, (128, 128)).astype(np.uint8)
    mask = np.zeros((128, 128), np.uint8); mask[20:110, 20:110] = 255
    a = quantile_harmonise(base, mask)
    # a globally brighter copy must harmonise to (nearly) the same thing
    b = quantile_harmonise(np.clip(base.astype(np.int16) + 40, 0, 255).astype(np.uint8), mask)
    inside = mask > 0
    assert np.corrcoef(a[inside], b[inside])[0, 1] > 0.99, \
        "harmonisation is still sensitive to absolute brightness"
    assert (a[~inside] == 0).all(), "outside-lung pixels must be zeroed"


def test_gradient_reversal_flips_sign():
    import torch
    from src.models.dann import grad_reverse
    x = torch.ones(4, 3, requires_grad=True)
    y = grad_reverse(x, 1.0)
    y.sum().backward()
    assert torch.allclose(x.grad, -torch.ones_like(x)), "gradient was not reversed"


def test_dann_lambda_ramps_from_zero_to_one():
    from src.models.dann import dann_lambda
    assert dann_lambda(0, 100) == 0.0
    assert dann_lambda(100, 100) > 0.9
    # monotonic
    vals = [dann_lambda(i, 100) for i in range(0, 101, 10)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_dann_can_be_disabled():
    import torch
    from src.models.dann import DANNClassifier
    m = DANNClassifier("resnet18", n_classes=2, pretrained=False, use_dann=False)
    m.eval()
    logits, dom = m(torch.randn(2, 3, 128, 128))
    assert logits.shape == (2, 2)
    assert dom is None, "domain head must be inactive when use_dann=False"


@pytest.mark.skipif(not (REPORTS / "external_validation.json").exists(),
                    reason="external validation not run")
def test_readiness_level_is_blocked_by_external_validation():
    """
    The readiness system must not award level 3 while the external CI includes
    chance. This is the guard that stops the product overstating itself.
    """
    from src.evaluation.readiness import assess
    r = assess(REPORTS)
    ext = json.load(open(REPORTS / "external_validation.json"))
    lo = ext["slice_level"]["ci95"][0]
    if lo <= 0.5:
        assert r["level"] < 3, "level 3 awarded despite external CI including chance"
        assert r["clinically_deployable"] is False
    assert 0 <= r["level"] <= 5


@pytest.mark.skipif(not (REPORTS / "domain_signature_probe.json").exists(),
                    reason="probe not run")
def test_domain_signature_findings_are_recorded():
    d = json.load(open(REPORTS / "domain_signature_probe.json"))
    assert d["results"]["whole"]["domain_auc"] > 0.9, \
        "the recorded probe should show a strong whole-image signature"
    assert "conclusion" in d and d["conclusion"]


@pytest.mark.skipif(not (REPORTS / "hybrid_quantum_roi.json").exists(),
                    reason="ROI quantum comparison not run")
def test_quantum_claim_matches_the_measurement():
    """If the quantum arm is not better, nothing may claim that it is."""
    d = json.load(open(REPORTS / "hybrid_quantum_roi.json"))
    for _, v in d["quantum_advantage_test"].items():
        assert "p_value" in v and "significant" in v
        if v["significant"] and v["mean_diff"] < 0:
            # a significant NEGATIVE result must never be described as an advantage
            assert v["mean_diff"] < 0


@pytest.mark.skipif(not (REPORTS / "overfitting_fixes.json").exists(),
                    reason="overfitting comparison not run")
def test_overfitting_comparison_used_grouped_cv():
    """
    The comparison must rest on grouped CV over all patients, not the small
    single split that could not rank strategies.
    """
    d = json.load(open(REPORTS / "overfitting_fixes.json"))
    assert d["n_patients"] >= 50
    assert d["n_rois"] > 500
    assert d["folds"] >= 5
    for name, v in d["results"].items():
        assert 0.0 <= v["roc_auc"] <= 1.0
        assert v["ci95"][0] <= v["roc_auc"] <= v["ci95"][1], f"{name}: AUC outside its CI"


@pytest.mark.skipif(not (REPORTS / "overfitting_fixes.json").exists(),
                    reason="overfitting comparison not run")
def test_best_strategy_is_reported_honestly():
    """Whatever won must actually be the maximum - no cherry-picking."""
    d = json.load(open(REPORTS / "overfitting_fixes.json"))
    best = max(d["results"].items(), key=lambda kv: kv[1]["roc_auc"])
    assert d["best"]["strategy"] == best[0]
    assert abs(d["best"]["roc_auc"] - best[1]["roc_auc"]) < 1e-9


@pytest.mark.skipif(not (REPORTS / "overfitting_check.json").exists(),
                    reason="overfitting check not run")
def test_overfitting_gap_is_measured_and_small():
    """The chosen ROI model must be measured, and its gap must be modest."""
    d = json.load(open(REPORTS / "overfitting_check.json"))
    lr = d["gaps"]["Logistic regression"]
    assert lr["train_auc"] > lr["heldout_auc"] - 0.01, "train should not trail held-out"
    assert lr["gap"] < 0.10, f"train/held-out gap too large: {lr['gap']}"
    # the deliberate positive control must actually show a bigger gap
    ctrl = d["gaps"]["SVM-RBF, high C=100 (deliberately overfit)"]
    assert ctrl["gap"] > lr["gap"], "the overfitting detector failed its positive control"


@pytest.mark.skipif(not (REPORTS / "overfitting_check.json").exists(),
                    reason="overfitting check not run")
def test_roi_result_is_not_claimed_as_cross_dataset():
    """The ROI result is cross-patient. Nothing may present it as cross-dataset."""
    d = json.load(open(REPORTS / "overfitting_check.json"))
    a = d["answer"]
    assert "cross-PATIENT" in a["caveat"] or "cross-patient" in a["caveat"].lower()
    assert "UNCHANGED" in a["deployed_platform_model"]
