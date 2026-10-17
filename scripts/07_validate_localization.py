"""
Phase 6/9 - quantitative validation of Grad-CAM localization.

The IQ-OTH/NCCD training set has no spatial labels at all, so ordinarily one can
only assert that heatmaps "look plausible". Here we exploit the second
sub-dataset: the 197 DeepLesion key-slices carry real lesion bounding boxes
burned in as a green overlay. We:

  1. Inpaint the green boxes out, so the model never sees the annotation.
  2. Run the model + Grad-CAM on the cleaned image.
  3. Score the attention peak against the recovered box (pointing game).

A random-baseline comparison is included, because with a median lesion covering
only 0.2% of the image, any hit rate must be judged against chance.

IMPORTANT INTERPRETATION LIMIT: DeepLesion lesions are general CT lesions across
many body sites and are NOT all lung cancers, and these images carry no class
label. So this measures *spatial attention quality*, not diagnostic accuracy.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (MODELS, REPORTS, FIGURES, DEEPLESION_DIR, IMG_SIZE,
                        CLASSES, set_seed, SEED)
from src.models.cnn import LungCNN
from src.preprocessing.medical_image import load_image, preprocess, to_model_tensor, lung_mask
from src.explainability.gradcam import GradCAM, resize_cam, pointing_game, overlay_heatmap


def load_model(tag: str, device):
    ck = torch.load(MODELS / f"cnn_{tag}.pt", map_location=device, weights_only=False)
    m = LungCNN(backbone=ck["backbone"], pretrained=False).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    boxes = json.load(open(REPORTS / "deeplesion_boxes.json"))
    print(f"lesion boxes: {len(boxes)} from {len({b['patient_id'] for b in boxes})} patients")

    results = {}
    for tag in ("efficientnet_b3", "densenet121"):
        if not (MODELS / f"cnn_{tag}.pt").exists():
            continue
        model = load_model(tag, device)
        cam_engine = GradCAM(model, plus_plus=True)

        hits, energies, rnd_hits, per_file = 0, [], 0, []
        box_fracs = []
        rng = np.random.default_rng(SEED)
        t0 = time.time()

        for b in boxes:
            path = DEEPLESION_DIR / b["file"]
            if not path.exists():
                continue
            li = load_image(str(path))            # green overlay inpainted here
            proc = preprocess(li.array, IMG_SIZE)
            x = to_model_tensor(proc).to(device)

            cam, cls_idx, probs = cam_engine(x)

            # Map the box from original 512x512 coords into the letterboxed
            # IMG_SIZE frame the model actually saw.
            H, W = li.array.shape[:2]
            scale = IMG_SIZE / max(H, W)
            nh, nw = int(round(H * scale)), int(round(W * scale))
            top, left = (IMG_SIZE - nh) // 2, (IMG_SIZE - nw) // 2
            bx = [int(b["x1"] * scale) + left, int(b["y1"] * scale) + top,
                  int(b["x2"] * scale) + left, int(b["y2"] * scale) + top]

            # Is this lesion actually pulmonary? DeepLesion covers the whole
            # body, and a chest-CT classifier cannot reasonably be expected to
            # attend to a liver or abdominal lesion. We therefore score the
            # lung-field subset separately as the fair comparison.
            lm = lung_mask(li.array)
            ccx = min((b["x1"] + b["x2"]) // 2, lm.shape[1] - 1)
            ccy = min((b["y1"] + b["y2"]) // 2, lm.shape[0] - 1)
            in_lung = bool(lm[ccy, ccx] > 0)

            pg = pointing_game(cam, bx, (IMG_SIZE, IMG_SIZE))
            hits += int(pg["hit"])
            energies.append(pg["energy_fraction_in_box"])
            box_fracs.append(pg["box_area_fraction"])

            # chance baseline: a uniformly random peak location
            rx, ry = rng.integers(0, IMG_SIZE), rng.integers(0, IMG_SIZE)
            rnd_hits += int(bx[0] <= rx <= bx[2] and bx[1] <= ry <= bx[3])

            per_file.append({"file": b["file"], "patient_id": b["patient_id"],
                             "hit": pg["hit"], "in_lung_field": in_lung,
                             "energy_in_box": pg["energy_fraction_in_box"],
                             "box_area_fraction": pg["box_area_fraction"],
                             "predicted_class": CLASSES[cls_idx],
                             "confidence": round(float(probs[cls_idx]), 4)})

        cam_engine.remove()
        n = len(per_file)
        hit_rate = hits / max(n, 1)
        rnd_rate = rnd_hits / max(n, 1)
        mean_box = float(np.mean(box_fracs)) if box_fracs else float("nan")
        mean_energy = float(np.mean(energies)) if energies else float("nan")

        # fair subset: pulmonary lesions only
        lung = [p for p in per_file if p["in_lung_field"]]
        lung_stats = {}
        if lung:
            lh = sum(p["hit"] for p in lung) / len(lung)
            le = float(np.mean([p["energy_in_box"] for p in lung]))
            lb = float(np.mean([p["box_area_fraction"] for p in lung]))
            lung_stats = {
                "n_lung_lesions": len(lung),
                "pointing_game_hit_rate": round(lh, 4),
                "chance_hit_rate": round(lb, 5),
                "lift_over_chance": round(lh / max(lb, 1e-6), 2),
                "mean_attention_energy_in_box": round(le, 4),
                "energy_lift_over_area": round(le / max(lb, 1e-6), 2)}

        results[tag] = {
            "n_images": n,
            "n_lung_field_lesions": len(lung),
            "lung_field_subset": lung_stats,
            "n_patients": len({p["patient_id"] for p in per_file}),
            "pointing_game_hit_rate": round(hit_rate, 4),
            "random_baseline_hit_rate": round(rnd_rate, 4),
            "expected_chance_hit_rate": round(mean_box, 4),
            "lift_over_chance": round(hit_rate / max(mean_box, 1e-6), 2),
            "mean_attention_energy_in_box": round(mean_energy, 4),
            "energy_lift_over_area": round(mean_energy / max(mean_box, 1e-6), 2),
            "mean_lesion_area_fraction": round(mean_box, 5),
            "runtime_sec": round(time.time() - t0, 1),
            "per_file": per_file,
        }
        r = results[tag]
        print(f"\n{tag}:  n={n} images / {r['n_patients']} patients  ({r['runtime_sec']}s)")
        print(f"  pointing-game hit rate     : {hit_rate:.3f}")
        print(f"  random-peak baseline       : {rnd_rate:.3f}")
        print(f"  chance rate (= mean box area frac): {mean_box:.5f}")
        print(f"  LIFT over chance           : {r['lift_over_chance']}x")
        print(f"  mean attention energy in box: {mean_energy:.4f} "
              f"({r['energy_lift_over_area']}x the box's area share)")
        if lung_stats:
            print(f"  -- PULMONARY SUBSET ONLY (n={lung_stats['n_lung_lesions']}) --")
            print(f"     hit rate {lung_stats['pointing_game_hit_rate']:.3f} "
                  f"vs chance {lung_stats['chance_hit_rate']:.5f} "
                  f"-> lift {lung_stats['lift_over_chance']}x")
            print(f"     energy in box {lung_stats['mean_attention_energy_in_box']:.4f} "
                  f"-> {lung_stats['energy_lift_over_area']}x area share")

    # ---- qualitative figure: best-localised examples -------------------------
    tag = "densenet121" if "densenet121" in results else next(iter(results), None)
    if tag:
        model = load_model(tag, device)
        eng = GradCAM(model, plus_plus=True)
        top = sorted(results[tag]["per_file"], key=lambda r: -r["energy_in_box"])[:6]
        panels = []
        for rec in top:
            b = next(x for x in boxes if x["file"] == rec["file"])
            li = load_image(str(DEEPLESION_DIR / b["file"]))
            proc = preprocess(li.array, IMG_SIZE)
            cam, _, _ = eng(to_model_tensor(proc).to(device))
            _, ov = overlay_heatmap(proc, cam)
            H, W = li.array.shape[:2]
            s = IMG_SIZE / max(H, W)
            nh, nw = int(round(H * s)), int(round(W * s))
            t_, l_ = (IMG_SIZE - nh) // 2, (IMG_SIZE - nw) // 2
            cv2.rectangle(ov, (int(b["x1"] * s) + l_, int(b["y1"] * s) + t_),
                          (int(b["x2"] * s) + l_, int(b["y2"] * s) + t_), (0, 255, 0), 2)
            panels.append(ov)
        eng.remove()
        if panels:
            rows = [np.hstack(panels[i:i + 3]) for i in range(0, len(panels), 3)]
            w = min(r.shape[1] for r in rows)
            grid = np.vstack([r[:, :w] for r in rows])
            cv2.imwrite(str(FIGURES / "localization_validation.png"),
                        cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            print(f"\nwrote figures/localization_validation.png "
                  f"(green box = true lesion, heat = model attention)")

    with open(REPORTS / "localization_validation.json", "w") as fh:
        json.dump({"note": ("Grad-CAM++ attention scored against real DeepLesion "
                            "lesion boxes. Green overlays were inpainted before "
                            "inference so the model could not see the annotation. "
                            "DeepLesion lesions are general CT lesions, not "
                            "necessarily lung cancer, and carry no class label: "
                            "this measures spatial attention quality only."),
                   "results": results}, fh, indent=1)
    print("wrote reports/localization_validation.json")


if __name__ == "__main__":
    main()
