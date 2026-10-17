"""
Clinical information & treatment CONSIDERATIONS module.

This is deliberately NOT a prescription engine. It returns static, educational,
guideline-flavoured information keyed only on the coarse risk band the model
produced. It never names a drug for a patient, never gives a dose, and never
states a stage.

Everything here is generic patient/clinician education of the kind found in
public oncology references. The decision-making language is always conditional
("clinicians may consider"), never directive ("you should take").
"""
from __future__ import annotations

from src.config import DISCLAIMER

TREATMENT_DISCLAIMER = (
    "Treatment decisions require confirmation through pathology, clinical staging, "
    "patient history, and evaluation by qualified oncology professionals. "
    "Nothing here is a prescription or a personalised medical recommendation.")


NEXT_STEPS = {
    "normal": [
        "No suspicious finding was flagged by the model on this single slice.",
        "A negative AI result does not exclude disease - small or early lesions can be missed, "
        "especially on a single 2D slice rather than a full volumetric study.",
        "Routine screening intervals should follow national guidance and individual risk "
        "factors (age, smoking history, occupational exposure, family history).",
        "Any persistent clinical symptoms warrant medical review regardless of this output.",
    ],
    "benign": [
        "Radiologist review of the complete CT series, not just this slice.",
        "Correlation with prior imaging, where available, to assess interval change - "
        "stability over time is one of the strongest indicators of a benign process.",
        "Structured nodule characterisation (size, margin, density, growth rate) by a radiologist.",
        "Interval follow-up imaging may be considered, with timing guided by nodule size and "
        "patient risk, following established protocols such as Fleischner Society guidance.",
    ],
    "malignant": [
        "Prompt review by a radiologist and referral to a pulmonologist or thoracic specialist.",
        "Assessment of the full CT volume, and further diagnostic imaging where indicated.",
        "Tissue diagnosis: biopsy and histopathological confirmation is the definitive step - "
        "imaging alone cannot establish malignancy.",
        "Staging workup as directed by the treating team (which may include PET-CT, "
        "brain imaging, and mediastinal assessment).",
        "Discussion at a multidisciplinary tumour board.",
    ],
}


CONSIDERATIONS = {
    "normal": {
        "headline": "No suspicious finding flagged on this image",
        "summary": ("The model did not identify features it associates with a benign or "
                    "malignant lesion on this slice. This is a screening-support signal "
                    "only, not a clearance."),
        "modalities": [],
    },
    "benign": {
        "headline": "Benign-appearing finding flagged",
        "summary": ("The model flagged features it associates with a non-malignant lesion. "
                    "Benign-appearing findings on imaging still require radiological "
                    "characterisation, and many are managed with surveillance rather than "
                    "intervention."),
        "modalities": [
            ("Surveillance imaging",
             "Serial CT at guideline-directed intervals is a common approach for stable, "
             "benign-appearing nodules."),
            ("Further characterisation",
             "Additional imaging or tissue sampling may be considered if features are "
             "indeterminate or change over time."),
        ],
    },
    "malignant": {
        "headline": "Suspicious finding flagged - specialist review indicated",
        "summary": ("The model flagged features it associates with malignancy. This is a "
                    "prompt for expert human review and tissue diagnosis, not a diagnosis. "
                    "The treatment approaches below are general education about what "
                    "clinicians consider once a diagnosis and stage are actually established "
                    "- neither of which this tool provides."),
        "modalities": [
            ("Surgery",
             "Resection (e.g. lobectomy, segmentectomy, wedge resection) is often considered "
             "for early-stage, localised, operable non-small cell lung cancer."),
            ("Radiation therapy",
             "Including stereotactic body radiotherapy for selected early-stage disease, or "
             "as part of combined-modality treatment in more advanced disease."),
            ("Chemotherapy",
             "Platinum-based regimens are widely used, in adjuvant, neoadjuvant, or palliative "
             "settings depending on stage and performance status."),
            ("Targeted therapy",
             "Applies where molecular testing identifies an actionable driver alteration "
             "(for example EGFR, ALK, ROS1, KRAS G12C). Requires tissue or liquid biopsy "
             "molecular profiling - it cannot be inferred from a CT image."),
            ("Immunotherapy",
             "Immune checkpoint inhibitors are used in selected patients, with PD-L1 "
             "expression and other biomarkers informing eligibility."),
        ],
    },
}


CANCER_EDUCATION = {
    "what_is_it": (
        "Lung cancer is an uncontrolled growth of abnormal cells originating in lung tissue, "
        "most often in the cells lining the airways. It is among the leading causes of "
        "cancer-related death worldwide, largely because it is frequently detected late, "
        "once symptoms have appeared."),
    "major_types": [
        ("Non-small cell lung cancer (NSCLC)",
         "Roughly 80-85% of cases. Subtypes include adenocarcinoma (most common, often "
         "peripheral), squamous cell carcinoma (often central, strongly smoking-associated), "
         "and large cell carcinoma."),
        ("Small cell lung cancer (SCLC)",
         "Roughly 10-15% of cases. Grows and spreads rapidly, strongly associated with "
         "smoking, and is usually treated with chemotherapy and radiotherapy rather than surgery."),
        ("Benign lung nodules",
         "Not cancer. Includes granulomas, hamartomas, and infectious or inflammatory "
         "changes. These are common incidental findings and often need only surveillance."),
    ],
    "risk_factors": [
        "Tobacco smoking - by far the dominant risk factor",
        "Second-hand smoke exposure",
        "Radon gas exposure",
        "Occupational exposures: asbestos, silica, diesel exhaust, arsenic, chromium",
        "Outdoor and indoor air pollution",
        "Prior chest radiotherapy",
        "Family history of lung cancer",
        "Chronic lung disease such as COPD or pulmonary fibrosis",
    ],
    "why_early_detection": (
        "Stage at diagnosis is the single strongest determinant of outcome in lung cancer. "
        "Disease confined to the lung is often treatable with curative intent, whereas "
        "metastatic disease is generally not. Low-dose CT screening in high-risk populations "
        "has been shown in large randomised trials to reduce lung-cancer mortality, which is "
        "precisely why computational tools that help prioritise suspicious studies are of "
        "interest."),
    "how_ct_is_used": (
        "CT provides cross-sectional X-ray images of the chest at high spatial resolution, "
        "making it far more sensitive than a chest radiograph for detecting small pulmonary "
        "nodules. Radiologists assess size, shape, margin, density, location and change over "
        "time. A full diagnostic study is a 3D volume of many slices - the analysis in this "
        "platform operates on a single 2D slice, which is a real limitation."),
    "what_ai_means": (
        "The models here learn statistical associations between image patterns and the "
        "labels in their training data. They do not reason about anatomy, physiology or "
        "patient history, and they cannot examine the patient. A model output is a "
        "probability derived from pixel patterns - a prioritisation signal for a human "
        "expert, not an independent diagnostic judgement."),
    "ai_limitations": [
        "Trained on a single-centre dataset from one scanner in one country - performance on "
        "other populations, scanners and protocols is unproven.",
        "Operates on one 2D slice, whereas radiologists read the full 3D volume with clinical context.",
        "The training images are 8-bit JPEG, so true Hounsfield Unit densitometry is unavailable.",
        "The benign class had only 100 usable images, so benign performance carries wide "
        "confidence intervals.",
        "The model cannot determine cancer stage, subtype, or molecular markers - no such "
        "labels existed in the training data.",
        "Attention heatmaps show what influenced the prediction; they are not validated "
        "tumour boundaries, and on out-of-domain lesion data their localisation was measured "
        "to be no better than chance.",
        "The model can be confidently wrong. Confidence is not correctness.",
    ],
    "why_pathology": (
        "Histopathology - examining tissue under a microscope, supported by "
        "immunohistochemistry and molecular testing - is the reference standard for "
        "diagnosing cancer. It establishes whether malignancy is present, what type it is, "
        "and which targeted treatments might apply. Imaging, human or AI-read, can raise or "
        "lower suspicion, but it cannot replace tissue diagnosis."),
}


def clinical_package(predicted_class: str, confidence: float) -> dict:
    """Assemble the clinical information block for a given risk category."""
    c = predicted_class if predicted_class in CONSIDERATIONS else "normal"
    return {
        "risk_category": c,
        "headline": CONSIDERATIONS[c]["headline"],
        "summary": CONSIDERATIONS[c]["summary"],
        "possible_next_steps": NEXT_STEPS[c],
        "treatment_considerations": [
            {"modality": m, "description": d} for m, d in CONSIDERATIONS[c]["modalities"]],
        "treatment_disclaimer": TREATMENT_DISCLAIMER,
        "general_disclaimer": DISCLAIMER,
        "confidence_caveat": (
            "A high confidence score reflects how strongly the model's learned patterns "
            "matched this image. It is not a probability that the patient has cancer, and "
            "it does not account for clinical history, symptoms, or prior imaging."),
    }
