/**
 * Domain types for the QuantumLung clinical platform.
 *
 * These mirror the FastAPI response contracts exactly. Where the backend is
 * honest about a capability being unavailable (staging, subtype), the type
 * system makes that explicit rather than optional-and-forgotten.
 */

export type RiskBand = "Low" | "Intermediate" | "High";
export type LungClass = "normal" | "benign" | "malignant";

/* ---------------------------------------------------------------- analysis */

export interface BranchResult {
  model: string;
  predicted_class: LungClass;
  display: string;
  confidence: number;
  probabilities: Record<LungClass, number>;
}

export interface AttentionRegion {
  available: boolean;
  peak_xy?: [number, number];
  centroid_xy?: [number, number];
  bbox_xyxy?: [number, number, number, number];
  attended_area_px?: number;
  attended_area_fraction: number;
  laterality?: string;
  zone?: string;
  location_description?: string;
  threshold_used?: number;
  n_attention_blobs?: number;
  label: string;
  caveat: string;
}

export interface Lesion {
  area_px: number;
  area_fraction_of_image: number;
  area_fraction_of_lung: number | null;
  equivalent_diameter_px: number;
  bbox_xyxy: [number, number, number, number];
  centroid_xy: [number, number];
  mean_confidence: number;
  peak_confidence: number;
  laterality: string;
  zone: string;
  location: string;
}

export interface SegmentationResult {
  available: boolean;
  message?: string;
  model?: string;
  trained_on?: string;
  threshold?: number;
  n_lesions?: number;
  lesions?: Lesion[];
  total_area_fraction?: number;
  lung_field_detected?: boolean;
  images?: { mask?: string; overlay?: string };
  in_domain_performance?: {
    dice: number | null;
    iou: number | null;
    measured_on: string;
    /** Detection rate matters as much as Dice — always show both. */
    detection_rate: number | null;
    dice_when_detected: number | null;
    median_centroid_error_mm: number | null;
    false_positive_blobs_per_slice: number | null;
    diagnosis: string | null;
  };
  transfer?: {
    verdict?: string;
    malignant_to_normal_firing_ratio?: number;
    probe?: Record<string, { n: number; fired: number; fire_rate: number }>;
  };
  interpretation?: {
    what_this_is: string;
    what_this_is_not: string;
    domain_caveat: string;
    firing_ratio_note: string | null;
    false_positive_note: string | null;
    sensitivity_note: string | null;
    concordance?: {
      state: "agree" | "disagree_seg_positive" | "disagree_seg_negative";
      note: string;
      classifier_class: LungClass;
      segmenter_regions: number;
    };
  };
}

export interface MeasuredParameters {
  width_px: number;
  height_px: number;
  intensity_mean: number;
  intensity_std: number;
  intensity_min: number;
  intensity_max: number;
  intensity_p05: number;
  intensity_p95: number;
  lung_field_detected: boolean;
  lung_field_area_fraction: number;
  lung_field_mean_intensity: number | null;
  contrast_rms: number;
  sharpness_laplacian_var: number;
}

export interface InputValidationCheck {
  id: string; label: string; passed: boolean; detail: string;
}

/**
 * Input gate. The classifier has no reject option, so these checks decide
 * whether its answer should be presented as actionable at all.
 */
export interface InputValidation {
  resolution: {
    status: "ok" | "degraded" | "unreliable";
    headline: string; detail: string;
    short_side_px: number; min_validated_px: number; min_usable_px: number;
    prediction_is_reliable: boolean; evidence: string;
  };
  plausibility: {
    status: "ok" | "degraded" | "unreliable";
    headline: string; checks: InputValidationCheck[]; failed: string[]; note: string;
  };
  overall_status: "ok" | "degraded" | "unreliable";
  prediction_is_reliable: boolean;
  reasons: string[];
  guidance: string;
  note: string;
}

export interface AnalysisResult {
  input_validation: InputValidation;
  disclaimer: string;
  input: {
    filename: string;
    source_format: string;
    original_size: [number, number];
    is_dicom: boolean;
    hu_available: boolean;
    windowing_applied: string;
    overlay_removed: boolean;
    dicom_meta: Record<string, string | number>;
    notes: string[];
    analysis_resolution: [number, number];
  };
  final_prediction: {
    selected_model: string;
    selection_reason: string;
    predicted_class: LungClass;
    display: string;
    risk_band: RiskBand;
    confidence: number;
    probabilities: Record<LungClass, number>;
    branch_agreement: number;
    branch_agreement_note: string;
    reliability: "ok" | "degraded" | "unreliable";
    reliability_warning: string | null;
    decision_policy_applied: boolean;
    decision_policy_changed_outcome: boolean;
    raw_model_class: LungClass;
    raw_model_probabilities: Record<LungClass, number>;
    decision_policy_note: string;
  };
  branch_results: {
    classical_deep_learning: BranchResult;
    classical_ml: BranchResult[];
    quantum: BranchResult;
    hybrid: BranchResult | null;
  };
  explainability: {
    method: string;
    target_layer: string;
    explained_class: LungClass;
    label: string;
    caveat: string;
    region: AttentionRegion;
    images: { original: string; heatmap: string; overlay: string };
  };
  measured_parameters: MeasuredParameters;
  measured_parameters_note: string;
  not_supported: Record<string, string>;
  /** Permanently unavailable — no stage labels exist in any training data. */
  staging: { available: false; message: string };
  segmentation: SegmentationResult;
  model_performance: {
    note: string;
    backbone: string;
    primary_model: string;
    test_metrics: Record<string, number>;
    decision_policy: unknown;
    quantum_circuit: QuantumCircuitInfo;
  };
  clinical_information: ClinicalInformation;
  ai_limitations: string[];
  runtime_seconds: number;
}

/* ---------------------------------------------------------------- clinical */

export interface ClinicalInformation {
  risk_category: LungClass;
  headline: string;
  summary: string;
  possible_next_steps: string[];
  treatment_considerations: { modality: string; description: string }[];
  treatment_disclaimer: string;
  general_disclaimer: string;
  confidence_caveat: string;
}

/* ------------------------------------------------------------------ quantum */

export interface QuantumCircuitInfo {
  n_qubits: number;
  n_variational_layers: number;
  encoding: string;
  entanglement: string;
  measurement: string;
  backend: string;
  trainable_circuit_params: number;
  differentiation: string;
  gate_count?: number;
  circuit_depth?: number;
  gate_types?: Record<string, number>;
}

/* ------------------------------------------------------------------- meta */

export interface PlatformMeta {
  project: {
    name: string;
    problem_statement_id: string;
    team: string;
    event: string;
  };
  classes: LungClass[];
  class_display: Record<LungClass, string>;
  class_risk_band: Record<LungClass, RiskBand>;
  unsupported_capabilities: Record<string, string>;
  deployed: { backbone: string; primary_model: string; validation_f1_macro: number };
  engine: { ready: boolean; backbone?: string; primary_model?: string; device?: string; n_qubits?: number; error?: string };
  hardware: { torch: string | false; cuda: boolean; device: string; gpu_name: string | null; vram_gb: number | null };
  quantum: { circuit: QuantumCircuitInfo; qubit_sweep: number[]; variational_layers: number };
  disclaimer: string;
  accepted_formats: string[];
  triage_rule?: string;
  security_posture?: {
    statement: string;
    implemented: string[];
    designed_not_enforced: string[];
    not_claimed: string[];
  };
}

/* -------------------------------------------------------- model comparison */

export interface ModelMetrics {
  n: number;
  accuracy: number;
  balanced_accuracy: number;
  precision_macro: number;
  recall_macro: number;
  f1_macro: number;
  roc_auc_macro_ovr: number;
  cohen_kappa: number;
  per_class: Record<string, {
    precision: number;
    recall_sensitivity: number;
    specificity: number;
    f1: number;
    support: number;
    roc_auc_ovr: number;
  }>;
  confusion_matrix: number[][];
  confusion_matrix_labels: string[];
  malignant_sensitivity?: number;
  malignant_specificity?: number;
  malignant_auc?: number;
  malignant_fn_count?: number;
}

export interface ModelEntry {
  val: ModelMetrics;
  test: ModelMetrics;
  val_ci: { metric: string; mean: number; lo95: number; hi95: number };
  test_ci: { metric: string; mean: number; lo95: number; hi95: number };
  val_calibration: { ece: number };
  test_calibration: { ece: number };
}

export interface BackboneComparison {
  models: Record<string, ModelEntry>;
  argmax_validation: string;
  statistically_tied: string[];
  primary_model: string;
  selection_policy: string;
  quantum_advantage_test: Record<string, {
    mean_diff: number;
    ci95_lo: number;
    ci95_hi: number;
    p_value: number;
    "significant_at_0.05": boolean;
    quantum_model?: string;
    hybrid_model?: string;
    classical_model: string;
  }>;
  curves_test: Record<string, {
    roc: Record<string, { fpr: number[]; tpr: number[]; auc: number }>;
    pr: Record<string, { precision: number[]; recall: number[]; ap: number }>;
  }>;
}

export interface ModelComparison {
  backbones: Record<string, BackboneComparison>;
  seed: number;
  n_bootstrap: number;
  deployed: { backbone: string; primary_model: string; validation_f1_macro: number };
}

/* ------------------------------------------------------- workflow entities */

export type ReviewStatus = "ai_generated" | "in_review" | "edited" | "approved" | "further_review" | "rejected";
export type Priority = "high" | "attention" | "routine";
export type UserRole = "administrator" | "radiologist" | "doctor" | "technician" | "researcher";

export interface Patient {
  id: string;
  external_id: string;
  age: number | null;
  sex: "M" | "F" | "O" | null;
  is_demo: boolean;
  created_at: string;
  scan_count: number;
  latest_risk: RiskBand | null;
  latest_scan_at: string | null;
}

export interface Scan {
  id: string;
  patient_id: string;
  filename: string;
  source_format: string;
  is_dicom: boolean;
  acquired_at: string | null;
  uploaded_at: string;
  is_demo: boolean;
}

export interface AnalysisRecord {
  id: string;
  scan_id: string;
  patient_id: string;
  predicted_class: LungClass;
  risk_band: RiskBand;
  confidence: number;
  branch_agreement: number;
  priority: Priority;
  review_status: ReviewStatus;
  n_lesions: number;
  runtime_seconds: number;
  created_at: string;
  model_version: string;
  is_demo: boolean;
}

export interface QueueItem extends AnalysisRecord {
  patient_external_id: string;
  scan_filename: string;
  patient_age: number | null;
  patient_sex: string | null;
}

export interface DoctorFeedback {
  id: string;
  analysis_id: string;
  ai_prediction: LungClass;
  doctor_assessment: LungClass | "uncertain";
  agreement: "agree" | "modify" | "reject";
  comments: string;
  reviewer_role: UserRole;
  created_at: string;
}

export interface FollowUp {
  id: string;
  patient_id: string;
  patient_external_id: string;
  analysis_id: string | null;
  risk_band: RiskBand;
  last_scan_at: string;
  /** Clinician-set. The platform never derives intervals autonomously. */
  review_due: string;
  status: "due" | "upcoming" | "completed" | "overdue";
  notes: string;
  is_demo: boolean;
}


/* ------------------------------------------------- API response shapes */

export interface PatientSummary {
  id: string; external_id: string; age: number | null; sex: string | null;
  is_demo: boolean; created_at: string; scan_count: number;
  latest_risk: RiskBand | null; latest_scan_at: string | null;
}

export interface TimelineEntry {
  analysis_id: string; scan_id: string; filename: string;
  predicted_class: LungClass; risk_band: RiskBand; confidence: number;
  priority: Priority; review_status: ReviewStatus; n_lesions: number;
  created_at: string; acquired_at: string | null; is_demo: boolean;
}

export interface PatientDetail {
  patient: {
    id: string; external_id: string; age: number | null; sex: string | null;
    notes: string; is_demo: boolean; created_at: string;
  };
  timeline: TimelineEntry[];
  follow_ups: {
    id: string; risk_band: string; review_due: string; status: string;
    notes: string; last_scan_at: string | null;
  }[];
  comparison_available: boolean;
}


/**
 * Structured preliminary report body, mirroring
 * api/services/analysis_service.py::build_report_body exactly.
 */
export interface ReportBody {
  patient_information: { identifier: string; age: number | null; sex: string | null; note: string };
  scan_information: {
    filename: string; format: string | null; dicom: boolean | null;
    original_size: [number, number] | null; analysis_resolution: [number, number] | null;
    windowing: string | null; hu_available: boolean | null; acquisition_notes: string[];
  };
  image_quality: {
    lung_field_detected: boolean | null; lung_field_area_fraction: number | null;
    sharpness_laplacian_var: number | null; contrast_rms: number | null;
    intensity_mean: number | null; intensity_std: number | null; note: string;
  };
  ai_findings: {
    classification: LungClass; display: string; confidence: number;
    probabilities: Record<LungClass, number>; selected_model: string;
    branch_agreement: number | null; decision_policy_applied: boolean | null;
  };
  suspected_region: {
    segmentation_regions: number; lesions: Lesion[];
    attention_location: string | null; attention_caveat: string | null;
  };
  risk_assessment: { band: RiskBand; confidence: number; caveat: string | null };
  explainability_summary: {
    method: string | null; label: string | null; caveat: string | null;
    segmentation_model: string | null;
    segmentation_detection_rate: number | null;
    segmentation_dice_when_detected: number | null;
  };
  model_information: {
    backbone: string | null; primary_model: string | null;
    test_metrics: Record<string, number>; quantum_circuit: Partial<QuantumCircuitInfo>;
  };
  limitations: string[];
  unsupported: Record<string, string | null>;
  clinical_review: {
    status: string; reviewer: string | null; reviewed_at: string | null; clinician_notes: string;
  };
  disclaimer: string;
}

export interface ReportRecord {
  id: string; analysis_id?: string; status: ReviewStatus;
  body: ReportBody;
  clinician_edits: Record<string, string | undefined>;
  reviewed_at?: string | null; updated_at?: string;
}

export interface AnalysisDetail {
  analysis: AnalysisRecord & { model_version: string };
  patient: { id: string; external_id: string; age: number | null; sex: string | null } | null;
  scan: { id: string; filename: string; source_format: string; is_dicom: boolean } | null;
  result: AnalysisResult;
  report: ReportRecord | null;
  feedback: DoctorFeedback[];
}

export interface ScanComparisonSide {
  analysis_id: string; created_at: string; predicted_class: LungClass;
  risk_band: RiskBand; confidence: number; n_lesions: number;
  total_lesion_area_px: number;
}

export interface ScanComparison {
  patient_id: string;
  earlier: ScanComparisonSide;
  later: ScanComparisonSide;
  observed_differences: {
    risk_band_changed: boolean; class_changed: boolean;
    confidence_delta: number; lesion_count_delta: number;
    lesion_area_delta_px: number; lung_field_fraction_delta: number;
  };
  interpretation_note: string;
  images: Record<string, string | undefined>;
}

export interface LidcExperiment {
  answer: string;
  acquired: {
    series_downloaded: number; gigabytes: number; slices_extracted: number;
    nodules_indeterminate_excluded: number;
  };
  training_set_change: { benign: { before: number; after: number; multiplier: number } };
  dataset_identity_probe: { source_prediction_roc_auc: number; nmi_source_label: number };
  results_on_unchanged_iqoth_test: Record<string, { delta: number; p_value?: number }>;
  mitigation_attempted: string;
  one_genuine_benefit: string;
  lesson: string;
  decision: string;
  question: string;
}

export interface LocalizationValidation {
  note: string;
  results: Record<string, {
    n_images: number; pointing_game_hit_rate: number;
    expected_chance_hit_rate: number; mean_attention_energy_in_box: number;
  }>;
}


/** External validation on a dataset the deployed model never saw (D26). */
export interface ExternalValidation {
  external_dataset: string;
  why_fair: string;
  n_slices: number;
  n_patients: number;
  label_counts: { benign: number; malignant: number };
  prediction_distribution: Record<string, number>;
  lesion_slices_called_normal: number;
  slice_level: {
    roc_auc: number; ci95: [number, number]; average_precision: number;
    sensitivity_argmax: number; specificity_argmax: number;
  };
  patient_level: { roc_auc: number; ci95: [number, number] } | null;
  in_domain_reference: { n: number; roc_auc: number; ci95: [number, number] };
  auc_change_external_minus_internal: number;
  caveats: string[];
}
