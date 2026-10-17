/**
 * Typed API client.
 *
 * All network access goes through here so error handling, base-URL resolution
 * and response typing are consistent across the app.
 */

import type {
  AnalysisDetail, AnalysisResult, ExternalValidation, FollowUp, LidcExperiment, LocalizationValidation, ModelComparison, PatientDetail, PatientSummary, PlatformMeta, QueueItem, ReportRecord, ScanComparison,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init?.headers ?? {}) },
    });
  } catch {
    throw new ApiError(0, "Cannot reach the analysis server. Is the backend running on port 8000?");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch { /* non-JSON error body */ }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

/* ------------------------------------------------------------- platform */

export const getMeta = () => request<PlatformMeta>("/api/meta");
export const getHealth = () => request<{ status: string; engine: Record<string, unknown> }>("/api/health");
export const getModelComparison = () => request<ModelComparison>("/api/model-comparison");
export const getEducation = () =>
  request<{ cancer_education: Record<string, unknown> }>("/api/education");
export const getLidcExperiment = () => request<LidcExperiment>("/api/lidc-experiment");
export const getExternalValidation = () =>
  request<ExternalValidation>("/api/external-validation");
export const getLocalizationValidation = () => request<LocalizationValidation>("/api/localization-validation");

/* -------------------------------------------------------------- clinical */

export interface OverviewResponse {
  stats: {
    total_analyses: number; today_analyses: number; high_priority: number;
    pending_review: number; completed: number; patients: number;
    follow_ups_due: number; by_risk: Record<string, number>;
    demo_analyses: number; contains_demo_data: boolean;
  };
  recent_activity: {
    id: string; action: string; entity_type: string; entity_id: string;
    actor_role: string; detail: Record<string, unknown>; created_at: string;
  }[];
  triage_rule: string;
}

export const getOverview = () => request<OverviewResponse>("/api/overview");

export const getQueue = (params: {
  priority?: string; review_status?: string; search?: string; limit?: number;
} = {}) => {
  const q = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => v != null && v !== "" && q.set(k, String(v)));
  const qs = q.toString();
  return request<{ items: QueueItem[]; total: number; rule: string }>(
    `/api/priority-queue${qs ? `?${qs}` : ""}`);
};

export const getPatients = (search?: string) =>
  request<{ items: PatientSummary[]; total: number }>(
    `/api/patients${search ? `?search=${encodeURIComponent(search)}` : ""}`);

export const getPatient = (id: string) => request<PatientDetail>(`/api/patients/${id}`);

export const getAnalysis = (id: string) => request<AnalysisDetail>(`/api/analysis/${id}`);

export const getReport = (analysisId: string) => request<ReportRecord>(`/api/reports/${analysisId}`);

export const updateReport = (analysisId: string, body: {
  status?: string; clinician_notes?: string; edits?: Record<string, unknown>;
}) => request<ReportRecord>(`/api/reports/${analysisId}`, {
  method: "PATCH",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const submitFeedback = (analysisId: string, body: {
  doctor_assessment: string; agreement: string; comments: string;
}) => request<{ id: string; stored: boolean; note: string }>(
  `/api/doctor-feedback?analysis_id=${encodeURIComponent(analysisId)}`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const getFollowUps = (status?: string) =>
  request<{ items: FollowUp[]; note: string }>(`/api/follow-ups${status ? `?status=${status}` : ""}`);

export const createFollowUp = (body: {
  patient_id: string; analysis_id?: string; review_due: string; notes: string;
}) => request<{ id: string; status: string }>("/api/follow-ups", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const compareScans = (a: string, b: string) =>
  request<ScanComparison>(`/api/scan-comparison?analysis_a=${a}&analysis_b=${b}`);

/* -------------------------------------------------------------- analysis */

export async function analyzeScan(
  file: File,
  opts: { patientExternalId?: string; age?: number; sex?: string; persist?: boolean } = {},
): Promise<AnalysisResult & { analysis_id?: string; patient_id?: string; persisted?: boolean }> {
  const fd = new FormData();
  fd.append("file", file);
  if (opts.patientExternalId) fd.append("patient_external_id", opts.patientExternalId);
  if (opts.age != null) fd.append("patient_age", String(opts.age));
  if (opts.sex) fd.append("patient_sex", opts.sex);
  fd.append("persist", String(opts.persist ?? true));

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/api/analyze`, { method: "POST", body: fd });
  } catch {
    throw new ApiError(0, "Cannot reach the analysis server. Is the backend running on port 8000?");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(res.status, body?.detail ?? res.statusText);
  return body;
}
