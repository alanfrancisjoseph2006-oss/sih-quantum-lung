"use client";

/**
 * Preliminary report + clinician review workflow.
 *
 *   AI generated -> Doctor review -> Doctor edit -> Approve -> Final report
 *
 * The AI-generated body is never overwritten; clinician changes accumulate
 * separately so the original model output stays auditable.
 */

import * as React from "react";

import {
  CheckCircle2, Eye, FileText, MessageSquare, Pencil,
  Printer, RotateCcw, Send, XCircle,
} from "lucide-react";
import { PageContainer } from "@/components/PageContainer";
import { DisclaimerBar } from "@/components/DisclaimerBar";
import {
  Badge, Button, Card, CardBody, CardHeader, CardTitle, DemoBadge, EmptyState,
  ErrorState, KeyValue, Modal, Notice, RiskBadge, SectionHead, Select, Skeleton,
  Table, Td, Th, Tr, Textarea,
} from "@/components/ui";
import { getAnalysis, getQueue, submitFeedback, updateReport } from "@/lib/api";
import type { AnalysisDetail, QueueItem, ReportBody, RiskBand } from "@/lib/types";
import {
  classLabel, cn, formatDateTime, num, pct, relativeTime, reviewStatusLabel,
} from "@/lib/utils";

const WORKFLOW: { key: string; label: string }[] = [
  { key: "ai_generated", label: "AI generated" },
  { key: "in_review", label: "Doctor review" },
  { key: "edited", label: "Doctor edit" },
  { key: "approved", label: "Approved" },
];

export default function ReportsPage() {
  const [analysisId, setAnalysisId] = React.useState<string | null>(null);
  const [list, setList] = React.useState<QueueItem[]>([]);
  const [data, setData] = React.useState<AnalysisDetail | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [notes, setNotes] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const [fbOpen, setFbOpen] = React.useState(false);
  const [fbAssessment, setFbAssessment] = React.useState<string>("malignant");
  const [fbAgreement, setFbAgreement] = React.useState<string>("agree");
  const [fbComments, setFbComments] = React.useState("");

  // Deep-link support (?analysis=...) plus the initial report list.
  React.useEffect(() => {
    let cancelled = false;
    const t = setTimeout(() => {
      const q = new URLSearchParams(window.location.search).get("analysis");
      if (q) setAnalysisId(q);
      getQueue({ limit: 100 })
        .then((r) => { if (!cancelled) setList(r.items); })
        .catch(() => {})
        .finally(() => { if (!cancelled) setLoading(false); });
    }, 0);
    return () => { cancelled = true; clearTimeout(t); };
  }, []);

  const loadAnalysis = React.useCallback(async (id: string) => {
    setLoading(true); setError(null);
    try {
      const d = await getAnalysis(id);
      setData(d);
      setNotes(d.report?.clinician_edits?.clinician_notes ?? "");
      setFbAssessment(d.analysis.predicted_class);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load report");
    } finally { setLoading(false); }
  }, []);

  React.useEffect(() => {
    if (!analysisId) return;
    const t = setTimeout(() => { void loadAnalysis(analysisId); }, 0);
    return () => clearTimeout(t);
  }, [analysisId, loadAnalysis]);

  async function setStatus(status: string) {
    if (!analysisId) return;
    setBusy(true);
    try {
      await updateReport(analysisId, { status, clinician_notes: notes });
      await loadAnalysis(analysisId);
    } catch (e) { setError(e instanceof Error ? e.message : "Update failed"); }
    finally { setBusy(false); }
  }

  async function sendFeedback() {
    if (!analysisId) return;
    setBusy(true);
    try {
      await submitFeedback(analysisId, {
        doctor_assessment: fbAssessment, agreement: fbAgreement, comments: fbComments });
      setFbOpen(false); setFbComments("");
      await loadAnalysis(analysisId);
    } catch (e) { setError(e instanceof Error ? e.message : "Feedback failed"); }
    finally { setBusy(false); }
  }

  /* ---------------------------------------------------------- list view */
  if (!analysisId) {
    return (
      <PageContainer>
        <SectionHead title="Reports"
          description="Preliminary AI-assisted reports awaiting clinician review." />
        <div className="mt-5 space-y-5">
          <DisclaimerBar compact />
          <Card>
            {loading ? <Skeleton className="m-4 h-48" />
             : list.length === 0 ? <EmptyState icon={FileText} title="No reports yet"
                 description="Analyse a scan to generate a preliminary report." />
             : (
              <Table>
                <thead><tr><Th>Patient</Th><Th>Finding</Th><Th>Risk</Th><Th>Status</Th><Th>Analysed</Th><Th /></tr></thead>
                <tbody>
                  {list.map((q) => (
                    <Tr key={q.id}>
                      <Td><span className="font-medium">{q.patient_external_id}</span>
                        {q.is_demo && <DemoBadge className="ml-2" />}</Td>
                      <Td className="text-[var(--color-ink-secondary)]">{classLabel[q.predicted_class]}</Td>
                      <Td><RiskBadge risk={q.risk_band as RiskBand} /></Td>
                      <Td><Badge tone={q.review_status === "approved" ? "low"
                        : q.review_status === "rejected" ? "high" : "neutral"}>
                        {reviewStatusLabel[q.review_status] ?? q.review_status}</Badge></Td>
                      <Td className="text-[var(--color-ink-tertiary)]">{relativeTime(q.created_at)}</Td>
                      <Td className="text-right">
                        <Button size="sm" icon={Eye} onClick={() => setAnalysisId(q.id)}>Open</Button>
                      </Td>
                    </Tr>
                  ))}
                </tbody>
              </Table>
             )}
          </Card>
        </div>
      </PageContainer>
    );
  }

  /* -------------------------------------------------------- detail view */
  if (error) return <PageContainer><ErrorState message={error} onRetry={() => loadAnalysis(analysisId)} /></PageContainer>;
  if (loading || !data) {
    return <PageContainer><Card className="mt-5"><CardBody><Skeleton className="h-96 w-full" /></CardBody></Card></PageContainer>;
  }

  const body = data.report?.body as ReportBody | undefined;
  const status = data.report?.status ?? "ai_generated";
  const r = data.result;
  const stepIndex = Math.max(0, WORKFLOW.findIndex((w) => w.key === status));
  const rejected = status === "rejected";

  return (
    <PageContainer>
      <button onClick={() => { setAnalysisId(null); setData(null); }}
              className="mb-3 text-[13px] text-[var(--color-ink-tertiary)] hover:text-[var(--color-ink)]">
        ← All reports
      </button>

      <SectionHead
        eyebrow="Preliminary AI-assisted report"
        title={data.patient?.external_id ?? "Report"}
        description={`Generated ${formatDateTime(data.analysis.created_at)} · model ${data.analysis.model_version}`}
        actions={
          <div className="no-print flex gap-2">
            <Button icon={Printer} onClick={() => window.print()}>Print</Button>
            <Button icon={MessageSquare} onClick={() => setFbOpen(true)}>Add assessment</Button>
          </div>
        }
      />

      <div className="mt-5 space-y-5">
        <DisclaimerBar />

        {/* Workflow tracker */}
        <Card>
          <CardBody>
            {rejected ? (
              <Notice tone="danger" title="AI finding rejected by reviewer">
                This report was marked as rejected. The original model output is retained for audit.
              </Notice>
            ) : (
              <ol className="flex flex-wrap items-center gap-2">
                {WORKFLOW.map((w, i) => (
                  <li key={w.key} className="flex items-center gap-2">
                    <span className={cn(
                      "flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[12.5px] font-medium",
                      i < stepIndex ? "border-[var(--color-low-border)] bg-[var(--color-low-soft)] text-[var(--color-low)]"
                        : i === stepIndex ? "border-[var(--color-primary-border)] bg-[var(--color-primary-soft)] text-[var(--color-primary)]"
                        : "border-[var(--color-line)] text-[var(--color-ink-disabled)]")}>
                      {i < stepIndex && <CheckCircle2 className="size-3.5" aria-hidden />}
                      {w.label}
                    </span>
                    {i < WORKFLOW.length - 1 && <span className="text-[var(--color-ink-disabled)]" aria-hidden>→</span>}
                  </li>
                ))}
              </ol>
            )}

            <div className="no-print mt-4 flex flex-wrap gap-2">
              <Button icon={Eye} loading={busy} disabled={status === "in_review"}
                      onClick={() => setStatus("in_review")}>Start review</Button>
              <Button icon={Pencil} loading={busy} onClick={() => setStatus("edited")}>Save edits</Button>
              <Button variant="primary" icon={CheckCircle2} loading={busy}
                      onClick={() => setStatus("approved")}>Approve report</Button>
              <Button icon={RotateCcw} loading={busy}
                      onClick={() => setStatus("further_review")}>Request further review</Button>
              <Button variant="danger" icon={XCircle} loading={busy}
                      onClick={() => setStatus("rejected")}>Reject AI finding</Button>
            </div>
          </CardBody>
        </Card>

        <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
          <div className="space-y-5 lg:col-span-2">
            <Section title="Patient information">
              <KeyValue label="Identifier" value={body?.patient_information?.identifier} />
              <KeyValue label="Age" value={body?.patient_information?.age ?? "—"} />
              <KeyValue label="Sex" value={body?.patient_information?.sex ?? "—"} />
              <p className="pt-2 text-[12px] text-[var(--color-ink-tertiary)]">
                {body?.patient_information?.note}
              </p>
            </Section>

            <Section title="Scan information">
              <KeyValue label="File" value={<span className="font-normal">{body?.scan_information?.filename}</span>} mono={false} />
              <KeyValue label="Format" value={body?.scan_information?.format} />
              <KeyValue label="DICOM" value={body?.scan_information?.dicom ? "yes" : "no"} />
              <KeyValue label="Original size" value={(body?.scan_information?.original_size ?? []).join(" × ")} />
              <KeyValue label="Windowing" value={<span className="font-normal">{body?.scan_information?.windowing}</span>} mono={false} />
            </Section>

            <Section title="AI findings">
              <div className="mb-3 flex flex-wrap items-center gap-3">
                <RiskBadge risk={data.analysis.risk_band as RiskBand} />
                <span className="text-[14px] font-medium">{body?.ai_findings?.display}</span>
              </div>
              <KeyValue label="Classification" value={body?.ai_findings?.classification} />
              <KeyValue label="Confidence" value={pct(body?.ai_findings?.confidence)} />
              <KeyValue label="Selected model" value={body?.ai_findings?.selected_model} />
              <KeyValue label="Branch agreement" value={pct(body?.ai_findings?.branch_agreement, 0)} />
            </Section>

            <Section title="Suspected region">
              <KeyValue label="Segmented regions" value={body?.suspected_region?.segmentation_regions ?? 0} />
              <KeyValue label="Attention location"
                        value={<span className="font-normal">{body?.suspected_region?.attention_location ?? "—"}</span>} mono={false} />
              {body?.suspected_region?.attention_caveat && (
                <Notice tone="warning" className="mt-3">{body?.suspected_region.attention_caveat}</Notice>
              )}
            </Section>

            <Section title="Image quality">
              <KeyValue label="Lung field detected" value={body?.image_quality?.lung_field_detected ? "yes" : "no"} />
              <KeyValue label="Lung field area" value={pct(body?.image_quality?.lung_field_area_fraction)} />
              <KeyValue label="Sharpness" value={num(body?.image_quality?.sharpness_laplacian_var, 0)} />
              <p className="pt-2 text-[12px] text-[var(--color-ink-tertiary)]">{body?.image_quality?.note}</p>
            </Section>

            <Section title="Explainability summary">
              <KeyValue label="Method" value={body?.explainability_summary?.method} />
              <KeyValue label="Segmentation detection rate"
                        value={body?.explainability_summary?.segmentation_detection_rate != null
                          ? pct(body?.explainability_summary.segmentation_detection_rate) : "—"} />
              <KeyValue label="Dice when detected"
                        value={num(body?.explainability_summary?.segmentation_dice_when_detected)} />
              <Notice tone="warning" className="mt-3">{body?.explainability_summary?.caveat}</Notice>
            </Section>

            <Section title="Clinical review">
              <label htmlFor="notes" className="mb-1.5 block text-[12.5px] font-medium text-[var(--color-ink-secondary)]">
                Clinician notes
              </label>
              <Textarea id="notes" rows={5} value={notes} onChange={(e) => setNotes(e.target.value)}
                        placeholder="Radiologist observations, correlation with prior imaging, recommended next steps…" />
              <p className="mt-2 text-[12px] text-[var(--color-ink-tertiary)]">
                Notes are stored separately from the AI-generated body, which is never overwritten.
              </p>
            </Section>
          </div>

          {/* Sidebar */}
          <div className="space-y-5">
            {r?.explainability?.images?.overlay && (
              <Card>
                <CardHeader><CardTitle>Attention overlay</CardTitle></CardHeader>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={r.explainability.images.overlay} alt="Attention overlay"
                     className="aspect-square w-full bg-black object-contain" />
              </Card>
            )}

            <Section title="Limitations">
              <ul className="space-y-1.5 text-[12.5px] leading-relaxed text-[var(--color-ink-secondary)]">
                {(body?.limitations ?? []).slice(0, 6).map((l, i) => (
                  <li key={i} className="flex gap-1.5"><span aria-hidden>·</span><span>{l}</span></li>
                ))}
              </ul>
            </Section>

            <Section title="Not provided">
              {Object.entries(body?.unsupported ?? {}).map(([k, v]) => v ? (
                <div key={k} className="mb-2 rounded-[var(--radius-md)] bg-[var(--color-surface-sunken)] p-2.5">
                  <div className="text-[12px] font-medium capitalize">{k.replace(/_/g, " ")}</div>
                  <p className="mt-0.5 text-[11.5px] text-[var(--color-ink-tertiary)]">{String(v)}</p>
                </div>
              ) : null)}
            </Section>

            {data.feedback?.length > 0 && (
              <Section title="Clinician assessments">
                {data.feedback.map((f) => (
                  <div key={f.id} className="mb-2.5 rounded-[var(--radius-md)] border border-[var(--color-line)] p-2.5">
                    <div className="mb-1 flex items-center justify-between">
                      <Badge tone={f.agreement === "agree" ? "low" : f.agreement === "reject" ? "high" : "moderate"}>
                        {f.agreement}
                      </Badge>
                      <span className="text-[11.5px] text-[var(--color-ink-tertiary)]">{relativeTime(f.created_at)}</span>
                    </div>
                    <div className="text-[12.5px]">
                      AI said <strong>{f.ai_prediction}</strong> · clinician said <strong>{f.doctor_assessment}</strong>
                    </div>
                    {f.comments && <p className="mt-1 text-[12px] text-[var(--color-ink-tertiary)]">{f.comments}</p>}
                  </div>
                ))}
                <Notice tone="neutral">
                  Stored for future validated training pipelines. This platform does not retrain
                  production models from clinician feedback.
                </Notice>
              </Section>
            )}
          </div>
        </div>
      </div>

      {/* Feedback modal */}
      <Modal open={fbOpen} onClose={() => setFbOpen(false)} title="Clinician assessment"
        description="Record whether you agree with the AI finding."
        footer={<>
          <Button onClick={() => setFbOpen(false)}>Cancel</Button>
          <Button variant="primary" icon={Send} loading={busy} onClick={sendFeedback}>Submit assessment</Button>
        </>}>
        <div className="space-y-3.5">
          <div className="rounded-[var(--radius-md)] bg-[var(--color-surface-sunken)] p-3 text-[13px]">
            AI prediction: <strong>{data.analysis.predicted_class}</strong> at {pct(data.analysis.confidence)} confidence
          </div>
          <div>
            <label htmlFor="agree" className="mb-1 block text-[12.5px] font-medium">Agreement</label>
            <Select id="agree" value={fbAgreement} onChange={(e) => setFbAgreement(e.target.value)}>
              <option value="agree">Agree with AI finding</option>
              <option value="modify">Modify — different assessment</option>
              <option value="reject">Reject AI finding</option>
            </Select>
          </div>
          <div>
            <label htmlFor="assess" className="mb-1 block text-[12.5px] font-medium">Your assessment</label>
            <Select id="assess" value={fbAssessment} onChange={(e) => setFbAssessment(e.target.value)}>
              <option value="normal">No suspicious finding</option>
              <option value="benign">Benign-appearing finding</option>
              <option value="malignant">Suspicious finding</option>
              <option value="uncertain">Uncertain — needs further imaging</option>
            </Select>
          </div>
          <div>
            <label htmlFor="fbc" className="mb-1 block text-[12.5px] font-medium">Comments</label>
            <Textarea id="fbc" rows={3} value={fbComments} onChange={(e) => setFbComments(e.target.value)} />
          </div>
          <Notice tone="neutral">
            Feedback is stored for a future validated training pipeline. It does not retrain any
            production model, because an unvalidated feedback loop on a clinical model is unsafe.
          </Notice>
        </div>
      </Modal>
    </PageContainer>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Card className="print-break">
      <CardHeader><CardTitle>{title}</CardTitle></CardHeader>
      <CardBody>{children}</CardBody>
    </Card>
  );
}
