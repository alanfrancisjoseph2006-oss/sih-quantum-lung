"use client";

/**
 * Patient timeline + scan comparison.
 *
 * Comparison wording is deliberately "observed imaging and model differences" —
 * this platform is not validated to measure disease progression, and saying so
 * is more useful to a clinician than a confident number would be.
 */

import * as React from "react";
import Link from "next/link";
import { use } from "react";
import {
  ArrowLeft, ArrowRight, CalendarPlus, GitCompare, Stethoscope,
} from "lucide-react";
import { PageContainer } from "@/components/PageContainer";
import { DisclaimerBar } from "@/components/DisclaimerBar";
import { Badge, Button, Card, CardBody, CardHeader, CardTitle, EmptyState, ErrorState, Input, KeyValue, Modal, Notice, RiskBadge, SectionHead, Skeleton, Table, Td, Th, Tr, Textarea } from "@/components/ui";
import { compareScans, createFollowUp, getPatient } from "@/lib/api";
import type { PatientDetail, RiskBand, ScanComparison } from "@/lib/types";
import { classLabel, cn, formatDate, num, pct, reviewStatusLabel } from "@/lib/utils";

export default function PatientPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const [data, setData] = React.useState<PatientDetail | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);
  const [cmp, setCmp] = React.useState<ScanComparison | null>(null);
  const [cmpBusy, setCmpBusy] = React.useState(false);
  const [fuOpen, setFuOpen] = React.useState(false);
  const [fuDate, setFuDate] = React.useState("");
  const [fuNotes, setFuNotes] = React.useState("");

  const load = React.useCallback(async () => {
    setLoading(true); setError(null);
    try { setData(await getPatient(id)); }
    catch (e) { setError(e instanceof Error ? e.message : "Failed to load patient"); }
    finally { setLoading(false); }
  }, [id]);
  React.useEffect(() => {
    // Deferred a tick so the state updates inside `load` do not run
    // synchronously inside the effect body (react-hooks/set-state-in-effect).
    const t = setTimeout(() => { void load(); }, 0);
    return () => clearTimeout(t);
  }, [load]);

  async function runCompare() {
    const t = data?.timeline ?? [];
    if (t.length < 2) return;
    setCmpBusy(true);
    try { setCmp(await compareScans(t[t.length - 2].analysis_id, t[t.length - 1].analysis_id)); }
    catch (e) { setError(e instanceof Error ? e.message : "Comparison failed"); }
    finally { setCmpBusy(false); }
  }

  async function saveFollowUp() {
    if (!fuDate) return;
    await createFollowUp({ patient_id: id, review_due: new Date(fuDate).toISOString(), notes: fuNotes });
    setFuOpen(false); setFuDate(""); setFuNotes(""); load();
  }

  if (error) return <PageContainer><ErrorState message={error} onRetry={load} /></PageContainer>;
  if (loading || !data) {
    return <PageContainer><Card className="mt-5"><CardBody><Skeleton className="h-64 w-full" /></CardBody></Card></PageContainer>;
  }

  const p = data.patient;
  const timeline = data.timeline ?? [];

  return (
    <PageContainer>
      <Link href="/dashboard/patients"
            className="mb-3 inline-flex items-center gap-1.5 text-[13px] text-[var(--color-ink-tertiary)] hover:text-[var(--color-ink)]">
        <ArrowLeft className="size-3.5" aria-hidden /> All patients
      </Link>

      <SectionHead
        title={p.external_id}
        description={`${p.age ?? "—"} years · ${p.sex ?? "—"} · ${timeline.length} scan${timeline.length === 1 ? "" : "s"} on record`}
        actions={
          <div className="flex gap-2">
            <Button icon={CalendarPlus} onClick={() => setFuOpen(true)}>Set follow-up</Button>
            {data.comparison_available && (
              <Button variant="primary" icon={GitCompare} loading={cmpBusy} onClick={runCompare}>
                Compare last two
              </Button>
            )}
          </div>
        }
      />

      <div className="mt-5 space-y-5">
        <DisclaimerBar compact />
        {p.is_demo && (
          <Notice tone="neutral" title="Demonstration record">
            This is a synthetic patient. The identifier is fictional, but every analysis below was
            produced by the real model on a real CT image.
          </Notice>
        )}

        {/* Timeline */}
        <Card>
          <CardHeader>
            <CardTitle>Scan timeline</CardTitle>
            <p className="mt-0.5 text-[12.5px] text-[var(--color-ink-tertiary)]">Oldest first</p>
          </CardHeader>
          {timeline.length === 0 ? (
            <EmptyState icon={Stethoscope} title="No scans recorded" />
          ) : (
            <CardBody>
              <ol className="relative space-y-4 border-l border-[var(--color-line)] pl-6">
                {timeline.map((t, i) => (
                  <li key={t.analysis_id} className="relative">
                    <span className={cn(
                      "absolute -left-[29px] top-1 flex size-[18px] items-center justify-center rounded-full border-2 border-[var(--color-surface)]",
                      t.risk_band === "High" ? "bg-[var(--color-high)]"
                        : t.risk_band === "Intermediate" ? "bg-[var(--color-moderate)]"
                        : "bg-[var(--color-low)]")} aria-hidden />
                    <div className="rounded-[var(--radius-md)] border border-[var(--color-line)] p-3.5">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <RiskBadge risk={t.risk_band as RiskBand} />
                          <span className="text-[13px] font-medium">{classLabel[t.predicted_class as keyof typeof classLabel]}</span>
                          {i === timeline.length - 1 && <Badge tone="primary">Most recent</Badge>}
                        </div>
                        <Link href={`/dashboard/reports?analysis=${t.analysis_id}`}>
                          <Button variant="ghost" size="sm" iconRight={ArrowRight}>Report</Button>
                        </Link>
                      </div>
                      <div className="mt-2 grid grid-cols-2 gap-x-6 gap-y-1 text-[12.5px] sm:grid-cols-4">
                        <span className="text-[var(--color-ink-tertiary)]">Analysed{" "}
                          <span className="text-[var(--color-ink)]">{formatDate(t.created_at)}</span></span>
                        <span className="text-[var(--color-ink-tertiary)]">Confidence{" "}
                          <span className="tabular text-[var(--color-ink)]">{pct(t.confidence)}</span></span>
                        <span className="text-[var(--color-ink-tertiary)]">Regions{" "}
                          <span className="tabular text-[var(--color-ink)]">{t.n_lesions}</span></span>
                        <span className="text-[var(--color-ink-tertiary)]">Review{" "}
                          <span className="text-[var(--color-ink)]">{reviewStatusLabel[t.review_status] ?? t.review_status}</span></span>
                      </div>
                    </div>
                  </li>
                ))}
              </ol>
            </CardBody>
          )}
        </Card>

        {/* Comparison */}
        {cmp && (
          <Card>
            <CardHeader><CardTitle>Observed imaging and model differences</CardTitle></CardHeader>
            <CardBody className="space-y-4">
              <Notice tone="warning" title="Not a measurement of disease progression">
                {cmp.interpretation_note}
              </Notice>

              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                {(["earlier", "later"] as const).map((k) => (
                  <div key={k} className="rounded-[var(--radius-md)] border border-[var(--color-line)] p-3">
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-[12px] font-semibold uppercase tracking-[0.05em] text-[var(--color-ink-tertiary)]">
                        {k === "earlier" ? "Earlier scan" : "Later scan"}
                      </span>
                      <RiskBadge risk={cmp[k].risk_band as RiskBand} />
                    </div>
                    {cmp.images?.[`${k}_overlay`] && (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={cmp.images[`${k}_overlay`]} alt={`${k} scan attention overlay`}
                           className="mb-2 aspect-square w-full rounded border border-[var(--color-line)] bg-black object-contain" />
                    )}
                    <KeyValue label="Analysed" value={formatDate(cmp[k].created_at)} mono={false} />
                    <KeyValue label="Finding" value={cmp[k].predicted_class} mono={false} />
                    <KeyValue label="Confidence" value={pct(cmp[k].confidence)} />
                    <KeyValue label="Regions" value={cmp[k].n_lesions} />
                  </div>
                ))}
              </div>

              <div className="rounded-[var(--radius-md)] bg-[var(--color-surface-sunken)] p-3">
                <div className="mb-2 text-[12px] font-semibold uppercase tracking-[0.05em] text-[var(--color-ink-tertiary)]">
                  Differences
                </div>
                <div className="grid grid-cols-2 gap-x-6 sm:grid-cols-3">
                  <KeyValue label="Risk band changed" value={cmp.observed_differences.risk_band_changed ? "yes" : "no"} mono={false} />
                  <KeyValue label="Class changed" value={cmp.observed_differences.class_changed ? "yes" : "no"} mono={false} />
                  <KeyValue label="Confidence Δ" value={num(cmp.observed_differences.confidence_delta)} />
                  <KeyValue label="Region count Δ" value={cmp.observed_differences.lesion_count_delta} />
                  <KeyValue label="Region area Δ" value={`${cmp.observed_differences.lesion_area_delta_px} px`} />
                  <KeyValue label="Lung field Δ" value={num(cmp.observed_differences.lung_field_fraction_delta)} />
                </div>
              </div>
            </CardBody>
          </Card>
        )}

        {/* Follow-ups */}
        <Card>
          <CardHeader><CardTitle>Follow-ups</CardTitle></CardHeader>
          {(data.follow_ups ?? []).length === 0 ? (
            <EmptyState icon={CalendarPlus} title="No follow-up scheduled"
              description="Review dates are set by clinicians. The platform does not derive intervals autonomously."
              action={<Button icon={CalendarPlus} onClick={() => setFuOpen(true)}>Set follow-up</Button>} />
          ) : (
            <Table>
              <thead><tr><Th>Review due</Th><Th>Status</Th><Th>Risk at last scan</Th><Th>Notes</Th></tr></thead>
              <tbody>
                {data.follow_ups.map((f) => (
                  <Tr key={f.id}>
                    <Td className="tabular">{formatDate(f.review_due)}</Td>
                    <Td><Badge tone={f.status === "overdue" ? "high" : f.status === "due" ? "moderate" : "neutral"}>
                      {f.status}</Badge></Td>
                    <Td>{f.risk_band ? <RiskBadge risk={f.risk_band as RiskBand} /> : "—"}</Td>
                    <Td className="text-[var(--color-ink-tertiary)]">{f.notes || "—"}</Td>
                  </Tr>
                ))}
              </tbody>
            </Table>
          )}
        </Card>
      </div>

      <Modal open={fuOpen} onClose={() => setFuOpen(false)} title="Set follow-up review date"
        description="The clinician sets this date. The platform does not recommend intervals."
        footer={<>
          <Button onClick={() => setFuOpen(false)}>Cancel</Button>
          <Button variant="primary" disabled={!fuDate} onClick={saveFollowUp}>Save follow-up</Button>
        </>}>
        <div className="space-y-3">
          <div>
            <label htmlFor="fudate" className="mb-1 block text-[12.5px] font-medium">Review due date</label>
            <Input id="fudate" type="date" value={fuDate} onChange={(e) => setFuDate(e.target.value)} />
          </div>
          <div>
            <label htmlFor="funotes" className="mb-1 block text-[12.5px] font-medium">Notes</label>
            <Textarea id="funotes" rows={3} value={fuNotes} onChange={(e) => setFuNotes(e.target.value)}
                      placeholder="Clinical rationale for this interval…" />
          </div>
          <Notice tone="warning">
            This platform does not suggest follow-up intervals. Timing must follow clinical
            judgement and local guidance.
          </Notice>
        </div>
      </Modal>
    </PageContainer>
  );
}
