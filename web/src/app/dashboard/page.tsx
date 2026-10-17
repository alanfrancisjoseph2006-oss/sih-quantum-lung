"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Activity, AlertTriangle, ArrowRight, CheckCircle2, ClipboardList, Scan, Stethoscope,
} from "lucide-react";
import { PageContainer } from "@/components/PageContainer";
import { DisclaimerBar } from "@/components/DisclaimerBar";
import { Button, Card, CardBody, CardHeader, CardTitle, DemoBadge, EmptyState, ErrorState, Metric, Notice, RiskBadge, SectionHead, Skeleton, Table, Td, Th, Tr } from "@/components/ui";
import { getOverview, getQueue, type OverviewResponse } from "@/lib/api";
import type { QueueItem, RiskBand } from "@/lib/types";
import { classLabel, pct, relativeTime } from "@/lib/utils";

export default function OverviewPage() {
  const router = useRouter();
  const [data, setData] = React.useState<OverviewResponse | null>(null);
  const [queue, setQueue] = React.useState<QueueItem[]>([]);
  const [error, setError] = React.useState<string | null>(null);
  const [loading, setLoading] = React.useState(true);

  const load = React.useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const [o, q] = await Promise.all([getOverview(), getQueue({ limit: 6 })]);
      setData(o); setQueue(q.items);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load dashboard");
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    // Deferred a tick so the state updates inside `load` do not run
    // synchronously inside the effect body (react-hooks/set-state-in-effect).
    const t = setTimeout(() => { void load(); }, 0);
    return () => clearTimeout(t);
  }, [load]);

  if (error) {
    return <PageContainer><ErrorState message={error} onRetry={load} /></PageContainer>;
  }

  const s = data?.stats;

  return (
    <PageContainer>
      <SectionHead
        title="Overview"
        description="Current screening workload, triage status and recent activity."
        actions={
          <Button variant="primary" icon={Scan} onClick={() => router.push("/dashboard/analyze")}>
            Analyze scan
          </Button>
        }
      />

      <div className="mt-5 space-y-5">
        <DisclaimerBar />

        {s?.contains_demo_data && (
          <Notice tone="neutral" title="This workspace contains demonstration data">
            {s.demo_analyses} of {s.total_analyses} analyses are synthetic records created for
            demonstration. Patient identifiers are fictional, but <strong>every risk score shown was
            produced by the real model</strong> running on real CT images — no metrics are fabricated.
            Demo rows are flagged in the database and labelled throughout the interface.
          </Notice>
        )}

        {/* Metrics */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {loading ? (
            Array.from({ length: 4 }).map((_, i) => (
              <Card key={i}><div className="p-5"><Skeleton className="h-4 w-24" />
                <Skeleton className="mt-3 h-8 w-16" /></div></Card>
            ))
          ) : (
            <>
              <Card><Metric label="Scans today" value={s?.today_analyses ?? 0} icon={Scan}
                sublabel={`${s?.total_analyses ?? 0} total analyses`} /></Card>
              <Card><Metric label="High priority" value={s?.high_priority ?? 0} icon={AlertTriangle}
                tone={(s?.high_priority ?? 0) > 0 ? "high" : "neutral"}
                sublabel="Suspicious findings awaiting review" /></Card>
              <Card><Metric label="Pending review" value={s?.pending_review ?? 0} icon={ClipboardList}
                tone={(s?.pending_review ?? 0) > 0 ? "moderate" : "neutral"}
                sublabel="Reports not yet approved" /></Card>
              <Card><Metric label="Follow-ups due" value={s?.follow_ups_due ?? 0} icon={Stethoscope}
                sublabel="Clinician-set review dates" /></Card>
            </>
          )}
        </div>

        <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
          {/* Priority queue preview */}
          <Card className="lg:col-span-2">
            <CardHeader className="flex items-center justify-between">
              <div>
                <CardTitle>Priority queue</CardTitle>
                <p className="mt-0.5 text-[12.5px] text-[var(--color-ink-tertiary)]">
                  Highest-priority cases first
                </p>
              </div>
              <Link href="/dashboard/queue">
                <Button variant="ghost" size="sm" iconRight={ArrowRight}>View all</Button>
              </Link>
            </CardHeader>
            {loading ? (
              <div className="space-y-2 p-4">
                {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-10 w-full" />)}
              </div>
            ) : queue.length === 0 ? (
              <EmptyState
                icon={ClipboardList}
                title="No analyses yet"
                description="Upload a chest CT to run the hybrid quantum-classical pipeline and populate the queue."
                action={<Link href="/dashboard/analyze"><Button variant="primary" icon={Scan}>Analyze scan</Button></Link>}
              />
            ) : (
              <Table>
                <thead>
                  <tr>
                    <Th>Patient</Th><Th>Finding</Th><Th>Risk</Th>
                    <Th className="text-right">Confidence</Th><Th>Analysed</Th>
                  </tr>
                </thead>
                <tbody>
                  {queue.map((q) => (
                    <Tr key={q.id} className="cursor-pointer"
                        onClick={() => router.push(`/dashboard/reports?analysis=${q.id}`)}>
                      <Td>
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{q.patient_external_id}</span>
                          {q.is_demo && <DemoBadge />}
                        </div>
                      </Td>
                      <Td className="text-[var(--color-ink-secondary)]">
                        {classLabel[q.predicted_class]}
                      </Td>
                      <Td><RiskBadge risk={q.risk_band as RiskBand} /></Td>
                      <Td className="text-right tabular">{pct(q.confidence)}</Td>
                      <Td className="text-[var(--color-ink-tertiary)]">{relativeTime(q.created_at)}</Td>
                    </Tr>
                  ))}
                </tbody>
              </Table>
            )}
          </Card>

          {/* Right column */}
          <div className="space-y-5">
            <Card>
              <CardHeader><CardTitle>Risk distribution</CardTitle></CardHeader>
              <CardBody className="space-y-3">
                {loading ? <Skeleton className="h-24 w-full" /> : (
                  (["High", "Intermediate", "Low"] as RiskBand[]).map((band) => {
                    const count = s?.by_risk?.[band] ?? 0;
                    const total = Object.values(s?.by_risk ?? {}).reduce((a, b) => a + b, 0) || 1;
                    const tone = band === "High" ? "bg-[var(--color-high)]"
                      : band === "Intermediate" ? "bg-[var(--color-moderate)]"
                      : "bg-[var(--color-low)]";
                    return (
                      <div key={band}>
                        <div className="mb-1 flex items-center justify-between text-[13px]">
                          <span className="flex items-center gap-2">
                            <span className={`size-2 rounded-full ${tone}`} aria-hidden />
                            {band}
                          </span>
                          <span className="tabular font-medium">{count}</span>
                        </div>
                        <div className="h-1.5 overflow-hidden rounded-full bg-[var(--color-surface-sunken)]">
                          <div className={`h-full rounded-full ${tone}`}
                               style={{ width: `${(count / total) * 100}%` }} />
                        </div>
                      </div>
                    );
                  })
                )}
              </CardBody>
            </Card>

            <Card>
              <CardHeader><CardTitle>Recent activity</CardTitle></CardHeader>
              <CardBody className="space-y-3 pt-0">
                {loading ? <Skeleton className="h-32 w-full" /> :
                 !data?.recent_activity?.length ? (
                  <p className="py-3 text-[13px] text-[var(--color-ink-tertiary)]">No activity recorded.</p>
                 ) : (
                  <ul className="space-y-3 pt-3">
                    {data.recent_activity.slice(0, 6).map((a) => (
                      <li key={a.id} className="flex gap-2.5">
                        <div className="mt-1 flex size-6 shrink-0 items-center justify-center rounded-full bg-[var(--color-surface-sunken)]">
                          {a.action.startsWith("analysis") ? <Activity className="size-3 text-[var(--color-ink-tertiary)]" aria-hidden />
                            : a.action.startsWith("report") ? <ClipboardList className="size-3 text-[var(--color-ink-tertiary)]" aria-hidden />
                            : <CheckCircle2 className="size-3 text-[var(--color-ink-tertiary)]" aria-hidden />}
                        </div>
                        <div className="min-w-0 text-[12.5px]">
                          <div className="text-[var(--color-ink)]">
                            {a.action.replace(/[._]/g, " ")}
                            {a.detail?.patient ? <> · <span className="font-medium">{String(a.detail.patient)}</span></> : null}
                          </div>
                          <div className="text-[var(--color-ink-tertiary)]">
                            {a.actor_role} · {relativeTime(a.created_at)}
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                 )}
              </CardBody>
            </Card>
          </div>
        </div>

        {data?.triage_rule && (
          <Notice tone="info" title="How priority is assigned">
            {data.triage_rule}
          </Notice>
        )}
      </div>
    </PageContainer>
  );
}
