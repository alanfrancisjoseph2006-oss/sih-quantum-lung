"use client";

import * as React from "react";
import Link from "next/link";
import {
  CalendarClock, CheckCircle2, Clock, Stethoscope,
} from "lucide-react";
import { PageContainer } from "@/components/PageContainer";
import {
  Badge, Card, DemoBadge, EmptyState, ErrorState, Notice, RiskBadge,
  SectionHead, SkeletonTable, Table, Td, Th, Tr,
} from "@/components/ui";
import { getFollowUps } from "@/lib/api";
import type { FollowUp, RiskBand } from "@/lib/types";
import { cn, formatDate, relativeTime } from "@/lib/utils";

const TABS = [
  { key: "", label: "All", icon: CalendarClock },
  { key: "overdue", label: "Overdue", icon: Clock },
  { key: "due", label: "Due soon", icon: CalendarClock },
  { key: "upcoming", label: "Upcoming", icon: Stethoscope },
  { key: "completed", label: "Completed", icon: CheckCircle2 },
] as const;

export default function FollowUpsPage() {
  const [items, setItems] = React.useState<FollowUp[]>([]);
  const [note, setNote] = React.useState("");
  const [tab, setTab] = React.useState<string>("");
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  const load = React.useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const r = await getFollowUps(tab || undefined);
      setItems(r.items); setNote(r.note);
    } catch (e) { setError(e instanceof Error ? e.message : "Failed to load follow-ups"); }
    finally { setLoading(false); }
  }, [tab]);
  React.useEffect(() => {
    // Deferred a tick so the state updates inside `load` do not run
    // synchronously inside the effect body (react-hooks/set-state-in-effect).
    const t = setTimeout(() => { void load(); }, 0);
    return () => clearTimeout(t);
  }, [load]);

  const counts = React.useMemo(() => items.reduce<Record<string, number>>((acc, i) => {
    acc[i.status] = (acc[i.status] ?? 0) + 1; return acc;
  }, {}), [items]);

  return (
    <PageContainer>
      <SectionHead title="Follow-up management"
        description="Review dates set by clinicians. The platform proposes no intervals of its own." />

      <div className="mt-5 space-y-5">
        <Notice tone="warning" title="Clinician-managed by design">
          Follow-up timing depends on nodule characteristics, patient risk factors and local
          guidance. This platform is not validated to recommend surveillance intervals, so it
          records the dates clinicians set rather than suggesting any.
        </Notice>

        <Card className="p-3">
          <div className="flex flex-wrap gap-1" role="tablist" aria-label="Filter follow-ups">
            {TABS.map((t) => {
              const active = tab === t.key;
              const n = t.key ? (counts[t.key] ?? 0) : items.length;
              return (
                <button key={t.key} role="tab" aria-selected={active} onClick={() => setTab(t.key)}
                  className={cn("inline-flex items-center gap-1.5 rounded-[var(--radius-md)] border px-2.5 py-1.5 text-[12.5px] font-medium transition-colors",
                    active ? "border-[var(--color-primary-border)] bg-[var(--color-primary-soft)] text-[var(--color-primary)]"
                           : "border-[var(--color-line)] text-[var(--color-ink-secondary)] hover:bg-[var(--color-surface-sunken)]")}>
                  <t.icon className="size-3.5" aria-hidden />{t.label}
                  {tab === "" && <span className="tabular text-[var(--color-ink-tertiary)]">{n}</span>}
                </button>
              );
            })}
          </div>
        </Card>

        <Card>
          {error ? <ErrorState message={error} onRetry={load} />
           : loading ? <SkeletonTable rows={6} cols={6} />
           : items.length === 0 ? (
            <EmptyState icon={CalendarClock} title="No follow-ups in this view"
              description="Open a patient record to set a review date." />
           ) : (
            <Table>
              <thead>
                <tr><Th>Patient</Th><Th>Risk at last scan</Th><Th>Last scan</Th>
                    <Th>Review due</Th><Th>Status</Th><Th>Notes</Th></tr>
              </thead>
              <tbody>
                {items.map((f) => (
                  <Tr key={f.id}>
                    <Td>
                      <Link href={`/dashboard/patients/${f.patient_id}`}
                            className="font-medium text-[var(--color-primary)] hover:underline">
                        {f.patient_external_id}
                      </Link>
                      {f.is_demo && <DemoBadge className="ml-2" />}
                    </Td>
                    <Td>{f.risk_band ? <RiskBadge risk={f.risk_band as RiskBand} /> : "—"}</Td>
                    <Td className="text-[var(--color-ink-tertiary)]">{relativeTime(f.last_scan_at)}</Td>
                    <Td className="tabular">{formatDate(f.review_due)}</Td>
                    <Td>
                      <Badge tone={f.status === "overdue" ? "high" : f.status === "due" ? "moderate"
                        : f.status === "completed" ? "low" : "neutral"}>
                        {f.status}
                      </Badge>
                    </Td>
                    <Td className="max-w-xs truncate text-[var(--color-ink-tertiary)]" title={f.notes}>
                      {f.notes || "—"}
                    </Td>
                  </Tr>
                ))}
              </tbody>
            </Table>
           )}
        </Card>

        {note && <Notice tone="neutral">{note}</Notice>}
      </div>
    </PageContainer>
  );
}
