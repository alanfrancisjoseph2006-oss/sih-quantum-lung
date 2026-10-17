"use client";

import * as React from "react";
import Link from "next/link";
import { AlertTriangle, CheckCircle2, ClipboardList, Filter, Search, XCircle } from "lucide-react";
import { PageContainer } from "@/components/PageContainer";
import { DisclaimerBar } from "@/components/DisclaimerBar";
import {
  Badge, Button, Card, DemoBadge, EmptyState, ErrorState, Input, Notice,
  RiskBadge, SectionHead, Select, SkeletonTable, Table, Td, Th, Tr,
} from "@/components/ui";
import { getQueue } from "@/lib/api";
import type { QueueItem, RiskBand } from "@/lib/types";
import { classLabel, cn, pct, relativeTime, reviewStatusLabel } from "@/lib/utils";

const TABS = [
  { key: "", label: "All", icon: ClipboardList },
  { key: "high", label: "High priority", icon: XCircle },
  { key: "attention", label: "Attention required", icon: AlertTriangle },
  { key: "routine", label: "Routine", icon: CheckCircle2 },
] as const;

export default function QueuePage() {
  const [items, setItems] = React.useState<QueueItem[]>([]);
  const [rule, setRule] = React.useState("");
  const [tab, setTab] = React.useState<string>("");
  const [status, setStatus] = React.useState("");
  const [search, setSearch] = React.useState("");
  const [sort, setSort] = React.useState<"priority" | "confidence" | "recent">("priority");
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  const load = React.useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const r = await getQueue({ priority: tab || undefined, review_status: status || undefined,
                                 search: search || undefined, limit: 300 });
      setItems(r.items); setRule(r.rule);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load queue");
    } finally { setLoading(false); }
  }, [tab, status, search]);

  React.useEffect(() => {
    // Debounced, and deferred so no setState runs synchronously in the effect.
    const t = setTimeout(() => { void load(); }, search ? 300 : 0);
    return () => clearTimeout(t);
  }, [load, search]);

  const sorted = React.useMemo(() => {
    const order = { high: 0, attention: 1, routine: 2 } as Record<string, number>;
    const copy = [...items];
    if (sort === "confidence") copy.sort((a, b) => b.confidence - a.confidence);
    else if (sort === "recent") copy.sort((a, b) => b.created_at.localeCompare(a.created_at));
    else copy.sort((a, b) => order[a.priority] - order[b.priority] || b.created_at.localeCompare(a.created_at));
    return copy;
  }, [items, sort]);

  const counts = React.useMemo(() => ({
    high: items.filter((i) => i.priority === "high").length,
    attention: items.filter((i) => i.priority === "attention").length,
    routine: items.filter((i) => i.priority === "routine").length,
  }), [items]);

  return (
    <PageContainer>
      <SectionHead
        title="AI priority queue"
        description="Radiology triage ordered by model output. Every case still requires human review."
      />

      <div className="mt-5 space-y-5">
        <DisclaimerBar compact />

        {/* Filter bar */}
        <Card className="p-3">
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex flex-wrap gap-1" role="tablist" aria-label="Filter by priority">
              {TABS.map((t) => {
                const active = tab === t.key;
                const n = t.key ? counts[t.key as keyof typeof counts] : items.length;
                return (
                  <button
                    key={t.key} role="tab" aria-selected={active}
                    onClick={() => setTab(t.key)}
                    className={cn(
                      "inline-flex items-center gap-1.5 rounded-[var(--radius-md)] border px-2.5 py-1.5",
                      "text-[12.5px] font-medium transition-colors",
                      active ? "border-[var(--color-primary-border)] bg-[var(--color-primary-soft)] text-[var(--color-primary)]"
                             : "border-[var(--color-line)] text-[var(--color-ink-secondary)] hover:bg-[var(--color-surface-sunken)]",
                    )}
                  >
                    <t.icon className="size-3.5" aria-hidden />
                    {t.label}
                    <span className="tabular text-[var(--color-ink-tertiary)]">{n}</span>
                  </button>
                );
              })}
            </div>

            <div className="ml-auto flex flex-wrap items-center gap-2">
              <div className="relative">
                <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-[var(--color-ink-disabled)]" aria-hidden />
                <Input value={search} onChange={(e) => setSearch(e.target.value)}
                       placeholder="Patient ID" aria-label="Search patient ID" className="w-40 pl-8" />
              </div>
              <Select value={status} onChange={(e) => setStatus(e.target.value)}
                      aria-label="Filter by review status" className="w-44">
                <option value="">All review states</option>
                <option value="ai_generated">AI generated</option>
                <option value="in_review">In review</option>
                <option value="approved">Approved</option>
                <option value="rejected">Rejected</option>
              </Select>
              <Select value={sort} onChange={(e) => setSort(e.target.value as typeof sort)}
                      aria-label="Sort by" className="w-40">
                <option value="priority">Sort: priority</option>
                <option value="confidence">Sort: confidence</option>
                <option value="recent">Sort: most recent</option>
              </Select>
            </div>
          </div>
        </Card>

        <Card>
          {error ? <ErrorState message={error} onRetry={load} />
           : loading ? <SkeletonTable rows={8} cols={6} />
           : sorted.length === 0 ? (
            <EmptyState icon={Filter} title="No cases match these filters"
              description="Adjust the priority, review state or search term."
              action={<Button onClick={() => { setTab(""); setStatus(""); setSearch(""); }}>Clear filters</Button>} />
           ) : (
            <Table>
              <thead>
                <tr>
                  <Th>Priority</Th><Th>Patient</Th><Th>Scan</Th><Th>AI finding</Th>
                  <Th>Risk</Th><Th className="text-right">Confidence</Th>
                  <Th className="text-right">Agreement</Th><Th>Review</Th><Th>Analysed</Th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((q) => (
                  <Tr key={q.id}>
                    <Td>
                      <PriorityCell priority={q.priority} />
                    </Td>
                    <Td>
                      <Link href={`/dashboard/patients/${q.patient_id}`}
                            className="font-medium text-[var(--color-primary)] hover:underline">
                        {q.patient_external_id}
                      </Link>
                      {q.is_demo && <DemoBadge className="ml-2" />}
                    </Td>
                    <Td className="max-w-[180px] truncate text-[var(--color-ink-tertiary)]" title={q.scan_filename}>
                      {q.scan_filename}
                    </Td>
                    <Td className="text-[var(--color-ink-secondary)]">{classLabel[q.predicted_class]}</Td>
                    <Td><RiskBadge risk={q.risk_band as RiskBand} /></Td>
                    <Td className="text-right tabular">{pct(q.confidence)}</Td>
                    <Td className={cn("text-right tabular",
                      q.branch_agreement < 0.75 && "font-medium text-[var(--color-moderate)]")}>
                      {pct(q.branch_agreement, 0)}
                    </Td>
                    <Td>
                      <Badge tone={q.review_status === "approved" ? "low"
                        : q.review_status === "rejected" ? "high" : "neutral"}>
                        {reviewStatusLabel[q.review_status] ?? q.review_status}
                      </Badge>
                    </Td>
                    <Td className="whitespace-nowrap text-[var(--color-ink-tertiary)]">
                      {relativeTime(q.created_at)}
                    </Td>
                  </Tr>
                ))}
              </tbody>
            </Table>
           )}
        </Card>

        {rule && <Notice tone="info" title="How priority is assigned">{rule}</Notice>}
      </div>
    </PageContainer>
  );
}

/** Priority is shown as icon + text + colour, never colour alone. */
function PriorityCell({ priority }: { priority: string }) {
  const cfg = {
    high: { Icon: XCircle, label: "High", cls: "text-[var(--color-high)]", bar: "bg-[var(--color-high)]" },
    attention: { Icon: AlertTriangle, label: "Attention", cls: "text-[var(--color-moderate)]", bar: "bg-[var(--color-moderate)]" },
    routine: { Icon: CheckCircle2, label: "Routine", cls: "text-[var(--color-low)]", bar: "bg-[var(--color-low)]" },
  }[priority] ?? { Icon: CheckCircle2, label: priority, cls: "", bar: "bg-[var(--color-line)]" };

  return (
    <span className="flex items-center gap-2">
      <span className={cn("h-6 w-[3px] rounded-full", cfg.bar)} aria-hidden />
      <cfg.Icon className={cn("size-4", cfg.cls)} aria-hidden />
      <span className={cn("text-[12.5px] font-medium", cfg.cls)}>{cfg.label}</span>
    </span>
  );
}
