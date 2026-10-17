"use client";

import * as React from "react";
import Link from "next/link";
import { Search, Users } from "lucide-react";
import { PageContainer } from "@/components/PageContainer";
import {
  Card, DemoBadge, EmptyState, ErrorState, Input, Notice, RiskBadge,
  SectionHead, SkeletonTable, Table, Td, Th, Tr,
} from "@/components/ui";
import { getPatients } from "@/lib/api";
import type { PatientSummary, RiskBand } from "@/lib/types";
import {
  formatDate, relativeTime,
} from "@/lib/utils";

export default function PatientsPage() {
  const [items, setItems] = React.useState<PatientSummary[]>([]);
  const [search, setSearch] = React.useState("");
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  // Seed the search box from ?search= (e.g. arriving from the top-bar search).
  React.useEffect(() => {
    const t = setTimeout(() => {
      const q = new URLSearchParams(window.location.search).get("search");
      if (q) setSearch(q);
    }, 0);
    return () => clearTimeout(t);
  }, []);

  const load = React.useCallback(async () => {
    setLoading(true); setError(null);
    try { setItems((await getPatients(search || undefined)).items); }
    catch (e) { setError(e instanceof Error ? e.message : "Failed to load patients"); }
    finally { setLoading(false); }
  }, [search]);

  React.useEffect(() => {
    // Debounced, and deferred so no setState runs synchronously in the effect.
    const t = setTimeout(() => { void load(); }, search ? 300 : 0);
    return () => clearTimeout(t);
  }, [load, search]);

  return (
    <PageContainer>
      <SectionHead
        title="Patients"
        description="Opaque identifiers only — no names, dates of birth or contact details are stored."
      />

      <div className="mt-5 space-y-5">
        <Notice tone="neutral">
          This platform deliberately models no personally identifying information. Records are keyed
          by an opaque identifier so there is nowhere for protected health information to be stored.
        </Notice>

        <Card className="p-3">
          <div className="relative max-w-xs">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-[var(--color-ink-disabled)]" aria-hidden />
            <Input value={search} onChange={(e) => setSearch(e.target.value)}
                   placeholder="Search patient identifier" aria-label="Search patients" className="pl-8" />
          </div>
        </Card>

        <Card>
          {error ? <ErrorState message={error} onRetry={load} />
           : loading ? <SkeletonTable rows={6} cols={6} />
           : items.length === 0 ? (
            <EmptyState icon={Users} title="No patients found"
              description={search ? "No identifier matches that search." : "Analyse a scan to create the first patient record."} />
           ) : (
            <Table>
              <thead>
                <tr>
                  <Th>Identifier</Th><Th>Age</Th><Th>Sex</Th>
                  <Th className="text-right">Scans</Th><Th>Latest risk</Th>
                  <Th>Last analysed</Th><Th>Created</Th>
                </tr>
              </thead>
              <tbody>
                {items.map((p) => (
                  <Tr key={p.id}>
                    <Td>
                      <Link href={`/dashboard/patients/${p.id}`}
                            className="font-medium text-[var(--color-primary)] hover:underline">
                        {p.external_id}
                      </Link>
                      {p.is_demo && <DemoBadge className="ml-2" />}
                    </Td>
                    <Td className="tabular">{p.age ?? "—"}</Td>
                    <Td>{p.sex ?? "—"}</Td>
                    <Td className="text-right tabular">{p.scan_count}</Td>
                    <Td>{p.latest_risk ? <RiskBadge risk={p.latest_risk as RiskBand} /> : "—"}</Td>
                    <Td className="text-[var(--color-ink-tertiary)]">{relativeTime(p.latest_scan_at)}</Td>
                    <Td className="text-[var(--color-ink-tertiary)]">{formatDate(p.created_at)}</Td>
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
