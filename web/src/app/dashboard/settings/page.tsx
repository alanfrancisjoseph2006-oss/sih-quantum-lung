"use client";

import * as React from "react";
import { Check, Lock, ShieldCheck, ShieldX, UserCog, X } from "lucide-react";
import { PageContainer } from "@/components/PageContainer";
import {
  Badge, Card, CardBody, CardHeader, CardTitle, ErrorState, KeyValue,
  Notice, SectionHead, Skeleton,
} from "@/components/ui";
import { getMeta } from "@/lib/api";
import type { PlatformMeta } from "@/lib/types";

const ROLES = [
  { role: "Administrator", scope: "Full access, user and organisation management, audit review" },
  { role: "Radiologist", scope: "Review and approve reports, set follow-ups, submit assessments" },
  { role: "Doctor", scope: "View reports and patient timelines, submit assessments" },
  { role: "Technician", scope: "Upload scans and run analyses; cannot approve reports" },
  { role: "Researcher", scope: "Aggregate model analytics only; no patient-level access" },
];

export default function SettingsPage() {
  const [meta, setMeta] = React.useState<PlatformMeta | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  const load = React.useCallback(async () => {
    try { setMeta(await getMeta()); }
    catch (e) { setError(e instanceof Error ? e.message : "Failed to load"); }
  }, []);
  React.useEffect(() => {
    // Deferred a tick so the state updates inside `load` do not run
    // synchronously inside the effect body (react-hooks/set-state-in-effect).
    const t = setTimeout(() => { void load(); }, 0);
    return () => clearTimeout(t);
  }, [load]);

  if (error) return <PageContainer><ErrorState message={error} onRetry={load} /></PageContainer>;
  if (!meta) return <PageContainer><Card className="mt-5"><CardBody><Skeleton className="h-64" /></CardBody></Card></PageContainer>;

  const sec = meta.security_posture;

  return (
    <PageContainer>
      <SectionHead title="Settings"
        description="Platform configuration, deployed models and security posture." />

      <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle>Deployed configuration</CardTitle></CardHeader>
          <CardBody className="space-y-0">
            <KeyValue label="Backbone" value={meta.deployed?.backbone} />
            <KeyValue label="Primary model" value={meta.deployed?.primary_model} />
            <KeyValue label="Validation macro-F1" value={meta.deployed?.validation_f1_macro} />
            <KeyValue label="Quantum qubits" value={meta.engine?.n_qubits ?? "—"} />
            <KeyValue label="Engine status"
              value={meta.engine?.ready
                ? <Badge tone="low" icon={Check}>ready</Badge>
                : <Badge tone="high" icon={X}>unavailable</Badge>} mono={false} />
          </CardBody>
        </Card>

        <Card>
          <CardHeader><CardTitle>Compute</CardTitle></CardHeader>
          <CardBody className="space-y-0">
            <KeyValue label="Device" value={meta.hardware?.device} />
            <KeyValue label="GPU" value={<span className="font-normal">{meta.hardware?.gpu_name ?? "CPU only"}</span>} mono={false} />
            <KeyValue label="VRAM" value={meta.hardware?.vram_gb ? `${meta.hardware.vram_gb} GB` : "—"} />
            <KeyValue label="PyTorch" value={String(meta.hardware?.torch ?? "—")} />
            <KeyValue label="CUDA" value={meta.hardware?.cuda ? "available" : "not available"} />
          </CardBody>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader className="flex items-center justify-between">
            <CardTitle>Security &amp; privacy posture</CardTitle>
            <Badge tone="neutral" icon={ShieldCheck}>Research prototype</Badge>
          </CardHeader>
          <CardBody className="space-y-4">
            {sec?.statement && <Notice tone="info">{sec.statement}</Notice>}

            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <PostureList title="Implemented" icon={Check} tone="low" items={sec?.implemented ?? []} />
              <PostureList title="Designed, not enforced" icon={Lock} tone="moderate" items={sec?.designed_not_enforced ?? []} />
              <PostureList title="Explicitly not claimed" icon={ShieldX} tone="high" items={sec?.not_claimed ?? []} />
            </div>

            <Notice tone="warning" title="No regulatory claims are made">
              This is a research prototype. It carries no medical device approval, no compliance
              certification, and has not undergone clinical validation.
            </Notice>
          </CardBody>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader className="flex items-center justify-between">
            <CardTitle>Role-based access model</CardTitle>
            <Badge tone="neutral" icon={UserCog}>Designed</Badge>
          </CardHeader>
          <CardBody>
            <div className="overflow-hidden rounded-[var(--radius-md)] border border-[var(--color-line)]">
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="bg-[var(--color-surface-sunken)] text-[11.5px] uppercase tracking-[0.04em] text-[var(--color-ink-tertiary)]">
                    <th className="px-3 py-2 text-left font-medium">Role</th>
                    <th className="px-3 py-2 text-left font-medium">Intended scope</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--color-line)]">
                  {ROLES.map((r) => (
                    <tr key={r.role}>
                      <td className="px-3 py-2.5 font-medium">{r.role}</td>
                      <td className="px-3 py-2.5 text-[var(--color-ink-secondary)]">{r.scope}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-3 text-[12.5px] text-[var(--color-ink-tertiary)]">
              Roles exist in the data model and audit log. Endpoint-level enforcement is designed
              for but not active in this prototype — the platform resolves a single reviewer
              identity, and that assumption lives in one function so real authentication can be
              wired in without restructuring.
            </p>
          </CardBody>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader><CardTitle>Capabilities deliberately not provided</CardTitle></CardHeader>
          <CardBody className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {Object.entries(meta.unsupported_capabilities ?? {}).map(([k, v]) => (
              <div key={k} className="rounded-[var(--radius-md)] bg-[var(--color-surface-sunken)] p-3">
                <div className="text-[13px] font-medium capitalize">{k.replace(/_/g, " ")}</div>
                <p className="mt-0.5 text-[12.5px] text-[var(--color-ink-tertiary)]">{v}</p>
              </div>
            ))}
          </CardBody>
        </Card>
      </div>
    </PageContainer>
  );
}

function PostureList({ title, icon: Icon, tone, items }: {
  title: string; icon: React.ElementType; tone: "low" | "moderate" | "high"; items: string[];
}) {
  const color = { low: "text-[var(--color-low)]", moderate: "text-[var(--color-moderate)]", high: "text-[var(--color-high)]" }[tone];
  return (
    <div>
      <div className={`mb-2 flex items-center gap-1.5 text-[12.5px] font-semibold ${color}`}>
        <Icon className="size-3.5" aria-hidden />{title}
      </div>
      <ul className="space-y-1.5">
        {items.map((i, k) => (
          <li key={k} className="text-[12.5px] leading-relaxed text-[var(--color-ink-secondary)]">· {i}</li>
        ))}
      </ul>
    </div>
  );
}
