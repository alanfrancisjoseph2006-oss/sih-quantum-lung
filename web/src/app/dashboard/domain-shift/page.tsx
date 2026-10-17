"use client";

/**
 * Domain Shift Analysis (Phase 11).
 *
 * The investigative record: what was measured, in order, and what it showed.
 * Presented as an argument rather than a metric dump, because the conclusion
 * (the model was not looking at the nodule) only makes sense as a chain.
 */

import * as React from "react";
import { CheckCircle2, Search, TrendingDown } from "lucide-react";
import { PageContainer } from "@/components/PageContainer";
import {
  Badge, Card, CardBody, CardHeader, CardTitle, ErrorState, Notice,
  SectionHead, Skeleton, Table, Td, Th, Tr,
} from "@/components/ui";
import { API_BASE } from "@/lib/api";
import { cn, num } from "@/lib/utils";

interface DomainShift {
  signature_probe: {
    results: Record<string, { domain_auc: number; verdict: string }>;
    conclusion: string; n_slices: number;
  } | null;
  harmonisation: {
    baseline: { domain_auc: number; signal_auc: number };
    results: Record<string, { domain_auc: number; signal_auc: number; verdict: string }>;
    lowest_domain: { transform: string; domain_auc: number; signal_auc: number };
  } | null;
  robustness_arms: {
    results: {
      arm: string; in_domain: { roc_auc: number };
      external: { roc_auc: number; roc_auc_ci95: [number, number] }; gap: number;
    }[];
  } | null;
  inversion: {
    h1_size_confound: { auc_of_size_alone: number; benign_median_area_px: number; malignant_median_area_px: number };
    h2_score_tracks_size: { spearman_score_vs_area: number; p_value: number; model_auc: number };
  } | null;
  roi: { results: Record<string, { name: string; roc_auc: number; roc_auc_ci95: [number, number] }>;
         whole_image_reference_on_same_data: number; limitation: string } | null;
  hybrid_quantum_roi: {
    results: { name: string; roc_auc: number; roc_auc_ci95: [number, number] }[];
    quantum_advantage_test: Record<string, { mean_diff: number; p_value: number; significant: boolean }>;
  } | null;
  final_comparison: { table3_diagnosis: { step: string; value: string; meaning: string }[] } | null;
  external: { slice_level: { roc_auc: number; ci95: [number, number] } } | null;
}

export default function DomainShiftPage() {
  const [d, setD] = React.useState<DomainShift | null>(null);
  const [err, setErr] = React.useState<string | null>(null);

  React.useEffect(() => {
    const t = setTimeout(() => {
      fetch(`${API_BASE}/api/domain-shift`)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
        .then(setD)
        .catch((e) => setErr(e.message));
    }, 0);
    return () => clearTimeout(t);
  }, []);

  if (err) return <PageContainer><ErrorState message={err} /></PageContainer>;
  if (!d) {
    return <PageContainer><SectionHead title="Domain shift analysis" />
      <Card className="mt-5"><CardBody><Skeleton className="h-64" /></CardBody></Card></PageContainer>;
  }

  const sig = d.signature_probe;
  const har = d.harmonisation;
  const arms = d.robustness_arms?.results ?? [];
  const inv = d.inversion;
  const hq = d.hybrid_quantum_roi;
  const roi = d.roi;
  const best = hq?.results?.length
    ? hq.results.reduce((a, b) => (b.roc_auc > a.roc_auc ? b : a)) : null;

  return (
    <PageContainer>
      <SectionHead
        eyebrow="Phase 11"
        title="Domain shift analysis"
        description="Why the whole-image model failed on unseen hospitals, and what fixed it."
      />

      <div className="mt-5 space-y-5">
        <Notice tone="danger" title="Strong domain shift detected between IQ-OTH/NCCD and LIDC-IDRI">
          A linear probe on simple appearance descriptors names the source dataset at{" "}
          <strong>ROC-AUC {num(sig?.results?.whole?.domain_auc ?? 1, 3)}</strong>. Whatever a
          classifier learns from these images, dataset identity is freely available to it.
        </Notice>

        {/* 1 - where the signature lives */}
        <Card>
          <CardHeader>
            <CardTitle>1 · Where does the dataset signature live?</CardTitle>
            <p className="mt-0.5 text-[12.5px] text-[var(--color-ink-tertiary)]">
              Grouped cross-validation, so the probe cannot memorise individual patients
            </p>
          </CardHeader>
          <Table>
            <thead><tr><Th>View</Th><Th className="text-right">Domain ROC-AUC</Th><Th>Interpretation</Th></tr></thead>
            <tbody>
              {sig && Object.entries(sig.results).map(([k, v]) => (
                <Tr key={k}>
                  <Td className="font-medium capitalize">{k.replace(/_/g, " ")}</Td>
                  <Td className={cn("text-right tabular font-semibold",
                    v.domain_auc > 0.9 ? "text-[var(--color-high)]" : "text-[var(--color-moderate)]")}>
                    {num(v.domain_auc, 3)}
                  </Td>
                  <Td className="text-[var(--color-ink-secondary)]">{v.verdict}</Td>
                </Tr>
              ))}
            </tbody>
          </Table>
          <CardBody className="pt-4">
            <Notice tone="warning">{sig?.conclusion}</Notice>
          </CardBody>
        </Card>

        {/* 2 - harmonisation */}
        {har && (
          <Card>
            <CardHeader>
              <CardTitle>2 · Can preprocessing remove it?</CardTitle>
              <p className="mt-0.5 text-[12.5px] text-[var(--color-ink-tertiary)]">
                Domain AUC should fall toward 0.50; signal AUC should stay high
              </p>
            </CardHeader>
            <Table>
              <thead><tr><Th>Transform</Th><Th className="text-right">Domain AUC ↓</Th>
                <Th className="text-right">Signal AUC ↑</Th><Th>Verdict</Th></tr></thead>
              <tbody>
                {Object.entries(har.results).map(([k, v]) => (
                  <Tr key={k} className={cn(k === har.lowest_domain.transform && "bg-[var(--color-primary-soft)]/40")}>
                    <Td className="font-medium">{k}</Td>
                    <Td className="text-right tabular">{num(v.domain_auc, 3)}</Td>
                    <Td className="text-right tabular">{num(v.signal_auc, 3)}</Td>
                    <Td className="text-[12.5px] text-[var(--color-ink-secondary)]">{v.verdict}</Td>
                  </Tr>
                ))}
              </tbody>
            </Table>
            <CardBody className="pt-4">
              <Notice tone="warning">
                Best available is <strong>{num(har.lowest_domain.domain_auc, 3)}</strong> — far from
                0.50. Harmonisation helps but cannot remove the signature on its own.
              </Notice>
            </CardBody>
          </Card>
        )}

        {/* 3 - interventions */}
        {arms.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>3 · Do harmonisation and domain-adversarial training help?</CardTitle>
              <p className="mt-0.5 text-[12.5px] text-[var(--color-ink-tertiary)]">
                Identical architecture, seed and schedule — one switch differs per arm
              </p>
            </CardHeader>
            <Table>
              <thead><tr><Th>Arm</Th><Th className="text-right">In-domain AUC</Th>
                <Th className="text-right">External AUC</Th><Th className="text-right">95% CI</Th>
                <Th className="text-right">Gap</Th></tr></thead>
              <tbody>
                {arms.map((a) => (
                  <Tr key={a.arm}>
                    <Td className="font-medium">{a.arm.replace(/_/g, " + ")}</Td>
                    <Td className="text-right tabular">{num(a.in_domain.roc_auc, 4)}</Td>
                    <Td className="text-right tabular font-semibold text-[var(--color-high)]">
                      {num(a.external.roc_auc, 4)}
                    </Td>
                    <Td className="text-right tabular text-[12px] text-[var(--color-ink-tertiary)]">
                      {num(a.external.roc_auc_ci95[0], 3)}–{num(a.external.roc_auc_ci95[1], 3)}
                    </Td>
                    <Td className="text-right tabular">{num(a.gap, 3)}</Td>
                  </Tr>
                ))}
              </tbody>
            </Table>
            <CardBody className="pt-4">
              <Notice tone="danger" title="Neither intervention helped">
                Every arm remains at or below chance externally. Reported as measured — the
                adaptation code is correct, it simply cannot rescue a model whose features
                were wrong to begin with.
              </Notice>
            </CardBody>
          </Card>
        )}

        {/* 4 - the diagnosis */}
        {inv && (
          <Card className="border-[var(--color-primary-border)]">
            <CardHeader className="flex items-center justify-between">
              <CardTitle>4 · The diagnosis</CardTitle>
              <Badge tone="primary" icon={Search}>Key finding</Badge>
            </CardHeader>
            <CardBody className="space-y-4">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                <Stat label="Nodule size alone" value={num(inv.h1_size_confound.auc_of_size_alone, 3)}
                      sub="AUC predicting malignancy" tone="low" />
                <Stat label="The trained model" value={num(inv.h2_score_tracks_size.model_auc, 3)}
                      sub="AUC on the same data" tone="high" />
                <Stat label="Score vs nodule size"
                      value={`${inv.h2_score_tracks_size.spearman_score_vs_area >= 0 ? "+" : ""}${num(inv.h2_score_tracks_size.spearman_score_vs_area, 3)}`}
                      sub={`Spearman ρ (p = ${num(inv.h2_score_tracks_size.p_value, 2)})`} tone="high" />
              </div>
              <Notice tone="info" title="The model was never looking at the nodule">
                A single scalar — how large the nodule is, which is genuine radiology —
                achieves {num(inv.h1_size_confound.auc_of_size_alone, 3)}. The trained
                network&rsquo;s score is <strong>statistically uncorrelated</strong> with it
                (ρ = {num(inv.h2_score_tracks_size.spearman_score_vs_area, 3)},
                p = {num(inv.h2_score_tracks_size.p_value, 2)}). It was classifying the
                picture, not the lesion. The task was never impossible — the input was wrong.
              </Notice>
            </CardBody>
          </Card>
        )}

        {/* 5 - the fix */}
        {hq && roi && (
          <Card className="border-[var(--color-low-border)]">
            <CardHeader className="flex items-center justify-between">
              <div>
                <CardTitle>5 · The fix — classify the nodule, not the image</CardTitle>
                <p className="mt-0.5 text-[12.5px] text-[var(--color-ink-tertiary)]">
                  Patient-level split on real LIDC patient IDs
                </p>
              </div>
              <Badge tone="low" icon={CheckCircle2}>0.46 → {num(best?.roc_auc ?? 0, 2)}</Badge>
            </CardHeader>
            <Table>
              <thead><tr><Th>Model</Th><Th className="text-right">AUC</Th>
                <Th className="text-right">95% CI</Th></tr></thead>
              <tbody>
                {hq.results.map((r) => (
                  <Tr key={r.name} className={cn(r.name === best?.name && "bg-[var(--color-low-soft)]/50")}>
                    <Td className="font-medium">{r.name}</Td>
                    <Td className="text-right tabular font-semibold">{num(r.roc_auc, 4)}</Td>
                    <Td className="text-right tabular text-[12px] text-[var(--color-ink-tertiary)]">
                      {num(r.roc_auc_ci95[0], 3)}–{num(r.roc_auc_ci95[1], 3)}
                    </Td>
                  </Tr>
                ))}
                <Tr>
                  <Td className="text-[var(--color-ink-tertiary)]">ROI CNN (EfficientNet-B3)</Td>
                  <Td className="text-right tabular">{num(roi.results.roi_cnn?.roc_auc, 4)}</Td>
                  <Td />
                </Tr>
                <Tr>
                  <Td className="italic text-[var(--color-ink-tertiary)]">Whole-image model, same data</Td>
                  <Td className="text-right tabular italic text-[var(--color-high)]">
                    {num(roi.whole_image_reference_on_same_data, 4)}
                  </Td>
                  <Td />
                </Tr>
              </tbody>
            </Table>
            <CardBody className="space-y-3 pt-4">
              <Notice tone="warning" title="A 10-feature model beats EfficientNet-B3">
                {num(best?.roc_auc ?? 0, 3)} versus {num(roi.results.roi_cnn?.roc_auc, 3)}. With 35
                training patients the deep network overfits. Reaching for a bigger model here is
                the wrong instinct.
              </Notice>
              {hq.quantum_advantage_test && Object.entries(hq.quantum_advantage_test).map(([k, v]) => (
                <Notice key={k} tone={v.significant && v.mean_diff < 0 ? "warning" : "neutral"}
                        icon={TrendingDown}>
                  <strong>{k}</strong>: Δ AUC = {v.mean_diff >= 0 ? "+" : ""}{num(v.mean_diff, 4)},
                  p = {num(v.p_value, 3)} →{" "}
                  <strong>
                    {v.significant
                      ? (v.mean_diff < 0 ? "significantly worse than classical" : "significantly better")
                      : "no significant difference"}
                  </strong>
                </Notice>
              ))}
              <Notice tone="danger" title="The honest limit on this result">
                {roi.limitation}
              </Notice>
            </CardBody>
          </Card>
        )}
      </div>
    </PageContainer>
  );
}

function Stat({ label, value, sub, tone }: {
  label: string; value: string; sub: string; tone: "low" | "high" | "neutral";
}) {
  const c = { low: "text-[var(--color-low)]", high: "text-[var(--color-high)]",
              neutral: "text-[var(--color-ink)]" }[tone];
  return (
    <div className="rounded-[var(--radius-md)] border border-[var(--color-line)] p-3">
      <div className="text-[11.5px] font-medium uppercase tracking-[0.05em] text-[var(--color-ink-tertiary)]">
        {label}
      </div>
      <div className={cn("mt-1 text-[24px] font-semibold tabular", c)}>{value}</div>
      <div className="text-[11.5px] text-[var(--color-ink-tertiary)]">{sub}</div>
    </div>
  );
}
