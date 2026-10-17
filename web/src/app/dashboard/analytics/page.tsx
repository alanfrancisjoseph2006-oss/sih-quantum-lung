"use client";

/**
 * Model analytics — real experimental results only.
 *
 * Every number here is read from reports/model_comparison.json, which is
 * produced by scripts/06_model_selection.py. If a metric is missing the UI says
 * so rather than substituting a plausible value.
 */

import * as React from "react";
import {
  Activity, FlaskConical, TrendingDown,
} from "lucide-react";
import { PageContainer } from "@/components/PageContainer";
import {
  Badge, Card, CardBody, CardHeader, CardTitle, EmptyState, ErrorState,
  Notice, SectionHead, Select, Skeleton, Table, Td, Th, Tr,
} from "@/components/ui";
import { getExternalValidation, getModelComparison } from "@/lib/api";
import type { ExternalValidation, ModelComparison } from "@/lib/types";
import { cn, num, pct } from "@/lib/utils";

function modelKind(name: string) {
  if (name.startsWith("Quantum")) return { label: "Quantum", tone: "quantum" as const };
  if (name.startsWith("Hybrid")) return { label: "Hybrid", tone: "moderate" as const };
  if (name.startsWith("CNN")) return { label: "Deep learning", tone: "primary" as const };
  return { label: "Classical ML", tone: "neutral" as const };
}

export default function AnalyticsPage() {
  const [data, setData] = React.useState<ModelComparison | null>(null);
  const [ext, setExt] = React.useState<ExternalValidation | null>(null);
  const [backbone, setBackbone] = React.useState<string>("");
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  const load = React.useCallback(async () => {
    setLoading(true); setError(null);
    try {
      const d = await getModelComparison();
      setData(d);
      getExternalValidation().then(setExt).catch(() => {});
      setBackbone(d.deployed?.backbone ?? Object.keys(d.backbones)[0]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load results");
    } finally { setLoading(false); }
  }, []);

  React.useEffect(() => {
    // Deferred a tick so the state updates inside `load` do not run
    // synchronously inside the effect body (react-hooks/set-state-in-effect).
    const t = setTimeout(() => { void load(); }, 0);
    return () => clearTimeout(t);
  }, [load]);

  if (error) return <PageContainer><ErrorState message={error} onRetry={load} /></PageContainer>;
  if (loading) {
    return (
      <PageContainer>
        <SectionHead title="Model analytics" />
        <Card className="mt-5"><CardBody><Skeleton className="h-64 w-full" /></CardBody></Card>
      </PageContainer>
    );
  }
  if (!data?.backbones || !Object.keys(data.backbones).length) {
    return (
      <PageContainer>
        <SectionHead title="Model analytics" />
        <Card className="mt-5">
          <EmptyState icon={FlaskConical} title="No validated experiment available"
            description="Run scripts/06_model_selection.py to generate comparison results." />
        </Card>
      </PageContainer>
    );
  }

  const bb = data.backbones[backbone];
  const rows = Object.entries(bb.models)
    .sort((a, b) => b[1].test.f1_macro - a[1].test.f1_macro);
  const qa = bb.quantum_advantage_test ?? {};

  return (
    <PageContainer>
      <SectionHead
        eyebrow="Experimental results"
        title="Model analytics"
        description="Measured on the held-out test split under group-aware splitting. Confidence intervals are percentile bootstrap."
        actions={
          <Select value={backbone} onChange={(e) => setBackbone(e.target.value)}
                  aria-label="Select backbone" className="w-52">
            {Object.keys(data.backbones).map((k) => (
              <option key={k} value={k}>{k}{data.deployed?.backbone === k ? " (deployed)" : ""}</option>
            ))}
          </Select>
        }
      />

      <div className="mt-5 space-y-5">
        {ext && <ExternalValidationCard ext={ext} />}

        <Notice tone="info" title="These are real measurements, not illustrative figures">
          Produced by <code className="font-mono text-[12px]">scripts/06_model_selection.py</code> over{" "}
          {data.n_bootstrap.toLocaleString()} bootstrap resamples with seed {data.seed}. Where a
          metric could not be computed it is shown as “—” rather than estimated.
        </Notice>

        {/* Comparison table */}
        <Card>
          <CardHeader>
            <CardTitle>Model comparison</CardTitle>
            <p className="mt-0.5 text-[12.5px] text-[var(--color-ink-tertiary)]">
              {rows.length} candidates · sorted by test macro-F1
            </p>
          </CardHeader>
          <Table>
            <thead>
              <tr>
                <Th>Model</Th><Th>Type</Th>
                <Th className="text-right">Accuracy</Th><Th className="text-right">Precision</Th>
                <Th className="text-right">Recall</Th><Th className="text-right">F1 (macro)</Th>
                <Th className="text-right">95% CI</Th><Th className="text-right">ROC-AUC</Th>
                <Th className="text-right">Malig. sens.</Th><Th className="text-right">ECE</Th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([name, m]) => {
                const kind = modelKind(name);
                const isPrimary = name === bb.primary_model;
                return (
                  <Tr key={name} className={cn(isPrimary && "bg-[var(--color-primary-soft)]/40")}>
                    <Td>
                      <span className="flex items-center gap-2">
                        {isPrimary && <span className="h-5 w-[3px] rounded-full bg-[var(--color-primary)]" aria-hidden />}
                        <span className="font-medium">{name}</span>
                        {isPrimary && <Badge tone="primary">Deployed</Badge>}
                      </span>
                    </Td>
                    <Td><Badge tone={kind.tone}>{kind.label}</Badge></Td>
                    <Td className="text-right tabular">{pct(m.test.accuracy)}</Td>
                    <Td className="text-right tabular">{pct(m.test.precision_macro)}</Td>
                    <Td className="text-right tabular">{pct(m.test.recall_macro)}</Td>
                    <Td className="text-right tabular font-semibold">{pct(m.test.f1_macro)}</Td>
                    <Td className="text-right tabular text-[12px] text-[var(--color-ink-tertiary)]">
                      {num(m.test_ci?.lo95, 3)}–{num(m.test_ci?.hi95, 3)}
                    </Td>
                    <Td className="text-right tabular">{num(m.test.roc_auc_macro_ovr)}</Td>
                    <Td className="text-right tabular">
                      {m.test.malignant_sensitivity != null ? pct(m.test.malignant_sensitivity) : "—"}
                    </Td>
                    <Td className="text-right tabular">{num(m.test_calibration?.ece)}</Td>
                  </Tr>
                );
              })}
            </tbody>
          </Table>
        </Card>

        {/* F1 bars with CI */}
        <Card>
          <CardHeader>
            <CardTitle>Test macro-F1 with 95% confidence intervals</CardTitle>
            <p className="mt-0.5 text-[12.5px] text-[var(--color-ink-tertiary)]">
              Overlapping intervals mean the difference is not statistically resolvable
            </p>
          </CardHeader>
          <CardBody className="space-y-2.5">
            {rows.map(([name, m]) => {
              const kind = modelKind(name);
              const barColor = {
                Quantum: "bg-[var(--color-quantum)]", Hybrid: "bg-[var(--color-moderate)]",
                "Deep learning": "bg-[var(--color-primary)]", "Classical ML": "bg-[#5b7fd0]",
              }[kind.label]!;
              const lo = m.test_ci?.lo95 ?? m.test.f1_macro;
              const hi = m.test_ci?.hi95 ?? m.test.f1_macro;
              return (
                <div key={name} className="flex items-center gap-3">
                  <div className="w-52 shrink-0 truncate text-[12.5px] text-[var(--color-ink-secondary)]" title={name}>
                    {name}
                  </div>
                  <div className="relative h-5 flex-1 rounded bg-[var(--color-surface-sunken)]">
                    <div className={cn("absolute inset-y-0 left-0 rounded", barColor)}
                         style={{ width: `${(m.test.f1_macro / 1) * 100}%`, opacity: 0.85 }} />
                    <div className="absolute inset-y-0 border-l border-r border-[var(--color-ink)]/40"
                         style={{ left: `${lo * 100}%`, width: `${Math.max((hi - lo) * 100, 0.4)}%` }}
                         title={`95% CI ${num(lo)}–${num(hi)}`} />
                  </div>
                  <div className="w-14 shrink-0 text-right text-[12.5px] tabular font-medium">
                    {pct(m.test.f1_macro)}
                  </div>
                </div>
              );
            })}
            <div className="flex flex-wrap gap-4 border-t border-[var(--color-line)] pt-3 text-[11.5px] text-[var(--color-ink-tertiary)]">
              {[["Deep learning", "bg-[var(--color-primary)]"], ["Classical ML", "bg-[#5b7fd0]"],
                ["Quantum", "bg-[var(--color-quantum)]"], ["Hybrid", "bg-[var(--color-moderate)]"]].map(([l, c]) => (
                <span key={l} className="flex items-center gap-1.5">
                  <span className={cn("h-1 w-4 rounded", c)} aria-hidden />{l}
                </span>
              ))}
            </div>
          </CardBody>
        </Card>

        {/* Selection + quantum advantage */}
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
          <Card>
            <CardHeader><CardTitle>Model selection</CardTitle></CardHeader>
            <CardBody className="space-y-3">
              <Notice tone="info" title={`argmax on validation: ${bb.argmax_validation}`}>
                <strong>{bb.statistically_tied?.length ?? 0} of {rows.length}</strong> candidates were
                statistically indistinguishable from it (paired bootstrap, p ≥ 0.05). The clinical
                tie-break then chose <strong>{bb.primary_model}</strong>.
              </Notice>
              <p className="text-[12.5px] leading-relaxed text-[var(--color-ink-secondary)]">
                {bb.selection_policy}
              </p>
            </CardBody>
          </Card>

          <Card>
            <CardHeader className="flex items-center justify-between">
              <CardTitle>Quantum advantage test</CardTitle>
              <Badge tone="quantum" icon={Activity}>Falsifiable</Badge>
            </CardHeader>
            <CardBody className="space-y-2.5">
              {Object.keys(qa).length === 0 ? (
                <p className="text-[13px] text-[var(--color-ink-tertiary)]">No validated experiment available.</p>
              ) : Object.entries(qa).map(([k, v]) => (
                <Notice key={k} tone={v["significant_at_0.05"] ? "success" : "warning"}
                        icon={v["significant_at_0.05"] ? undefined : TrendingDown}
                        title={k.replace(/_/g, " ")}>
                  Δ macro-F1 = <strong>{v.mean_diff >= 0 ? "+" : ""}{num(v.mean_diff)}</strong>{" "}
                  (95% CI {num(v.ci95_lo)} … {num(v.ci95_hi)}), p = <strong>{num(v.p_value)}</strong> →{" "}
                  <strong>
                    {v["significant_at_0.05"] ? "statistically significant"
                      : "not significant — no quantum advantage demonstrated"}
                  </strong>.
                </Notice>
              ))}
            </CardBody>
          </Card>
        </div>

        {/* Confusion matrix for the deployed model */}
        {bb.models[bb.primary_model] && (
          <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
            <ConfusionCard name={bb.primary_model} m={bb.models[bb.primary_model].test} />
            <Card>
              <CardHeader><CardTitle>Per-class performance — {bb.primary_model}</CardTitle></CardHeader>
              <Table>
                <thead>
                  <tr>
                    <Th>Class</Th><Th className="text-right">Sensitivity</Th>
                    <Th className="text-right">Specificity</Th><Th className="text-right">Precision</Th>
                    <Th className="text-right">F1</Th><Th className="text-right">n</Th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(bb.models[bb.primary_model].test.per_class).map(([cls, v]) => (
                    <Tr key={cls}>
                      <Td className="font-medium capitalize">{cls}</Td>
                      <Td className="text-right tabular">{pct(v.recall_sensitivity)}</Td>
                      <Td className="text-right tabular">{pct(v.specificity)}</Td>
                      <Td className="text-right tabular">{pct(v.precision)}</Td>
                      <Td className="text-right tabular">{pct(v.f1)}</Td>
                      <Td className="text-right tabular text-[var(--color-ink-tertiary)]">{v.support}</Td>
                    </Tr>
                  ))}
                </tbody>
              </Table>
            </Card>
          </div>
        )}

        <Notice tone="warning" title="Why these numbers are lower than published results on this dataset">
          Papers reporting 99–100% on IQ-OTH/NCCD typically use a random split, which leaks
          near-duplicate slices of the same patient across train and test. Under our group-aware
          split that leakage path is removed, so these figures are a more realistic estimate of
          generalisation.
        </Notice>
      </div>
    </PageContainer>
  );
}


/**
 * External validation (D26).
 *
 * Placed at the top of this page deliberately: every other number below is
 * in-domain, and this is the context needed to read them correctly.
 */
function ExternalValidationCard({ ext }: { ext: ExternalValidation }) {
  const sl = ext.slice_level;
  const idr = ext.in_domain_reference;
  const atChance = sl.ci95[0] <= 0.5 && sl.ci95[1] >= 0.5;

  return (
    <Card className="border-[var(--color-high-border)]">
      <CardHeader className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <CardTitle>External validation — {ext.external_dataset}</CardTitle>
          <p className="mt-0.5 text-[12.5px] text-[var(--color-ink-tertiary)]">
            {ext.n_slices} slices · {ext.n_patients} patients · a dataset this model has never seen
          </p>
        </div>
        <Badge tone="high" icon={TrendingDown}>Does not generalise</Badge>
      </CardHeader>
      <CardBody className="space-y-4">
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div className="rounded-[var(--radius-md)] border border-[var(--color-line)] p-3">
            <div className="text-[11.5px] font-medium uppercase tracking-[0.05em] text-[var(--color-ink-tertiary)]">
              In-domain (IQ-OTH)
            </div>
            <div className="mt-1 text-[24px] font-semibold tabular text-[var(--color-low)]">
              {num(idr.roc_auc)}
            </div>
            <div className="text-[11.5px] text-[var(--color-ink-tertiary)]">
              ROC-AUC · n={idr.n}
            </div>
          </div>
          <div className="rounded-[var(--radius-md)] border border-[var(--color-high-border)] bg-[var(--color-high-soft)] p-3">
            <div className="text-[11.5px] font-medium uppercase tracking-[0.05em] text-[var(--color-high)]">
              External (LIDC)
            </div>
            <div className="mt-1 text-[24px] font-semibold tabular text-[var(--color-high)]">
              {num(sl.roc_auc)}
            </div>
            <div className="text-[11.5px] text-[var(--color-ink-tertiary)]">
              95% CI {num(sl.ci95[0], 2)}–{num(sl.ci95[1], 2)} · n={ext.n_slices}
            </div>
          </div>
          <div className="rounded-[var(--radius-md)] border border-[var(--color-line)] p-3">
            <div className="text-[11.5px] font-medium uppercase tracking-[0.05em] text-[var(--color-ink-tertiary)]">
              Change
            </div>
            <div className="mt-1 text-[24px] font-semibold tabular text-[var(--color-high)]">
              {num(ext.auc_change_external_minus_internal)}
            </div>
            <div className="text-[11.5px] text-[var(--color-ink-tertiary)]">ROC-AUC</div>
          </div>
        </div>

        {atChance && (
          <Notice tone="danger" title="At chance on external data">
            0.50 is chance, and the external confidence interval
            ({num(sl.ci95[0], 3)} – {num(sl.ci95[1], 3)}) straddles it. On scans from other
            hospitals this model has <strong>no measurable ability</strong> to separate benign
            from malignant. Every other figure on this page is <strong>in-domain only</strong>.
          </Notice>
        )}

        <Notice tone="warning" title="Why the argmax sensitivity looks high anyway">
          The model labelled{" "}
          <strong>
            {Math.round(100 * (ext.prediction_distribution.malignant ?? 0) / ext.n_slices)}%
          </strong>{" "}
          of external slices malignant. That yields sensitivity {num(sl.sensitivity_argmax, 3)}{" "}
          alongside specificity <strong>{num(sl.specificity_argmax, 3)}</strong> — the signature
          of a model answering &ldquo;malignant&rdquo; by default rather than discriminating.
          Quoting that sensitivity on its own would be misleading.
        </Notice>

        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          <div>
            <h4 className="mb-1.5 text-[13px] font-semibold">Why this is a fair test</h4>
            <p className="text-[12.5px] leading-relaxed text-[var(--color-ink-secondary)]">
              {ext.why_fair}
            </p>
          </div>
          <div>
            <h4 className="mb-1.5 text-[13px] font-semibold">Limits of this test</h4>
            <ul className="space-y-1 text-[12px] leading-relaxed text-[var(--color-ink-tertiary)]">
              {ext.caveats.map((c, i) => <li key={i}>· {c}</li>)}
            </ul>
          </div>
        </div>

        <Notice tone="neutral">
          <strong>Honest capability statement:</strong> this model is validated on IQ-OTH/NCCD and
          demonstrably does not generalise beyond it. It is a working research pipeline, not a
          deployable screening tool. Fixing this needs multi-centre training data, domain
          adaptation, and prospective validation at each target site.
        </Notice>
      </CardBody>
    </Card>
  );
}

function ConfusionCard({ name, m }: { name: string; m: { confusion_matrix: number[][]; confusion_matrix_labels: string[] } }) {
  const cm = m.confusion_matrix ?? [];
  const labels = m.confusion_matrix_labels ?? [];
  const max = Math.max(...cm.flat(), 1);
  return (
    <Card>
      <CardHeader>
        <CardTitle>Confusion matrix — {name}</CardTitle>
        <p className="mt-0.5 text-[12.5px] text-[var(--color-ink-tertiary)]">Rows: true class · Columns: predicted</p>
      </CardHeader>
      <CardBody>
        <div className="overflow-x-auto">
          <table className="border-collapse">
            <thead>
              <tr>
                <th />
                {labels.map((l) => (
                  <th key={l} className="px-2 pb-2 text-[11.5px] font-medium capitalize text-[var(--color-ink-tertiary)]">{l}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {cm.map((row, i) => (
                <tr key={i}>
                  <th className="pr-3 text-right text-[11.5px] font-medium capitalize text-[var(--color-ink-tertiary)]">
                    {labels[i]}
                  </th>
                  {row.map((v, j) => {
                    const a = v / max;
                    const diag = i === j;
                    return (
                      <td key={j} className="p-1">
                        <div
                          className="flex size-14 items-center justify-center rounded-[var(--radius-md)] text-[15px] font-semibold tabular"
                          style={{
                            background: diag
                              ? `rgba(6,118,71,${0.10 + 0.55 * a})`
                              : v > 0 ? `rgba(180,35,24,${0.10 + 0.5 * a})` : "var(--color-surface-sunken)",
                            color: a > 0.5 ? "#fff" : "var(--color-ink)",
                          }}
                        >
                          {v}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardBody>
    </Card>
  );
}
