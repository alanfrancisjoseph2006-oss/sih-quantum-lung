"use client";

import * as React from "react";
import { Cpu, Layers, Radio, Sigma } from "lucide-react";
import { PageContainer } from "@/components/PageContainer";
import { Card, CardBody, CardHeader, CardTitle, ErrorState, KeyValue, Metric, Notice, SectionHead, Skeleton } from "@/components/ui";
import { getLidcExperiment, getLocalizationValidation, getMeta } from "@/lib/api";
import type { LidcExperiment, LocalizationValidation, PlatformMeta, QuantumCircuitInfo } from "@/lib/types";
import { num } from "@/lib/utils";

export default function TechnologyPage() {
  const [meta, setMeta] = React.useState<PlatformMeta | null>(null);
  const [lidc, setLidc] = React.useState<LidcExperiment | null>(null);
  const [loc, setLoc] = React.useState<LocalizationValidation | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  const load = React.useCallback(async () => {
    setError(null);
    try {
      const m = await getMeta(); setMeta(m);
      getLidcExperiment().then(setLidc).catch(() => {});
      getLocalizationValidation().then(setLoc).catch(() => {});
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load");
    }
  }, []);
  React.useEffect(() => {
    // Deferred a tick so the state updates inside `load` do not run
    // synchronously inside the effect body (react-hooks/set-state-in-effect).
    const t = setTimeout(() => { void load(); }, 0);
    return () => clearTimeout(t);
  }, [load]);

  if (error) return <PageContainer><ErrorState message={error} onRetry={load} /></PageContainer>;
  if (!meta) {
    return <PageContainer><SectionHead title="Technology" />
      <Card className="mt-5"><CardBody><Skeleton className="h-56 w-full" /></CardBody></Card></PageContainer>;
  }

  const c: Partial<QuantumCircuitInfo> = meta.quantum?.circuit ?? {};
  const nq = c.n_qubits ?? meta.engine?.n_qubits ?? 4;
  const nl = c.n_variational_layers ?? meta.quantum?.variational_layers ?? 3;
  const locKey = loc?.results ? Object.keys(loc.results)[0] : null;
  const lk = locKey && loc ? loc.results[locKey] : null;

  return (
    <PageContainer>
      <SectionHead
        eyebrow="Hybrid quantum-classical architecture"
        title="Technology"
        description="How the pipeline works, what the quantum branch actually contributes, and what has been measured rather than claimed."
      />

      <div className="mt-5 space-y-5">
        {/* Pipeline diagram */}
        <Card>
          <CardHeader><CardTitle>Processing pipeline</CardTitle></CardHeader>
          <CardBody>
            <PipelineDiagram nQubits={nq} />
          </CardBody>
        </Card>

        {/* Circuit stats */}
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <Card><Metric label="Qubits deployed" value={nq} icon={Radio} tone="primary" /></Card>
          <Card><Metric label="Variational layers" value={nl} icon={Layers} /></Card>
          <Card><Metric label="Circuit depth" value={c.circuit_depth ?? "—"} icon={Sigma} /></Card>
          <Card><Metric label="Total gates" value={c.gate_count ?? "—"} icon={Cpu} /></Card>
        </div>

        <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
          <Card>
            <CardHeader><CardTitle>Circuit specification</CardTitle></CardHeader>
            <CardBody className="space-y-0">
              <KeyValue label="Encoding" value={<span className="font-normal">{c.encoding ?? "Angle (RY)"}</span>} mono={false} />
              <KeyValue label="Entanglement" value={<span className="font-normal">{c.entanglement ?? "CNOT ring"}</span>} mono={false} />
              <KeyValue label="Measurement" value={<span className="font-normal">{c.measurement ?? "Pauli-Z"}</span>} mono={false} />
              <KeyValue label="Backend" value={c.backend ?? "default.qubit"} />
              <KeyValue label="Differentiation" value={<span className="font-normal">{c.differentiation ?? "backprop"}</span>} mono={false} />
              <KeyValue label="Trainable circuit params" value={c.trainable_circuit_params ?? "—"} />
              <KeyValue label="Qubit sweep" value={(meta.quantum?.qubit_sweep ?? []).join(" · ")} />
              <KeyValue label="Gate breakdown" value={
                <span className="font-normal">
                  {c.gate_types ? Object.entries(c.gate_types).map(([k, v]) => `${k}×${v}`).join(", ") : "—"}
                </span>} mono={false} />
            </CardBody>
          </Card>

          <Card>
            <CardHeader><CardTitle>Design choices</CardTitle></CardHeader>
            <CardBody>
              <ul className="space-y-2.5 text-[13px] leading-relaxed text-[var(--color-ink-secondary)]">
                <li><strong className="text-[var(--color-ink)]">Simulator, not hardware.</strong> Statevector
                  simulation on <code className="font-mono text-[12px]">{c.backend ?? "default.qubit"}</code>. No
                  real QPU was used, so no hardware-noise claims are made.</li>
                <li><strong className="text-[var(--color-ink)]">Few qubits ({(meta.quantum?.qubit_sweep ?? [4,6,8]).join("/")}).</strong>{" "}
                  Swept experimentally; {nq} performed best on validation. More qubits overfit at this data scale.</li>
                <li><strong className="text-[var(--color-ink)]">Shallow depth ({nl} layers).</strong> Deeper ansätze
                  hit barren plateaus and train poorly on 735 images.</li>
                <li><strong className="text-[var(--color-ink)]">PCA to 32, then a learned projection.</strong> The
                  projection is trainable, so the network chooses which directions deserve the limited qubit budget.</li>
              </ul>
            </CardBody>
          </Card>
        </div>

        {/* Circuit visual */}
        <Card>
          <CardHeader><CardTitle>Variational circuit</CardTitle></CardHeader>
          <CardBody><CircuitDiagram nQubits={nq} nLayers={nl} /></CardBody>
        </Card>

        {/* Honest findings */}
        <Card>
          <CardHeader>
            <CardTitle>Measured findings</CardTitle>
            <p className="mt-0.5 text-[12.5px] text-[var(--color-ink-tertiary)]">
              Including the results that did not go our way
            </p>
          </CardHeader>
          <CardBody className="grid grid-cols-1 gap-3 md:grid-cols-3">
            <Notice tone="warning" title="No quantum advantage demonstrated">
              Paired bootstrap on the held-out test split gave p ≈ 0.69–0.71. The quantum branch
              performs comparably to classical models, not better. We report it rather than hide it.
            </Notice>
            <Notice tone="warning" title="Attention maps do not localise">
              {lk ? <>Scored against {lk.n_images} real lesion boxes, Grad-CAM++ pointing-game hit rate
                was {num(lk.pointing_game_hit_rate)} — no better than chance. That is why attention is
                never labelled a tumour boundary.</>
                : "Grad-CAM localisation was validated against real lesion annotations and found to be at chance."}
            </Notice>
            <Notice tone="warning" title="More data made it worse">
              {lidc ? <>Adding {lidc.acquired?.slices_extracted} LIDC slices took benign training images
                from {lidc.training_set_change?.benign?.before} to {lidc.training_set_change?.benign?.after},
                yet macro-F1 <strong>fell</strong> {num(Math.abs(lidc.results_on_unchanged_iqoth_test?.Hybrid_meta?.delta ?? 0))}{" "}
                (p = {num(lidc.results_on_unchanged_iqoth_test?.Hybrid_meta?.p_value ?? 0)}). Domain shift beat
                the extra examples, so it was not deployed.</>
                : "An external-data experiment degraded performance and was not deployed."}
            </Notice>
          </CardBody>
        </Card>

        <Notice tone="neutral" title="Scientific position">
          The quantum branch is a real, trainable, working component whose performance on this dataset
          is <strong>comparable</strong> to classical models — not superior. Reporting that honestly is
          the point: an advantage that has not been measured has not been demonstrated.
        </Notice>
      </div>
    </PageContainer>
  );
}

const TONE_CLS = {
  n: "border-[var(--color-line-strong)] bg-[var(--color-surface)]",
  c: "border-[var(--color-primary-border)] bg-[var(--color-primary-soft)]",
  q: "border-[var(--color-quantum-border)] bg-[var(--color-quantum-soft)]",
  g: "border-[var(--color-low-border)] bg-[var(--color-low-soft)]",
} as const;

function StageBox({ label, sub, tone }: {
  label: string; sub: string; tone: keyof typeof TONE_CLS;
}) {
  return (
    <div className={`rounded-[var(--radius-md)] border px-3 py-2.5 text-center ${TONE_CLS[tone]}`}>
      <div className="text-[12.5px] font-semibold text-[var(--color-ink)]">{label}</div>
      <div className="mt-0.5 text-[11px] text-[var(--color-ink-tertiary)]">{sub}</div>
    </div>
  );
}

function Arrow() {
  return (
    <div className="flex items-center justify-center text-[var(--color-ink-disabled)]" aria-hidden>
      <svg width="20" height="12" viewBox="0 0 20 12"><path d="M0 6h14M14 6l-4-4M14 6l-4 4"
        stroke="currentColor" strokeWidth="1.4" fill="none" /></svg>
    </div>
  );
}

function PipelineDiagram({ nQubits }: { nQubits: number }) {
  const box = (label: string, sub: string, tone: keyof typeof TONE_CLS) =>
    <StageBox label={label} sub={sub} tone={tone} />;

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 items-center gap-2 sm:grid-cols-[1fr_auto_1fr_auto_1fr]">
        {box("CT image", "PNG · JPG · DICOM", "n")}
        <Arrow />
        {box("Preprocessing", "denoise · CLAHE · letterbox", "n")}
        <Arrow />
        {box("EfficientNet-B3", "256-d embedding", "c")}
      </div>
      <div className="flex justify-center text-[var(--color-ink-disabled)]" aria-hidden>
        <svg width="12" height="20" viewBox="0 0 12 20"><path d="M6 0v14M6 14l-4-4M6 14l4-4"
          stroke="currentColor" strokeWidth="1.4" fill="none" /></svg>
      </div>
      <div className="grid grid-cols-1 items-center gap-2 sm:grid-cols-3">
        {box("Classical heads", "RF · SVM · XGBoost · LogReg", "c")}
        {box(`Quantum circuit (${nQubits}q)`, "angle encoding · VQC · ⟨Z⟩", "q")}
        {box("Grad-CAM++ · U-Net", "explainability · segmentation", "n")}
      </div>
      <div className="flex justify-center text-[var(--color-ink-disabled)]" aria-hidden>
        <svg width="12" height="20" viewBox="0 0 12 20"><path d="M6 0v14M6 14l-4-4M6 14l4-4"
          stroke="currentColor" strokeWidth="1.4" fill="none" /></svg>
      </div>
      <div className="grid grid-cols-1 items-center gap-2 sm:grid-cols-[1fr_auto_1fr_auto_1fr]">
        {box("Hybrid fusion", "4 strategies compared", "c")}
        <Arrow />
        {box("Adaptive router", "validation-selected", "c")}
        <Arrow />
        {box("Clinical decision support", "risk · region · limits", "g")}
      </div>
    </div>
  );
}

function CircuitDiagram({ nQubits, nLayers }: { nQubits: number; nLayers: number }) {
  const W = 860, top = 34, gap = 40;
  const H = top + nQubits * gap + 26;
  const x0 = 96, xEnd = W - 70;
  const gate = (x: number, y: number, label: string, color: string) => (
    <g key={`${x}-${y}-${label}`}>
      <rect x={x - 25} y={y - 12} width={50} height={24} rx={4} fill="#fff" stroke={color} strokeWidth={1.2} />
      <text x={x} y={y + 4} textAnchor="middle" fontSize={10} fill={color} fontFamily="ui-monospace, monospace">{label}</text>
    </g>
  );

  return (
    <div className="overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full min-w-[720px]" role="img"
           aria-label={`Quantum circuit with ${nQubits} qubits and ${nLayers} variational layers`}>
        <text x={x0 + 20} y={16} fontSize={10.5} fontWeight={600} fill="var(--color-primary)">ANGLE ENCODING</text>
        <text x={x0 + 250} y={16} fontSize={10.5} fontWeight={600} fill="var(--color-quantum)">
          {nLayers} × VARIATIONAL LAYER (RY·RZ + CNOT RING)
        </text>
        <text x={xEnd - 25} y={16} fontSize={10.5} fontWeight={600} fill="var(--color-low)" textAnchor="middle">MEASURE</text>

        {Array.from({ length: nQubits }).map((_, w) => {
          const y = top + w * gap;
          const y2 = top + ((w + 1) % nQubits) * gap;
          return (
            <g key={w}>
              <line x1={x0} y1={y} x2={xEnd} y2={y} stroke="var(--color-line-strong)" strokeWidth={1.2} />
              <text x={x0 - 10} y={y + 4} textAnchor="end" fontSize={11}
                    fill="var(--color-ink-tertiary)" fontFamily="ui-monospace, monospace">|0⟩ q{w}</text>
              {gate(x0 + 28, y, "RY(xᵢ)", "var(--color-primary)")}
              {Array.from({ length: Math.min(nLayers, 3) }).map((__, l) => {
                const bx = x0 + 110 + l * 190;
                const cx = bx + 122;
                return (
                  <g key={l}>
                    {gate(bx, y, "RY(θ)", "var(--color-quantum)")}
                    {gate(bx + 56, y, "RZ(φ)", "var(--color-quantum)")}
                    <circle cx={cx} cy={y} r={3.2} fill="var(--color-quantum)" />
                    <line x1={cx} y1={y} x2={cx} y2={y2} stroke="var(--color-quantum)" strokeWidth={1.1} opacity={0.55} />
                    <circle cx={cx} cy={y2} r={6} fill="none" stroke="var(--color-quantum)" strokeWidth={1.1} />
                    <line x1={cx - 6} y1={y2} x2={cx + 6} y2={y2} stroke="var(--color-quantum)" strokeWidth={1.1} />
                  </g>
                );
              })}
              <rect x={xEnd - 46} y={y - 12} width={42} height={24} rx={4}
                    fill="var(--color-low-soft)" stroke="var(--color-low)" strokeWidth={1.1} />
              <text x={xEnd - 25} y={y + 4} textAnchor="middle" fontSize={10.5}
                    fill="var(--color-low)" fontFamily="ui-monospace, monospace">⟨Z⟩</text>
            </g>
          );
        })}
        <text x={x0} y={H - 6} fontSize={10.5} fill="var(--color-ink-tertiary)">
          CNN embedding (256-d) → StandardScaler → PCA (32) → learned projection → tanh·π → the {nQubits} rotation angles above
        </text>
      </svg>
    </div>
  );
}
