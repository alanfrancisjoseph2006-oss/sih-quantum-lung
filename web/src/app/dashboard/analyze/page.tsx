"use client";

/**
 * Radiology workspace — the flagship interface.
 *
 * Three-column layout on large screens: image viewer | AI visualisation |
 * clinical findings. Measured image parameters are kept visually separate from
 * model predictions throughout, because conflating them is how a decision-support
 * tool starts to look like a diagnosis.
 */

import * as React from "react";
import Link from "next/link";
import {
  AlertTriangle, Ban, CheckCircle2, Cpu, Crosshair, FileText, ImageIcon, Layers, Minus, Plus, RotateCcw, Ruler, ScanLine,
  Sparkles, Upload, XCircle, Zap,
} from "lucide-react";
import { PageContainer } from "@/components/PageContainer";
import { DisclaimerBar } from "@/components/DisclaimerBar";
import {
  Badge, Button, Card, CardBody, CardHeader, CardTitle, KeyValue,
  Notice, ProgressBar, SectionHead, Tooltip,
} from "@/components/ui";
import {
  ApiError, analyzeScan,
} from "@/lib/api";
import type { AnalysisResult, LungClass } from "@/lib/types";
import { classLabel, cn, num, pct } from "@/lib/utils";

type Stage = { key: string; label: string };
const STAGES: Stage[] = [
  { key: "upload", label: "Uploading" },
  { key: "validate", label: "Validating image" },
  { key: "preprocess", label: "Preprocessing" },
  { key: "classical", label: "Classical AI analysis" },
  { key: "quantum", label: "Quantum analysis" },
  { key: "fusion", label: "Hybrid fusion" },
  { key: "xai", label: "Generating explainability" },
  { key: "report", label: "Preparing report" },
];

type ViewMode = "original" | "heatmap" | "overlay" | "segmentation";

export default function AnalyzePage() {
  const [file, setFile] = React.useState<File | null>(null);
  const [preview, setPreview] = React.useState<string | null>(null);
  const [patientId, setPatientId] = React.useState("");
  const [result, setResult] = React.useState<AnalysisResult & { analysis_id?: string } | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [stage, setStage] = React.useState(-1);
  const [error, setError] = React.useState<string | null>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const onPick = (f: File | null) => {
    if (!f) return;
    setFile(f); setResult(null); setError(null);
    setPreview(URL.createObjectURL(f));
  };

  async function run() {
    if (!file) return;
    setBusy(true); setError(null); setResult(null); setStage(0);

    /* The stage ticker is a UI affordance, not a real progress feed: the backend
       returns one response. It therefore advances to the second-to-last stage and
       waits — it never shows completion before the result actually arrives. */
    const ticker = setInterval(
      () => setStage((s) => Math.min(s + 1, STAGES.length - 2)), 260);

    try {
      const r = await analyzeScan(file, {
        patientExternalId: patientId.trim() || undefined,
        persist: true,
      });
      clearInterval(ticker);
      setStage(STAGES.length);
      setResult(r);
    } catch (e) {
      clearInterval(ticker);
      setStage(-1);
      setError(e instanceof ApiError ? e.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <PageContainer wide>
      <SectionHead
        title="CT scan analysis"
        description="Upload a chest CT slice to run the hybrid quantum-classical pipeline."
        actions={result?.analysis_id && (
          <Link href={`/dashboard/reports?analysis=${result.analysis_id}`}>
            <Button variant="primary" icon={FileText}>Open report</Button>
          </Link>
        )}
      />

      <div className="mt-5 space-y-5">
        <DisclaimerBar />

        {!result && (
          <UploadPanel
            file={file} preview={preview} busy={busy} stage={stage} error={error}
            patientId={patientId} setPatientId={setPatientId}
            inputRef={inputRef} onPick={onPick} onRun={run}
          />
        )}

        {result && <InputGateBanner result={result} />}

        {result && (
          <Workspace
            result={result}
            onReset={() => { setResult(null); setFile(null); setPreview(null); setStage(-1); }}
          />
        )}
      </div>
    </PageContainer>
  );
}


/**
 * Input gate banner.
 *
 * Sits above the workspace because a resolution or plausibility failure changes
 * how everything below it should be read. Measured behaviour: images downscaled
 * below ~400 px flipped to "malignant" on 12/12 ground-truth normal scans, and
 * the classifier confidently labels blank images because it has no reject class.
 */
function InputGateBanner({ result }: { result: AnalysisResult }) {
  const iv = result.input_validation;
  if (!iv || iv.overall_status === "ok") return null;
  const unreliable = iv.overall_status === "unreliable";

  return (
    <Notice
      tone={unreliable ? "danger" : "warning"}
      title={unreliable
        ? "This input is outside the range the model was validated on"
        : "Input quality below the validated range — treat this result with caution"}
    >
      <div className="space-y-2">
        <ul className="space-y-1">
          {iv.reasons.map((r, i) => <li key={i}>{"·"} {r}</li>)}
        </ul>
        <p className="font-medium">{iv.guidance}</p>
        {iv.resolution.status !== "ok" && (
          <p className="text-[12px] opacity-90">{iv.resolution.detail}</p>
        )}
        {iv.plausibility.status !== "ok" && (
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-[12px] opacity-90">
            {iv.plausibility.checks.filter((c) => !c.passed).map((c) => (
              <span key={c.id}>{"✗"} {c.label} {"—"} {c.detail}</span>
            ))}
          </div>
        )}
      </div>
    </Notice>
  );
}

/* ------------------------------------------------------------------ upload */

function UploadPanel({
  file, preview, busy, stage, error, patientId, setPatientId, inputRef, onPick, onRun,
}: {
  file: File | null; preview: string | null; busy: boolean; stage: number;
  error: string | null; patientId: string; setPatientId: (v: string) => void;
  inputRef: React.RefObject<HTMLInputElement | null>;
  onPick: (f: File | null) => void; onRun: () => void;
}) {
  const [drag, setDrag] = React.useState(false);

  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Upload chest CT</CardTitle>
          <p className="mt-0.5 text-[12.5px] text-[var(--color-ink-tertiary)]">
            PNG, JPG, BMP, TIFF or DICOM (.dcm). Maximum 40 MB. Processed locally on this server.
          </p>
        </CardHeader>
        <CardBody>
          <div
            role="button"
            tabIndex={0}
            aria-label="Choose a CT image to analyse"
            onClick={() => inputRef.current?.click()}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); onPick(e.dataTransfer.files?.[0] ?? null); }}
            className={cn(
              "flex cursor-pointer flex-col items-center justify-center rounded-[var(--radius-lg)]",
              "border-2 border-dashed px-6 py-12 text-center transition-colors",
              drag ? "border-[var(--color-primary)] bg-[var(--color-primary-soft)]"
                   : "border-[var(--color-line-strong)] hover:border-[var(--color-primary)] hover:bg-[var(--color-surface-hover)]",
            )}
          >
            {preview ? (
              <>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={preview} alt="Selected scan preview"
                     className="mb-3 max-h-52 rounded-[var(--radius-md)] border border-[var(--color-line)] object-contain" />
                <p className="text-[13.5px] font-medium">{file?.name}</p>
                <p className="text-[12.5px] text-[var(--color-ink-tertiary)]">
                  {file ? (file.size / 1024 / 1024).toFixed(2) : "0"} MB · click to replace
                </p>
              </>
            ) : (
              <>
                <div className="mb-3 flex size-11 items-center justify-center rounded-full bg-[var(--color-surface-sunken)]">
                  <Upload className="size-5 text-[var(--color-ink-tertiary)]" aria-hidden />
                </div>
                <p className="text-[14px] font-medium">Drop a CT image here, or click to browse</p>
                <p className="mt-1 text-[12.5px] text-[var(--color-ink-tertiary)]">
                  DICOM files are read with rescale slope/intercept and lung windowing
                </p>
              </>
            )}
            <input
              ref={inputRef} type="file" className="sr-only"
              accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff,.dcm,.dicom,.ima"
              onChange={(e) => onPick(e.target.files?.[0] ?? null)}
            />
          </div>

          <div className="mt-4 flex flex-wrap items-end gap-3">
            <div className="min-w-[220px] flex-1">
              <label htmlFor="pid" className="mb-1 block text-[12.5px] font-medium text-[var(--color-ink-secondary)]">
                Patient identifier <span className="font-normal text-[var(--color-ink-tertiary)]">(optional)</span>
              </label>
              <input
                id="pid" value={patientId} onChange={(e) => setPatientId(e.target.value)}
                placeholder="e.g. DEMO-0015"
                className="h-9 w-full rounded-[var(--radius-md)] border border-[var(--color-line-strong)] bg-[var(--color-surface)] px-3 text-[13.5px]"
              />
            </div>
            <Button variant="primary" size="lg" icon={ScanLine} loading={busy}
                    disabled={!file} onClick={onRun}>
              {busy ? "Analysing…" : "Run analysis"}
            </Button>
          </div>

          <p className="mt-2 text-[12px] text-[var(--color-ink-tertiary)]">
            Leave the identifier blank to create an anonymous record. Only an opaque
            identifier is stored — no names or dates of birth.
          </p>

          {error && (
            <Notice tone="danger" title="Analysis failed" className="mt-4">{error}</Notice>
          )}
        </CardBody>
      </Card>

      <Card>
        <CardHeader><CardTitle>Pipeline</CardTitle></CardHeader>
        <CardBody className="space-y-2.5">
          {STAGES.map((s, i) => {
            const done = stage > i;
            const active = stage === i;
            return (
              <div key={s.key} className="flex items-center gap-2.5">
                <div className={cn(
                  "flex size-5 shrink-0 items-center justify-center rounded-full border text-[10px] font-semibold",
                  done ? "border-[var(--color-low)] bg-[var(--color-low)] text-white"
                    : active ? "border-[var(--color-primary)] bg-[var(--color-primary-soft)] text-[var(--color-primary)]"
                    : "border-[var(--color-line-strong)] text-[var(--color-ink-disabled)]",
                )}>
                  {done ? <CheckCircle2 className="size-3" aria-hidden /> : i + 1}
                </div>
                <span className={cn("text-[13px]",
                  done ? "text-[var(--color-ink-secondary)]"
                    : active ? "font-medium text-[var(--color-primary)]"
                    : "text-[var(--color-ink-disabled)]")}>
                  {s.label}
                </span>
                {active && <span className="ml-auto size-1.5 animate-pulse rounded-full bg-[var(--color-primary)]" aria-hidden />}
              </div>
            );
          })}
          <p className="pt-2 text-[11.5px] leading-relaxed text-[var(--color-ink-tertiary)]">
            Stages reflect the real pipeline order. Progress is indicative — the
            server returns one response, and completion is only shown once the
            result has actually arrived.
          </p>
        </CardBody>
      </Card>
    </div>
  );
}

/* --------------------------------------------------------------- workspace */

function Workspace({ result, onReset }: {
  result: AnalysisResult & { analysis_id?: string }; onReset: () => void;
}) {
  const [view, setView] = React.useState<ViewMode>("overlay");
  const [zoom, setZoom] = React.useState(1);
  const [brightness, setBrightness] = React.useState(100);
  const seg = result.segmentation;
  const ex = result.explainability;

  const segAvailable = seg?.available && (seg.images?.overlay ?? null) !== null;

  const images: Record<ViewMode, string | undefined> = {
    original: ex.images.original,
    heatmap: ex.images.heatmap,
    overlay: ex.images.overlay,
    segmentation: seg?.images?.overlay,
  };

  const modes: { key: ViewMode; label: string; icon: React.ElementType; disabled?: boolean; hint?: string }[] = [
    { key: "original", label: "Original", icon: ImageIcon },
    { key: "heatmap", label: "Attention map", icon: Sparkles },
    { key: "overlay", label: "Attention overlay", icon: Layers },
    {
      key: "segmentation", label: "Segmentation", icon: Crosshair,
      disabled: !segAvailable,
      hint: segAvailable ? undefined : "No nodule region was segmented on this image",
    },
  ];

  return (
    <div className="grid grid-cols-1 gap-5 xl:grid-cols-12">
      {/* -------- Left: viewer -------- */}
      <div className="space-y-5 xl:col-span-5">
        <Card>
          <CardHeader className="flex flex-wrap items-center justify-between gap-2">
            <CardTitle>Image viewer</CardTitle>
            <div className="flex items-center gap-1">
              <Tooltip label="Zoom out">
                <Button variant="ghost" size="sm" aria-label="Zoom out"
                        onClick={() => setZoom((z) => Math.max(1, +(z - 0.25).toFixed(2)))}>
                  <Minus className="size-4" aria-hidden />
                </Button>
              </Tooltip>
              <span className="w-11 text-center text-[12px] tabular text-[var(--color-ink-tertiary)]">
                {(zoom * 100).toFixed(0)}%
              </span>
              <Tooltip label="Zoom in">
                <Button variant="ghost" size="sm" aria-label="Zoom in"
                        onClick={() => setZoom((z) => Math.min(4, +(z + 0.25).toFixed(2)))}>
                  <Plus className="size-4" aria-hidden />
                </Button>
              </Tooltip>
              <Tooltip label="Reset view">
                <Button variant="ghost" size="sm" aria-label="Reset view"
                        onClick={() => { setZoom(1); setBrightness(100); }}>
                  <RotateCcw className="size-4" aria-hidden />
                </Button>
              </Tooltip>
            </div>
          </CardHeader>

          <div className="relative aspect-square overflow-hidden bg-black">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={images[view] ?? images.original}
              alt={`CT scan — ${view} view`}
              className="size-full object-contain transition-transform duration-200"
              style={{ transform: `scale(${zoom})`, filter: `brightness(${brightness}%)` }}
            />
            <div className="absolute left-3 top-3 flex flex-wrap gap-1.5">
              <Badge tone="neutral" className="bg-black/60 text-white backdrop-blur border-white/20">
                {modes.find((m) => m.key === view)?.label}
              </Badge>
              {result.input.is_dicom && (
                <Badge tone="neutral" className="bg-black/60 text-white backdrop-blur border-white/20">
                  DICOM
                </Badge>
              )}
            </div>
          </div>

          <CardBody className="space-y-3">
            <div className="flex flex-wrap gap-1.5">
              {modes.map((m) => (
                <button
                  key={m.key}
                  disabled={m.disabled}
                  title={m.hint}
                  onClick={() => setView(m.key)}
                  className={cn(
                    "inline-flex items-center gap-1.5 rounded-[var(--radius-md)] border px-2.5 py-1.5",
                    "text-[12.5px] font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-40",
                    view === m.key
                      ? "border-[var(--color-primary-border)] bg-[var(--color-primary-soft)] text-[var(--color-primary)]"
                      : "border-[var(--color-line)] text-[var(--color-ink-secondary)] hover:bg-[var(--color-surface-sunken)]",
                  )}
                >
                  <m.icon className="size-3.5" aria-hidden />
                  {m.label}
                </button>
              ))}
            </div>

            <div>
              <label htmlFor="bright" className="mb-1 flex items-center justify-between text-[12.5px] text-[var(--color-ink-secondary)]">
                <span>Brightness</span>
                <span className="tabular text-[var(--color-ink-tertiary)]">{brightness}%</span>
              </label>
              <input id="bright" type="range" min={40} max={180} value={brightness}
                     onChange={(e) => setBrightness(+e.target.value)}
                     className="w-full accent-[var(--color-primary)]" />
            </div>

            <Button variant="secondary" size="sm" icon={Upload} onClick={onReset} className="w-full">
              Analyse another scan
            </Button>
          </CardBody>
        </Card>

        {/* Image quality — measured, not predicted */}
        <QualityCard result={result} />
      </div>

      {/* -------- Middle: AI visualisation -------- */}
      <div className="space-y-5 xl:col-span-3">
        <VerdictCard result={result} />
        <BranchesCard result={result} />
      </div>

      {/* -------- Right: clinical findings -------- */}
      <div className="space-y-5 xl:col-span-4">
        <SegmentationCard result={result} />
        <AttentionCard result={result} />
        <UnsupportedCard result={result} />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------- sub-panels */

function VerdictCard({ result }: { result: AnalysisResult }) {
  const fp = result.final_prediction;
  const tone = fp.risk_band === "High" ? "high" : fp.risk_band === "Intermediate" ? "moderate" : "low";
  const Icon = fp.risk_band === "High" ? XCircle : fp.risk_band === "Intermediate" ? AlertTriangle : CheckCircle2;

  return (
    <Card>
      <CardHeader><CardTitle>AI risk assessment</CardTitle></CardHeader>
      <CardBody className="space-y-4">
        <div className={cn(
          "rounded-[var(--radius-lg)] border p-4",
          tone === "high" ? "border-[var(--color-high-border)] bg-[var(--color-high-soft)]"
            : tone === "moderate" ? "border-[var(--color-moderate-border)] bg-[var(--color-moderate-soft)]"
            : "border-[var(--color-low-border)] bg-[var(--color-low-soft)]",
        )}>
          <div className="flex items-start gap-2.5">
            <Icon className={cn("mt-0.5 size-5 shrink-0",
              tone === "high" ? "text-[var(--color-high)]"
                : tone === "moderate" ? "text-[var(--color-moderate)]" : "text-[var(--color-low)]")} aria-hidden />
            <div className="min-w-0">
              <div className="text-[15px] font-semibold leading-tight text-[var(--color-ink)]">
                {fp.display}
              </div>
              <div className="mt-0.5 text-[12.5px] text-[var(--color-ink-secondary)]">
                {fp.risk_band} risk band · confidence {pct(fp.confidence)}
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-2">
          {(Object.entries(fp.probabilities) as [LungClass, number][]).map(([cls, p]) => (
            <div key={cls}>
              <div className="mb-1 flex items-center justify-between text-[12.5px]">
                <span className="text-[var(--color-ink-secondary)]">{classLabel[cls]}</span>
                <span className="tabular font-medium">{pct(p)}</span>
              </div>
              <ProgressBar value={p} tone={cls === "malignant" ? "high" : cls === "benign" ? "moderate" : "low"} />
            </div>
          ))}
        </div>

        <div className="space-y-2 border-t border-[var(--color-line)] pt-3">
          <KeyValue label="Selected model" value={fp.selected_model} />
          <KeyValue label="Branch agreement" value={pct(fp.branch_agreement, 0)} />
          <KeyValue label="Analysis time" value={`${num(result.runtime_seconds, 2)} s`} />
        </div>

        {fp.reliability_warning && (
          <Notice tone={fp.reliability === "unreliable" ? "danger" : "warning"}
                  title="Result reliability">
            {fp.reliability_warning}
          </Notice>
        )}

        {fp.decision_policy_applied && (
          <Notice tone={fp.decision_policy_changed_outcome ? "warning" : "neutral"}
                  title={fp.decision_policy_changed_outcome ? "Decision policy changed this outcome" : "Cost-sensitive policy applied"}>
            {fp.decision_policy_note}
            {fp.decision_policy_changed_outcome && (
              <> Raw model output was <strong>{fp.raw_model_class}</strong>.</>
            )}
          </Notice>
        )}
      </CardBody>
    </Card>
  );
}

function BranchesCard({ result }: { result: AnalysisResult }) {
  const b = result.branch_results;
  const rows = [
    { label: "Classical deep learning", icon: Cpu, r: b.classical_deep_learning, tone: "primary" as const },
    { label: "Quantum branch", icon: Zap, r: b.quantum, tone: "quantum" as const },
    ...(b.hybrid ? [{ label: "Hybrid fusion", icon: Layers, r: b.hybrid, tone: "info" as const }] : []),
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Model branches</CardTitle>
        <p className="mt-0.5 text-[12.5px] text-[var(--color-ink-tertiary)]">
          Each branch reported independently
        </p>
      </CardHeader>
      <CardBody className="space-y-3">
        {rows.map(({ label, icon: Icon, r, tone }) => (
          <div key={label} className="rounded-[var(--radius-md)] border border-[var(--color-line)] p-3">
            <div className="mb-2 flex items-center justify-between gap-2">
              <span className="flex items-center gap-1.5 text-[12.5px] font-medium text-[var(--color-ink-secondary)]">
                <Icon className="size-3.5" aria-hidden />{label}
              </span>
              <Badge tone={tone}>{r.predicted_class}</Badge>
            </div>
            <div className="flex items-center gap-2">
              <ProgressBar value={r.confidence} tone={tone === "quantum" ? "quantum" : "primary"} className="flex-1" />
              <span className="w-11 text-right text-[12px] tabular">{pct(r.confidence)}</span>
            </div>
          </div>
        ))}

        <div className="rounded-[var(--radius-md)] border border-[var(--color-line)]">
          <div className="border-b border-[var(--color-line)] px-3 py-2 text-[12px] font-medium uppercase tracking-[0.04em] text-[var(--color-ink-tertiary)]">
            Classical ML heads
          </div>
          <div className="divide-y divide-[var(--color-line)]">
            {b.classical_ml.map((m) => (
              <div key={m.model} className="flex items-center justify-between px-3 py-2 text-[12.5px]">
                <span className="text-[var(--color-ink-secondary)]">{m.model}</span>
                <span className="flex items-center gap-2">
                  <span className="text-[var(--color-ink)]">{m.predicted_class}</span>
                  <span className="w-11 text-right tabular text-[var(--color-ink-tertiary)]">{pct(m.confidence)}</span>
                </span>
              </div>
            ))}
          </div>
        </div>
      </CardBody>
    </Card>
  );
}

function SegmentationCard({ result }: { result: AnalysisResult }) {
  const seg = result.segmentation;
  if (!seg?.available) {
    return (
      <Card>
        <CardHeader><CardTitle>Nodule segmentation</CardTitle></CardHeader>
        <CardBody>
          <Notice tone="neutral">{seg?.message ?? "Segmentation unavailable."}</Notice>
        </CardBody>
      </Card>
    );
  }
  const idp = seg.in_domain_performance;
  const it = seg.interpretation;
  const conc = it?.concordance;

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <CardTitle>Nodule segmentation</CardTitle>
        <Badge tone={(seg.n_lesions ?? 0) > 0 ? "info" : "neutral"}>
          {seg.n_lesions ?? 0} region{(seg.n_lesions ?? 0) === 1 ? "" : "s"}
        </Badge>
      </CardHeader>
      <CardBody className="space-y-3">
        <Notice tone="warning">{it?.what_this_is_not}</Notice>

        {conc && (
          <Notice tone={conc.state === "agree" ? "info" : "warning"}
                  title={conc.state === "agree" ? "Models agree" : "Models disagree"}>
            {conc.note}
          </Notice>
        )}

        {(seg.lesions?.length ?? 0) > 0 ? (
          <div className="overflow-hidden rounded-[var(--radius-md)] border border-[var(--color-line)]">
            <table className="w-full text-[12.5px]">
              <thead>
                <tr className="bg-[var(--color-surface-sunken)] text-[11px] uppercase tracking-[0.04em] text-[var(--color-ink-tertiary)]">
                  <th className="px-2.5 py-2 text-left font-medium">Location</th>
                  <th className="px-2.5 py-2 text-right font-medium">Diameter</th>
                  <th className="px-2.5 py-2 text-right font-medium">% lung</th>
                  <th className="px-2.5 py-2 text-right font-medium">Conf.</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--color-line)]">
                {seg.lesions!.map((l, i) => (
                  <tr key={i}>
                    <td className="px-2.5 py-2">{l.location}</td>
                    <td className="px-2.5 py-2 text-right tabular">{l.equivalent_diameter_px} px</td>
                    <td className="px-2.5 py-2 text-right tabular">
                      {l.area_fraction_of_lung != null ? `${(100 * l.area_fraction_of_lung).toFixed(3)}%` : "—"}
                    </td>
                    <td className="px-2.5 py-2 text-right tabular">{pct(l.mean_confidence)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <Notice tone="warning">
            No nodule-like region found in the detected lung field. This is <strong>not</strong> a
            clearance — on held-out patients this model detected only{" "}
            {idp?.detection_rate != null ? pct(idp.detection_rate, 0) : "about half"} of annotated nodules.
          </Notice>
        )}

        <div className="space-y-0 border-t border-[var(--color-line)] pt-2">
          <KeyValue label="Detection rate" value={idp?.detection_rate != null ? pct(idp.detection_rate) : "—"} />
          <KeyValue label="Dice when detected" value={num(idp?.dice_when_detected)} />
          <KeyValue label="Centre error" value={idp?.median_centroid_error_mm != null ? `~${num(idp.median_centroid_error_mm, 2)} mm` : "—"} />
          <KeyValue label="Trained on" value={<span className="font-normal">LIDC-IDRI consensus contours</span>} mono={false} />
        </div>

        {it?.sensitivity_note && <Notice tone="warning">{it.sensitivity_note}</Notice>}
        {it?.domain_caveat && <Notice tone="neutral">{it.domain_caveat}</Notice>}
      </CardBody>
    </Card>
  );
}

function AttentionCard({ result }: { result: AnalysisResult }) {
  const ex = result.explainability;
  const reg = ex.region;
  return (
    <Card>
      <CardHeader>
        <CardTitle>Model attention</CardTitle>
        <p className="mt-0.5 text-[12.5px] text-[var(--color-ink-tertiary)]">{ex.method}</p>
      </CardHeader>
      <CardBody className="space-y-3">
        <Notice tone="warning">{ex.caveat}</Notice>
        {reg.available ? (
          <div className="space-y-0">
            <KeyValue label="Region" value={<span className="font-normal">{reg.location_description}</span>} mono={false} />
            <KeyValue label="Attended area" value={pct(reg.attended_area_fraction)} />
            <KeyValue label="Attention blobs" value={reg.n_attention_blobs ?? "—"} />
          </div>
        ) : (
          <p className="text-[13px] text-[var(--color-ink-tertiary)]">
            No focal attention region resolved on this image.
          </p>
        )}
        <Notice tone="neutral">
          Lobe-level localisation is not attempted — the training data carries no lobe
          labels, so any lobe claim would be guesswork.
        </Notice>
      </CardBody>
    </Card>
  );
}

function QualityCard({ result }: { result: AnalysisResult }) {
  const mp = result.measured_parameters;
  const inp = result.input;
  const iv = result.input_validation;

  const checks = [
    { ok: true, label: "Format valid", detail: inp.source_format },
    { ok: iv ? iv.resolution.status === "ok" : true,
      label: "Resolution within validated range",
      detail: `${inp.original_size?.[0]} × ${inp.original_size?.[1]}` +
              (iv ? ` (min ${iv.resolution.min_validated_px})` : "") },
    { ok: mp.lung_field_detected, label: "Lung region detected",
      detail: mp.lung_field_detected ? `${pct(mp.lung_field_area_fraction)} of image` : "not detected" },
    { ok: mp.sharpness_laplacian_var > 50, label: "Sharpness adequate",
      detail: `Laplacian var ${num(mp.sharpness_laplacian_var, 0)}` },
  ];
  const anyWarn = checks.some((c) => !c.ok);

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <CardTitle>Image quality</CardTitle>
        <Badge tone="neutral" icon={Ruler}>Measured</Badge>
      </CardHeader>
      <CardBody className="space-y-3">
        <ul className="space-y-2">
          {checks.map((c) => (
            <li key={c.label} className="flex items-start gap-2 text-[12.5px]">
              {c.ok
                ? <CheckCircle2 className="mt-0.5 size-3.5 shrink-0 text-[var(--color-low)]" aria-hidden />
                : <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-[var(--color-moderate)]" aria-hidden />}
              <span className="flex-1 text-[var(--color-ink-secondary)]">{c.label}</span>
              <span className="tabular text-[var(--color-ink-tertiary)]">{c.detail}</span>
            </li>
          ))}
        </ul>

        {anyWarn && (
          <Notice tone="warning">Model confidence may be affected by image quality.</Notice>
        )}

        <div className="space-y-0 border-t border-[var(--color-line)] pt-2">
          <KeyValue label="Windowing" value={<span className="font-normal">{inp.windowing_applied}</span>} mono={false} />
          <KeyValue label="Hounsfield units" value={inp.hu_available ? "available (DICOM)" : "not available"} />
          <KeyValue label="Intensity mean ± sd" value={`${num(mp.intensity_mean, 1)} ± ${num(mp.intensity_std, 1)}`} />
          <KeyValue label="Contrast (RMS)" value={num(mp.contrast_rms)} />
        </div>

        <p className="text-[11.5px] leading-relaxed text-[var(--color-ink-tertiary)]">
          {result.measured_parameters_note} These are automated technical checks, not a
          diagnostic image-quality certification.
        </p>

        {inp.notes?.map((n, i) => <Notice key={i} tone="info">{n}</Notice>)}
      </CardBody>
    </Card>
  );
}

function UnsupportedCard({ result }: { result: AnalysisResult }) {
  /* `not_supported` already contains clinical_staging, so de-duplicate by label
     rather than listing it twice (which also produced a duplicate React key). */
  const raw = [
    { label: "Clinical staging", reason: result.staging.message },
    ...Object.entries(result.not_supported ?? {}).map(([k, v]) => ({
      label: k.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase()), reason: v,
    })),
  ];
  const seen = new Set<string>();
  const items = raw.filter((i) => {
    const key = i.label.toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  if (!items.length) return null;

  return (
    <Card>
      <CardHeader className="flex items-center justify-between">
        <CardTitle>Not provided</CardTitle>
        <Badge tone="neutral" icon={Ban}>By design</Badge>
      </CardHeader>
      <CardBody className="space-y-2.5">
        {items.map((i) => (
          <div key={i.label} className="rounded-[var(--radius-md)] bg-[var(--color-surface-sunken)] p-3">
            <div className="text-[12.5px] font-medium text-[var(--color-ink)]">{i.label}</div>
            <p className="mt-0.5 text-[12px] leading-relaxed text-[var(--color-ink-tertiary)]">{i.reason}</p>
          </div>
        ))}
      </CardBody>
    </Card>
  );
}
