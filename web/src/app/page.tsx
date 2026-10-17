import Link from "next/link";
import {
  Activity, ArrowRight, Ban, Brain, CheckCircle2, Eye, FileText, GitBranch, Layers, Lock, Microscope, ShieldCheck, Workflow,
} from "lucide-react";

/**
 * Product landing page.
 *
 * Positions the platform as clinical decision support, not "upload an image and
 * detect cancer". The credibility section deliberately leads with measured
 * negative results — for a clinical audience that is more persuasive than a
 * headline accuracy number.
 */

export const metadata = {
  title: "QuantumLung — AI-Assisted Lung Screening & Radiology Decision Support",
};

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[var(--color-surface)]">
      <SiteHeader />
      <Hero />
      <Capabilities />
      <HowItWorks />
      <Architecture />
      <Explainability />
      <Workflow_ />
      <Security />
      <Credibility />
      <SiteFooter />
    </div>
  );
}

function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-[var(--color-line)] bg-[var(--color-surface)]/85 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-[1200px] items-center gap-3 px-5 sm:px-6">
        <div className="flex items-center gap-2.5">
          <div className="flex size-7 items-center justify-center rounded-[var(--radius-md)] bg-[var(--color-primary)]">
            <Microscope className="size-4 text-white" aria-hidden />
          </div>
          <span className="text-[14.5px] font-semibold tracking-[-0.01em]">QuantumLung</span>
        </div>
        <nav className="ml-6 hidden items-center gap-5 text-[13.5px] text-[var(--color-ink-secondary)] md:flex">
          <a href="#capabilities" className="hover:text-[var(--color-ink)]">Capabilities</a>
          <a href="#technology" className="hover:text-[var(--color-ink)]">Technology</a>
          <a href="#workflow" className="hover:text-[var(--color-ink)]">Workflow</a>
          <a href="#security" className="hover:text-[var(--color-ink)]">Security</a>
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <Link href="/dashboard/technology"
                className="hidden rounded-[var(--radius-md)] border border-[var(--color-line-strong)] px-3 py-1.5 text-[13px] font-medium text-[var(--color-ink-secondary)] hover:bg-[var(--color-surface-sunken)] sm:block">
            Explore technology
          </Link>
          <Link href="/dashboard/analyze"
                className="inline-flex items-center gap-1.5 rounded-[var(--radius-md)] bg-[var(--color-primary)] px-3.5 py-1.5 text-[13px] font-medium text-white hover:bg-[var(--color-primary-hover)]">
            Analyze scan <ArrowRight className="size-3.5" aria-hidden />
          </Link>
        </div>
      </div>
    </header>
  );
}

function Hero() {
  return (
    <section className="relative overflow-hidden border-b border-[var(--color-line)]">
      {/* Abstract CT-inspired backdrop — concentric rings and a scan sweep.
          No stock medical photography. */}
      <div className="pointer-events-none absolute inset-0" aria-hidden>
        <svg className="absolute right-[-10%] top-[-30%] h-[560px] w-[560px] opacity-[0.07]" viewBox="0 0 400 400">
          {[40, 80, 120, 160, 190].map((r) => (
            <circle key={r} cx="200" cy="200" r={r} fill="none" stroke="var(--color-primary)" strokeWidth="1" />
          ))}
          {Array.from({ length: 24 }).map((_, i) => {
            const a = (i / 24) * Math.PI * 2;
            return <line key={i} x1={200 + Math.cos(a) * 40} y1={200 + Math.sin(a) * 40}
                         x2={200 + Math.cos(a) * 190} y2={200 + Math.sin(a) * 190}
                         stroke="var(--color-primary)" strokeWidth="0.5" />;
          })}
        </svg>
        <div className="absolute inset-0 bg-gradient-to-b from-[var(--color-primary-soft)]/30 to-transparent" />
      </div>

      <div className="relative mx-auto max-w-[1200px] px-5 py-20 sm:px-6 lg:py-24">
        <div className="max-w-3xl">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-[var(--color-line-strong)] bg-[var(--color-surface)] px-3 py-1 text-[12.5px] text-[var(--color-ink-secondary)]">
            <span className="size-1.5 rounded-full bg-[var(--color-quantum)]" aria-hidden />
            Smart India Hackathon 2026 · Problem Statement 26139 · Team CodeNova
          </div>
          <h1 className="text-[40px] font-semibold leading-[1.1] tracking-[-0.03em] text-[var(--color-ink)] sm:text-[52px]">
            Hybrid quantum-classical AI<br />for smarter lung screening
          </h1>
          <p className="mt-5 max-w-2xl text-[16.5px] leading-relaxed text-[var(--color-ink-secondary)]">
            AI-assisted CT analysis combining deep learning, quantum machine learning and
            explainable AI — built to fit a radiology workflow, not to replace the radiologist.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <Link href="/dashboard/analyze"
                  className="inline-flex items-center gap-2 rounded-[var(--radius-md)] bg-[var(--color-primary)] px-5 py-2.5 text-[14.5px] font-medium text-white shadow-[var(--shadow-sm)] hover:bg-[var(--color-primary-hover)]">
              Analyze scan <ArrowRight className="size-4" aria-hidden />
            </Link>
            <Link href="/dashboard"
                  className="inline-flex items-center gap-2 rounded-[var(--radius-md)] border border-[var(--color-line-strong)] bg-[var(--color-surface)] px-5 py-2.5 text-[14.5px] font-medium text-[var(--color-ink)] hover:bg-[var(--color-surface-sunken)]">
              Open clinical dashboard
            </Link>
          </div>

          <div className="mt-8 flex items-start gap-2.5 rounded-[var(--radius-md)] border border-[var(--color-moderate-border)] bg-[var(--color-moderate-soft)] px-3.5 py-2.5">
            <ShieldCheck className="mt-0.5 size-4 shrink-0 text-[var(--color-moderate)]" aria-hidden />
            <p className="text-[12.5px] leading-relaxed text-[var(--color-ink-secondary)]">
              <span className="font-semibold text-[var(--color-moderate)]">
                AI-assisted research and clinical decision-support tool. Not a replacement for
                professional medical diagnosis.
              </span>{" "}
              Research prototype — no regulatory clearance, no clinical validation.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

function Capabilities() {
  const items = [
    { icon: Activity, title: "Analyze chest CT", body: "PNG, JPG or DICOM. Rescale slope/intercept and lung windowing applied correctly for genuine DICOM." },
    { icon: Eye, title: "Identify suspicious regions", body: "A U-Net trained on radiologist consensus contours localises nodules, with detection rate reported alongside Dice." },
    { icon: Layers, title: "Estimate AI-assisted risk", body: "Classical, quantum and hybrid branches each report independently, with their agreement shown." },
    { icon: Workflow, title: "Prioritise high-risk cases", body: "A transparent triage rule orders the work queue. Ambiguous cases escalate rather than hide." },
    { icon: FileText, title: "Generate preliminary reports", body: "Structured AI-generated reports move through doctor review, edit and approval." },
    { icon: GitBranch, title: "Track follow-ups", body: "Clinician-set review dates. The platform never invents surveillance intervals." },
  ];
  return (
    <Section id="capabilities" eyebrow="Capabilities" title="Built around the radiology workflow"
      subtitle="Not a single-purpose classifier — a decision-support layer that fits how screening actually runs.">
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {items.map((i) => (
          <div key={i.title} className="rounded-[var(--radius-lg)] border border-[var(--color-line)] p-5">
            <div className="mb-3 flex size-9 items-center justify-center rounded-[var(--radius-md)] bg-[var(--color-primary-soft)]">
              <i.icon className="size-[18px] text-[var(--color-primary)]" aria-hidden />
            </div>
            <h3 className="text-[14.5px] font-semibold">{i.title}</h3>
            <p className="mt-1.5 text-[13.5px] leading-relaxed text-[var(--color-ink-secondary)]">{i.body}</p>
          </div>
        ))}
      </div>
    </Section>
  );
}

function HowItWorks() {
  const steps = [
    { n: "01", t: "Ingest & validate", d: "Format, resolution and lung-field checks before any inference runs." },
    { n: "02", t: "Preprocess", d: "Bilateral denoise, CLAHE, aspect-preserving letterbox to 300×300." },
    { n: "03", t: "Classical branch", d: "EfficientNet-B3 embedding feeds a softmax head plus RF, SVM, XGBoost and logistic regression." },
    { n: "04", t: "Quantum branch", d: "PCA to 32 dimensions, learned projection, angle encoding into a variational circuit." },
    { n: "05", t: "Hybrid fusion", d: "Four fusion strategies compared; the winner is chosen on validation only." },
    { n: "06", t: "Explain & report", d: "Grad-CAM++ attention, U-Net segmentation, structured report for clinician review." },
  ];
  return (
    <Section id="technology" eyebrow="How it works" title="Six stages, all measured"
      subtitle="Every stage is reproducible from a numbered script in the repository."
      tinted>
      <div className="grid grid-cols-1 gap-px overflow-hidden rounded-[var(--radius-lg)] border border-[var(--color-line)] bg-[var(--color-line)] sm:grid-cols-2 lg:grid-cols-3">
        {steps.map((s) => (
          <div key={s.n} className="bg-[var(--color-surface)] p-5">
            <div className="mb-2 font-mono text-[12px] font-semibold text-[var(--color-primary)]">{s.n}</div>
            <h3 className="text-[14.5px] font-semibold">{s.t}</h3>
            <p className="mt-1.5 text-[13px] leading-relaxed text-[var(--color-ink-secondary)]">{s.d}</p>
          </div>
        ))}
      </div>
    </Section>
  );
}

const ARCH_TONE = {
  n: "border-[var(--color-line-strong)] bg-[var(--color-surface)]",
  c: "border-[var(--color-primary-border)] bg-[var(--color-primary-soft)]",
  q: "border-[var(--color-quantum-border)] bg-[var(--color-quantum-soft)]",
  g: "border-[var(--color-low-border)] bg-[var(--color-low-soft)]",
} as const;

function Node({ label, sub, tone }: {
  label: string; sub: string; tone: keyof typeof ARCH_TONE;
}) {
  return (
    <div className={`rounded-[var(--radius-md)] border px-4 py-3 text-center ${ARCH_TONE[tone]}`}>
      <div className="text-[13px] font-semibold">{label}</div>
      <div className="mt-0.5 text-[11.5px] text-[var(--color-ink-tertiary)]">{sub}</div>
    </div>
  );
}

function Down() {
  return (
    <div className="flex justify-center py-2 text-[var(--color-ink-disabled)]" aria-hidden>
      <svg width="12" height="22" viewBox="0 0 12 22"><path d="M6 0v16M6 16l-4-4M6 16l4-4"
        stroke="currentColor" strokeWidth="1.4" fill="none" /></svg>
    </div>
  );
}

function Architecture() {
  return (
    <Section eyebrow="Architecture" title="Hybrid intelligence"
      subtitle="Classical and quantum branches run in parallel over a shared representation, then fuse.">
      <div className="mx-auto max-w-2xl">
        <Node label="CT imaging" sub="PNG · JPG · DICOM" tone="n" />
        <Down />
        <div className="grid grid-cols-2 gap-4">
          <Node label="Classical AI" sub="EfficientNet-B3 · RF · SVM · XGBoost" tone="c" />
          <Node label="Quantum ML" sub="PennyLane variational circuit" tone="q" />
        </div>
        <Down />
        <Node label="Hybrid intelligence" sub="fusion + validation-selected router" tone="c" />
        <Down />
        <Node label="Clinical decision support" sub="risk · region · explainability · report" tone="g" />
      </div>
    </Section>
  );
}

function Explainability() {
  return (
    <Section eyebrow="Explainable AI" title="Two independent views of “where”" tinted
      subtitle="They answer different questions, so the platform shows both and says which is which.">
      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <div className="rounded-[var(--radius-lg)] border border-[var(--color-line)] bg-[var(--color-surface)] p-5">
          <div className="mb-2 flex items-center gap-2">
            <Brain className="size-4 text-[var(--color-primary)]" aria-hidden />
            <h3 className="text-[14.5px] font-semibold">Grad-CAM++ attention</h3>
          </div>
          <p className="text-[13.5px] leading-relaxed text-[var(--color-ink-secondary)]">
            Shows which regions influenced the classifier. Coarse by construction — a 10×10 feature
            grid upsampled to the display resolution.
          </p>
          <p className="mt-2 rounded-[var(--radius-md)] bg-[var(--color-moderate-soft)] px-3 py-2 text-[12.5px] text-[var(--color-ink-secondary)]">
            We measured it against 197 real lesion boxes: localisation was <strong>no better than
            chance</strong>. It is therefore never labelled a tumour boundary.
          </p>
        </div>
        <div className="rounded-[var(--radius-lg)] border border-[var(--color-line)] bg-[var(--color-surface)] p-5">
          <div className="mb-2 flex items-center gap-2">
            <Microscope className="size-4 text-[var(--color-primary)]" aria-hidden />
            <h3 className="text-[14.5px] font-semibold">U-Net segmentation</h3>
          </div>
          <p className="text-[13.5px] leading-relaxed text-[var(--color-ink-secondary)]">
            Trained on LIDC-IDRI radiologist consensus contours — genuine pixel-level supervision.
            This is the output that answers “where exactly”.
          </p>
          <p className="mt-2 rounded-[var(--radius-md)] bg-[var(--color-primary-soft)] px-3 py-2 text-[12.5px] text-[var(--color-ink-secondary)]">
            When it detects a nodule it outlines it to within <strong>~0.17 mm</strong> of the
            radiologist centre (Dice 0.71) — but it detects only <strong>55%</strong>. Both numbers
            are shown, always.
          </p>
        </div>
      </div>
    </Section>
  );
}

function Workflow_() {
  const steps = ["Scan uploaded", "AI analysis", "Triage queue", "Doctor review", "Approve or reject", "Follow-up"];
  return (
    <Section id="workflow" eyebrow="Hospital workflow" title="A human decides, every time"
      subtitle="The AI produces a preliminary report. A clinician reviews, edits, approves or rejects it.">
      <div className="flex flex-wrap items-center justify-center gap-2">
        {steps.map((s, i) => (
          <div key={s} className="flex items-center gap-2">
            <div className="rounded-full border border-[var(--color-line-strong)] bg-[var(--color-surface)] px-3.5 py-1.5 text-[13px] font-medium">
              {s}
            </div>
            {i < steps.length - 1 && <ArrowRight className="size-3.5 text-[var(--color-ink-disabled)]" aria-hidden />}
          </div>
        ))}
      </div>
      <div className="mx-auto mt-6 max-w-2xl rounded-[var(--radius-md)] border border-[var(--color-line)] bg-[var(--color-surface-sunken)] px-4 py-3 text-center text-[13px] text-[var(--color-ink-secondary)]">
        Clinician feedback is stored for future validated training pipelines. It never
        retrains a production model — an unvalidated feedback loop on a clinical model is unsafe.
      </div>
    </Section>
  );
}

function Security() {
  const cols = [
    { icon: CheckCircle2, tone: "text-[var(--color-low)]", title: "Implemented", items: [
      "Opaque patient identifiers — no names or dates of birth modelled",
      "DICOM patient tags never surfaced in API responses",
      "Append-only audit log",
      "Role model across five clinical roles",
      "Demo records flagged in the database",
    ]},
    { icon: Lock, tone: "text-[var(--color-moderate)]", title: "Designed, not enforced", items: [
      "Authentication and session management",
      "Endpoint-level role enforcement",
      "Encryption at rest and in transit",
    ]},
    { icon: Ban, tone: "text-[var(--color-high)]", title: "Explicitly not claimed", items: [
      "HIPAA / GDPR certification",
      "Medical device approval",
      "Clinical validation",
    ]},
  ];
  return (
    <Section id="security" eyebrow="Security & privacy" title="Designed with healthcare considerations" tinted
      subtitle="Stated precisely, because overstating compliance is worse than not having it.">
      <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
        {cols.map((c) => (
          <div key={c.title} className="rounded-[var(--radius-lg)] border border-[var(--color-line)] bg-[var(--color-surface)] p-5">
            <div className={`mb-3 flex items-center gap-2 text-[13.5px] font-semibold ${c.tone}`}>
              <c.icon className="size-4" aria-hidden />{c.title}
            </div>
            <ul className="space-y-1.5">
              {c.items.map((i) => (
                <li key={i} className="text-[13px] leading-relaxed text-[var(--color-ink-secondary)]">· {i}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </Section>
  );
}

function Credibility() {
  const items = [
    { t: "Data leakage measured and removed", d: "The training set has 1,097 slices from 110 patients but no patient IDs. A naive split leaks a near-duplicate of ~55% of test images. We built pseudo-patient groups and split on those." },
    { t: "No quantum advantage claimed", d: "Paired bootstrap on the held-out test split: p ≈ 0.69–0.71. The quantum branch performs comparably to classical models, not better. We publish the p-value either way." },
    { t: "Explainability validated, and it failed", d: "Grad-CAM++ scored against 197 real lesion boxes was no better than chance at localisation. That is why attention maps are never called tumour boundaries." },
    { t: "More data made it worse", d: "Adding 927 LIDC slices raised benign training images 6.5× yet degraded macro-F1 significantly (p = 0.009). It was not deployed." },
  ];
  return (
    <Section eyebrow="Scientific position" title="What we measured, including what did not work"
      subtitle="Four negative results kept in the product. For a clinical audience these matter more than a headline accuracy number.">
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {items.map((i) => (
          <div key={i.t} className="rounded-[var(--radius-lg)] border border-[var(--color-line)] border-l-2 border-l-[var(--color-moderate)] p-5">
            <h3 className="text-[14px] font-semibold">{i.t}</h3>
            <p className="mt-1.5 text-[13px] leading-relaxed text-[var(--color-ink-secondary)]">{i.d}</p>
          </div>
        ))}
      </div>
    </Section>
  );
}

function Section({ id, eyebrow, title, subtitle, children, tinted }: {
  id?: string; eyebrow: string; title: string; subtitle?: string;
  children: React.ReactNode; tinted?: boolean;
}) {
  return (
    <section id={id} className={tinted ? "border-y border-[var(--color-line)] bg-[var(--color-canvas)]" : ""}>
      <div className="mx-auto max-w-[1200px] px-5 py-16 sm:px-6">
        <div className="mb-8 max-w-2xl">
          <div className="mb-2 text-[11.5px] font-semibold uppercase tracking-[0.09em] text-[var(--color-primary)]">
            {eyebrow}
          </div>
          <h2 className="text-[28px] font-semibold leading-tight tracking-[-0.02em]">{title}</h2>
          {subtitle && <p className="mt-2.5 text-[15px] leading-relaxed text-[var(--color-ink-secondary)]">{subtitle}</p>}
        </div>
        {children}
      </div>
    </section>
  );
}

function SiteFooter() {
  return (
    <footer className="border-t border-[var(--color-line)] bg-[var(--color-canvas)]">
      <div className="mx-auto max-w-[1200px] px-5 py-10 sm:px-6">
        <div className="flex flex-wrap items-start justify-between gap-8">
          <div className="max-w-sm">
            <div className="mb-2 flex items-center gap-2.5">
              <div className="flex size-6 items-center justify-center rounded bg-[var(--color-primary)]">
                <Microscope className="size-3.5 text-white" aria-hidden />
              </div>
              <span className="text-[14px] font-semibold">QuantumLung</span>
            </div>
            <p className="text-[12.5px] leading-relaxed text-[var(--color-ink-tertiary)]">
              Hybrid quantum-classical AI for lung cancer screening support.
              Smart India Hackathon 2026 · Problem Statement 26139 · Team CodeNova.
            </p>
          </div>
          <div className="flex gap-12">
            <FooterCol title="Platform" links={[
              ["Clinical dashboard", "/dashboard"],
              ["Scan analysis", "/dashboard/analyze"],
              ["Priority queue", "/dashboard/queue"],
            ]} />
            <FooterCol title="Research" links={[
              ["Model analytics", "/dashboard/analytics"],
              ["Technology", "/dashboard/technology"],
              ["Security posture", "/dashboard/settings"],
            ]} />
          </div>
        </div>
        <div className="mt-8 border-t border-[var(--color-line)] pt-5">
          <p className="text-[12px] leading-relaxed text-[var(--color-ink-tertiary)]">
            <strong className="text-[var(--color-ink-secondary)]">
              AI-assisted research and clinical decision-support tool. Not a replacement for
              professional medical diagnosis.
            </strong>{" "}
            This platform has not been clinically validated, holds no regulatory clearance, and must
            not be used to make patient-care decisions. All demonstration patient data is synthetic.
          </p>
        </div>
      </div>
    </footer>
  );
}

function FooterCol({ title, links }: { title: string; links: [string, string][] }) {
  return (
    <div>
      <div className="mb-2.5 text-[12px] font-semibold uppercase tracking-[0.06em] text-[var(--color-ink-tertiary)]">
        {title}
      </div>
      <ul className="space-y-1.5">
        {links.map(([label, href]) => (
          <li key={href}>
            <Link href={href} className="text-[13px] text-[var(--color-ink-secondary)] hover:text-[var(--color-primary)]">
              {label}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
