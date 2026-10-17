"use client";

/**
 * Core UI primitives for the QuantumLung platform.
 *
 * Deliberately small and composable. Every interactive element carries a
 * visible focus state, an accessible name, and a disabled/loading treatment —
 * clinical software gets used by keyboard and under time pressure.
 */

import * as React from "react";
import { cn } from "@/lib/utils";
import {
  AlertTriangle, CheckCircle2, ChevronDown, Info, Loader2, X, XCircle,
} from "lucide-react";

/* =============================================================== Button === */

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger" | "subtle";
type ButtonSize = "sm" | "md" | "lg";

const buttonVariants: Record<ButtonVariant, string> = {
  primary:
    "bg-[var(--color-primary)] text-white border border-transparent hover:bg-[var(--color-primary-hover)] active:bg-[var(--color-primary-hover)] shadow-[var(--shadow-xs)]",
  secondary:
    "bg-[var(--color-surface)] text-[var(--color-ink)] border border-[var(--color-line-strong)] hover:bg-[var(--color-surface-hover)] shadow-[var(--shadow-xs)]",
  ghost:
    "bg-transparent text-[var(--color-ink-secondary)] border border-transparent hover:bg-[var(--color-surface-sunken)] hover:text-[var(--color-ink)]",
  danger:
    "bg-[var(--color-high)] text-white border border-transparent hover:brightness-110 shadow-[var(--shadow-xs)]",
  subtle:
    "bg-[var(--color-surface-sunken)] text-[var(--color-ink-secondary)] border border-transparent hover:bg-[var(--color-line)]",
};

const buttonSizes: Record<ButtonSize, string> = {
  sm: "h-8 px-3 text-[13px] gap-1.5",
  md: "h-9 px-3.5 text-[13.5px] gap-2",
  lg: "h-11 px-5 text-[15px] gap-2",
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  icon?: React.ElementType;
  iconRight?: React.ElementType;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "secondary", size = "md", loading, icon: Icon,
     iconRight: IconRight, children, disabled, ...props }, ref) => (
    <button
      ref={ref}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={cn(
        "inline-flex items-center justify-center rounded-[var(--radius-md)] font-medium",
        "transition-colors duration-150 select-none whitespace-nowrap",
        "disabled:opacity-50 disabled:pointer-events-none",
        buttonVariants[variant], buttonSizes[size], className,
      )}
      {...props}
    >
      {loading ? <Loader2 className="size-4 animate-spin" aria-hidden />
               : Icon ? <Icon className="size-4 shrink-0" aria-hidden /> : null}
      {children}
      {IconRight && !loading ? <IconRight className="size-4 shrink-0" aria-hidden /> : null}
    </button>
  ),
);
Button.displayName = "Button";

/* ================================================================= Card === */

export function Card({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "bg-[var(--color-surface)] border border-[var(--color-line)]",
        "rounded-[var(--radius-lg)] shadow-[var(--shadow-xs)]",
        className,
      )}
      {...props}
    />
  );
}

export function CardHeader({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("px-5 py-4 border-b border-[var(--color-line)]", className)} {...props} />;
}

export function CardTitle({ className, ...props }: React.HTMLAttributes<HTMLHeadingElement>) {
  return <h3 className={cn("text-[15px] font-semibold tracking-[-0.01em] text-[var(--color-ink)]", className)} {...props} />;
}

export function CardDescription({ className, ...props }: React.HTMLAttributes<HTMLParagraphElement>) {
  return <p className={cn("text-[13px] text-[var(--color-ink-tertiary)] mt-0.5", className)} {...props} />;
}

export function CardBody({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return <div className={cn("p-5", className)} {...props} />;
}

/* ================================================================ Badge === */

type BadgeTone = "neutral" | "primary" | "low" | "moderate" | "high" | "info" | "quantum";

const badgeTones: Record<BadgeTone, string> = {
  neutral: "bg-[var(--color-surface-sunken)] text-[var(--color-ink-secondary)] border-[var(--color-line)]",
  primary: "bg-[var(--color-primary-soft)] text-[var(--color-primary)] border-[var(--color-primary-border)]",
  low: "bg-[var(--color-low-soft)] text-[var(--color-low)] border-[var(--color-low-border)]",
  moderate: "bg-[var(--color-moderate-soft)] text-[var(--color-moderate)] border-[var(--color-moderate-border)]",
  high: "bg-[var(--color-high-soft)] text-[var(--color-high)] border-[var(--color-high-border)]",
  info: "bg-[var(--color-info-soft)] text-[var(--color-info)] border-[var(--color-info-border)]",
  quantum: "bg-[var(--color-quantum-soft)] text-[var(--color-quantum)] border-[var(--color-quantum-border)]",
};

export function Badge({
  tone = "neutral", icon: Icon, className, children, ...props
}: React.HTMLAttributes<HTMLSpanElement> & { tone?: BadgeTone; icon?: React.ElementType }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5",
        "text-[12px] font-medium leading-5 whitespace-nowrap",
        badgeTones[tone], className,
      )}
      {...props}
    >
      {Icon && <Icon className="size-3.5 shrink-0" aria-hidden />}
      {children}
    </span>
  );
}

/* ========================================================== Risk display === */

/**
 * Risk is communicated by icon + label + colour together, never colour alone.
 */
export function RiskBadge({ risk, className }: { risk: "Low" | "Intermediate" | "High"; className?: string }) {
  const map = {
    Low: { tone: "low" as const, Icon: CheckCircle2, label: "Low risk" },
    Intermediate: { tone: "moderate" as const, Icon: AlertTriangle, label: "Intermediate" },
    High: { tone: "high" as const, Icon: XCircle, label: "High risk" },
  };
  const { tone, Icon, label } = map[risk];
  return <Badge tone={tone} icon={Icon} className={className}>{label}</Badge>;
}

/* ================================================================ Metric === */

export function Metric({
  label, value, sublabel, tone = "neutral", icon: Icon, className,
}: {
  label: string;
  value: React.ReactNode;
  sublabel?: React.ReactNode;
  tone?: "neutral" | "low" | "moderate" | "high" | "primary";
  icon?: React.ElementType;
  className?: string;
}) {
  const valueTone = {
    neutral: "text-[var(--color-ink)]",
    low: "text-[var(--color-low)]",
    moderate: "text-[var(--color-moderate)]",
    high: "text-[var(--color-high)]",
    primary: "text-[var(--color-primary)]",
  }[tone];

  return (
    <div className={cn("px-5 py-4", className)}>
      <div className="flex items-center gap-2">
        {Icon && <Icon className="size-4 text-[var(--color-ink-tertiary)]" aria-hidden />}
        <span className="text-[12.5px] font-medium text-[var(--color-ink-tertiary)] uppercase tracking-[0.04em]">
          {label}
        </span>
      </div>
      <div className={cn("mt-1.5 text-[26px] font-semibold tracking-[-0.02em] tabular", valueTone)}>
        {value}
      </div>
      {sublabel && (
        <div className="mt-0.5 text-[12.5px] text-[var(--color-ink-tertiary)]">{sublabel}</div>
      )}
    </div>
  );
}

/* ================================================================ Notice === */

type NoticeTone = "info" | "warning" | "danger" | "success" | "neutral";

export function Notice({
  tone = "info", title, children, className, icon,
}: {
  tone?: NoticeTone;
  title?: React.ReactNode;
  children?: React.ReactNode;
  className?: string;
  icon?: React.ElementType;
}) {
  const cfg = {
    info: { cls: "bg-[var(--color-info-soft)] border-[var(--color-info-border)] text-[var(--color-info)]", Icon: Info },
    warning: { cls: "bg-[var(--color-moderate-soft)] border-[var(--color-moderate-border)] text-[var(--color-moderate)]", Icon: AlertTriangle },
    danger: { cls: "bg-[var(--color-high-soft)] border-[var(--color-high-border)] text-[var(--color-high)]", Icon: XCircle },
    success: { cls: "bg-[var(--color-low-soft)] border-[var(--color-low-border)] text-[var(--color-low)]", Icon: CheckCircle2 },
    neutral: { cls: "bg-[var(--color-surface-sunken)] border-[var(--color-line)] text-[var(--color-ink-secondary)]", Icon: Info },
  }[tone];
  const Icon = icon ?? cfg.Icon;

  return (
    <div className={cn("flex gap-2.5 rounded-[var(--radius-md)] border px-3.5 py-3", cfg.cls, className)} role="note">
      <Icon className="size-4 shrink-0 mt-0.5" aria-hidden />
      <div className="min-w-0 text-[13px] leading-relaxed">
        {title && <div className="font-semibold mb-0.5">{title}</div>}
        <div className={cn(title ? "text-[var(--color-ink-secondary)]" : "")}>{children}</div>
      </div>
    </div>
  );
}

/* ================================================================= Table === */

export function Table({ className, ...props }: React.TableHTMLAttributes<HTMLTableElement>) {
  return (
    <div className="w-full overflow-x-auto">
      <table className={cn("w-full border-collapse text-[13.5px]", className)} {...props} />
    </div>
  );
}

export function Th({ className, ...props }: React.ThHTMLAttributes<HTMLTableCellElement>) {
  return (
    <th
      className={cn(
        "sticky top-0 z-10 bg-[var(--color-surface-sunken)] text-left",
        "px-4 py-2.5 font-medium text-[12px] uppercase tracking-[0.04em]",
        "text-[var(--color-ink-tertiary)] border-b border-[var(--color-line)] whitespace-nowrap",
        className,
      )}
      {...props}
    />
  );
}

export function Td({ className, ...props }: React.TdHTMLAttributes<HTMLTableCellElement>) {
  return <td className={cn("px-4 py-3 border-b border-[var(--color-line)] align-middle", className)} {...props} />;
}

export function Tr({ className, ...props }: React.HTMLAttributes<HTMLTableRowElement>) {
  return <tr className={cn("transition-colors hover:bg-[var(--color-surface-hover)]", className)} {...props} />;
}

/* ============================================================== Skeleton === */

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn("skeleton rounded-[var(--radius-sm)]", className)} aria-hidden />;
}

export function SkeletonTable({ rows = 5, cols = 5 }: { rows?: number; cols?: number }) {
  return (
    <div className="p-4 space-y-2.5" aria-label="Loading" role="status">
      {Array.from({ length: rows }).map((_, r) => (
        <div key={r} className="flex gap-3">
          {Array.from({ length: cols }).map((_, c) => (
            <Skeleton key={c} className={cn("h-5", c === 0 ? "w-32" : "flex-1")} />
          ))}
        </div>
      ))}
    </div>
  );
}

/* ============================================================ EmptyState === */

export function EmptyState({
  icon: Icon, title, description, action, className,
}: {
  icon?: React.ElementType;
  title: string;
  description?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-col items-center justify-center text-center px-6 py-14", className)}>
      {Icon && (
        <div className="mb-3 flex size-11 items-center justify-center rounded-full bg-[var(--color-surface-sunken)]">
          <Icon className="size-5 text-[var(--color-ink-tertiary)]" aria-hidden />
        </div>
      )}
      <h3 className="text-[14.5px] font-semibold text-[var(--color-ink)]">{title}</h3>
      {description && (
        <p className="mt-1 max-w-sm text-[13px] text-[var(--color-ink-tertiary)] leading-relaxed">{description}</p>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

/* ============================================================ ErrorState === */

export function ErrorState({ title = "Something went wrong", message, onRetry }: {
  title?: string; message?: string; onRetry?: () => void;
}) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-12 text-center" role="alert">
      <div className="mb-3 flex size-11 items-center justify-center rounded-full bg-[var(--color-high-soft)]">
        <AlertTriangle className="size-5 text-[var(--color-high)]" aria-hidden />
      </div>
      <h3 className="text-[14.5px] font-semibold text-[var(--color-ink)]">{title}</h3>
      {message && <p className="mt-1 max-w-md text-[13px] text-[var(--color-ink-tertiary)]">{message}</p>}
      {onRetry && <Button className="mt-4" onClick={onRetry}>Try again</Button>}
    </div>
  );
}

/* ================================================================= Input === */

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        "h-9 w-full rounded-[var(--radius-md)] border border-[var(--color-line-strong)]",
        "bg-[var(--color-surface)] px-3 text-[13.5px] text-[var(--color-ink)]",
        "placeholder:text-[var(--color-ink-disabled)] transition-colors",
        "hover:border-[var(--color-ink-disabled)] disabled:opacity-50 disabled:bg-[var(--color-surface-sunken)]",
        className,
      )}
      {...props}
    />
  ),
);
Input.displayName = "Input";

export const Textarea = React.forwardRef<HTMLTextAreaElement, React.TextareaHTMLAttributes<HTMLTextAreaElement>>(
  ({ className, ...props }, ref) => (
    <textarea
      ref={ref}
      className={cn(
        "w-full rounded-[var(--radius-md)] border border-[var(--color-line-strong)]",
        "bg-[var(--color-surface)] px-3 py-2 text-[13.5px] leading-relaxed text-[var(--color-ink)]",
        "placeholder:text-[var(--color-ink-disabled)] transition-colors resize-y",
        "hover:border-[var(--color-ink-disabled)] disabled:opacity-50",
        className,
      )}
      {...props}
    />
  ),
);
Textarea.displayName = "Textarea";

export function Select({ className, children, ...props }: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <div className="relative">
      <select
        className={cn(
          "h-9 w-full appearance-none rounded-[var(--radius-md)] border border-[var(--color-line-strong)]",
          "bg-[var(--color-surface)] pl-3 pr-8 text-[13.5px] text-[var(--color-ink)]",
          "transition-colors hover:border-[var(--color-ink-disabled)] disabled:opacity-50",
          className,
        )}
        {...props}
      >
        {children}
      </select>
      <ChevronDown className="pointer-events-none absolute right-2.5 top-1/2 size-4 -translate-y-1/2 text-[var(--color-ink-tertiary)]" aria-hidden />
    </div>
  );
}

/* ================================================================= Modal === */

export function Modal({
  open, onClose, title, description, children, footer, size = "md",
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  description?: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
  size?: "sm" | "md" | "lg";
}) {
  React.useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [open, onClose]);

  if (!open) return null;
  const widths = { sm: "max-w-md", md: "max-w-xl", lg: "max-w-3xl" };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-[#101828]/40 backdrop-blur-[2px]" onClick={onClose} aria-hidden />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className={cn(
          "relative w-full animate-fade-up rounded-[var(--radius-xl)] bg-[var(--color-surface)]",
          "shadow-[var(--shadow-lg)] border border-[var(--color-line)]", widths[size],
        )}
      >
        <div className="flex items-start justify-between gap-4 border-b border-[var(--color-line)] px-5 py-4">
          <div>
            <h2 className="text-[15px] font-semibold text-[var(--color-ink)]">{title}</h2>
            {description && <p className="mt-0.5 text-[13px] text-[var(--color-ink-tertiary)]">{description}</p>}
          </div>
          <button
            onClick={onClose}
            aria-label="Close dialog"
            className="rounded-[var(--radius-sm)] p-1 text-[var(--color-ink-tertiary)] hover:bg-[var(--color-surface-sunken)] hover:text-[var(--color-ink)]"
          >
            <X className="size-4" aria-hidden />
          </button>
        </div>
        <div className="max-h-[70vh] overflow-y-auto px-5 py-4">{children}</div>
        {footer && (
          <div className="flex justify-end gap-2 border-t border-[var(--color-line)] px-5 py-3.5">{footer}</div>
        )}
      </div>
    </div>
  );
}

/* =============================================================== Tooltip === */

export function Tooltip({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <span className="group/tt relative inline-flex">
      {children}
      <span
        role="tooltip"
        className={cn(
          "pointer-events-none absolute bottom-full left-1/2 z-40 mb-1.5 -translate-x-1/2",
          "whitespace-nowrap rounded-[var(--radius-sm)] bg-[var(--color-ink)] px-2 py-1",
          "text-[12px] font-medium text-white opacity-0 shadow-[var(--shadow-md)]",
          "transition-opacity group-hover/tt:opacity-100 group-focus-within/tt:opacity-100",
        )}
      >
        {label}
      </span>
    </span>
  );
}

/* ============================================================= ProgressBar === */

export function ProgressBar({ value, tone = "primary", className }: {
  value: number; tone?: "primary" | "low" | "moderate" | "high" | "quantum"; className?: string;
}) {
  const bg = {
    primary: "bg-[var(--color-primary)]",
    low: "bg-[var(--color-low)]",
    moderate: "bg-[var(--color-moderate)]",
    high: "bg-[var(--color-high)]",
    quantum: "bg-[var(--color-quantum)]",
  }[tone];
  const clamped = Math.max(0, Math.min(1, value));
  return (
    <div
      className={cn("h-1.5 w-full overflow-hidden rounded-full bg-[var(--color-surface-sunken)]", className)}
      role="progressbar"
      aria-valuenow={Math.round(clamped * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div className={cn("h-full rounded-full transition-[width] duration-500 ease-out", bg)}
           style={{ width: `${clamped * 100}%` }} />
    </div>
  );
}

/* ============================================================= DemoBadge === */

/** Every synthetic record in the product is labelled. Non-negotiable. */
export function DemoBadge({ className }: { className?: string }) {
  return (
    <Badge tone="neutral" className={cn("font-normal", className)} title="Synthetic record for demonstration">
      Demo Data
    </Badge>
  );
}

/* ============================================================ SectionHead === */

export function SectionHead({ eyebrow, title, description, actions, className }: {
  eyebrow?: string;
  title: string;
  description?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-wrap items-end justify-between gap-4", className)}>
      <div className="min-w-0">
        {eyebrow && (
          <div className="mb-1 text-[11.5px] font-semibold uppercase tracking-[0.08em] text-[var(--color-primary)]">
            {eyebrow}
          </div>
        )}
        <h1 className="text-[21px] font-semibold tracking-[-0.02em] text-[var(--color-ink)]">{title}</h1>
        {description && (
          <p className="mt-1 max-w-2xl text-[13.5px] leading-relaxed text-[var(--color-ink-secondary)]">{description}</p>
        )}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}

/* ============================================================== KeyValue === */

export function KeyValue({ label, value, mono = true }: {
  label: React.ReactNode; value: React.ReactNode; mono?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b border-[var(--color-line)] py-2 last:border-0">
      <dt className="text-[13px] text-[var(--color-ink-tertiary)]">{label}</dt>
      <dd className={cn("text-right text-[13px] text-[var(--color-ink)]", mono && "font-mono text-[12.5px] tabular")}>
        {value}
      </dd>
    </div>
  );
}
