import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { LungClass, Priority, RiskBand } from "./types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/* ------------------------------------------------------------- formatting */

export const pct = (v: number | null | undefined, digits = 1) =>
  v == null || Number.isNaN(v) ? "—" : `${(100 * v).toFixed(digits)}%`;

export const num = (v: number | null | undefined, digits = 3) =>
  v == null || Number.isNaN(v) ? "—" : v.toFixed(digits);

export function formatDate(iso: string | null | undefined) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
}

export function formatDateTime(iso: string | null | undefined) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString("en-GB", {
    day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit",
  });
}

export function relativeTime(iso: string | null | undefined) {
  if (!iso) return "—";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";
  const mins = Math.round((Date.now() - then) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.round(hrs / 24);
  if (days < 30) return `${days}d ago`;
  return formatDate(iso);
}

/* --------------------------------------------------------------- semantics */

/**
 * Risk styling. Every consumer pairs these with an icon and a text label —
 * colour alone must never carry the meaning (WCAG 1.4.1).
 */
export const riskStyles: Record<RiskBand, { text: string; bg: string; border: string; dot: string }> = {
  Low: {
    text: "text-[var(--color-low)]",
    bg: "bg-[var(--color-low-soft)]",
    border: "border-[var(--color-low-border)]",
    dot: "bg-[var(--color-low)]",
  },
  Intermediate: {
    text: "text-[var(--color-moderate)]",
    bg: "bg-[var(--color-moderate-soft)]",
    border: "border-[var(--color-moderate-border)]",
    dot: "bg-[var(--color-moderate)]",
  },
  High: {
    text: "text-[var(--color-high)]",
    bg: "bg-[var(--color-high-soft)]",
    border: "border-[var(--color-high-border)]",
    dot: "bg-[var(--color-high)]",
  },
};

export const priorityLabel: Record<Priority, string> = {
  high: "High priority",
  attention: "Attention required",
  routine: "Routine",
};

export const priorityToRisk: Record<Priority, RiskBand> = {
  high: "High",
  attention: "Intermediate",
  routine: "Low",
};

export const classLabel: Record<LungClass, string> = {
  normal: "No suspicious finding",
  benign: "Benign-appearing finding",
  malignant: "Suspicious finding",
};

export const reviewStatusLabel: Record<string, string> = {
  ai_generated: "AI generated",
  in_review: "In review",
  edited: "Edited",
  approved: "Approved",
  further_review: "Further review",
  rejected: "AI finding rejected",
};

/**
 * Triage priority from the model output.
 *
 * Deliberately simple and transparent: it is a presentation-layer ordering
 * rule, not a clinical algorithm, and it is documented as such in the UI.
 * Low agreement escalates because branch disagreement marks ambiguous cases.
 */
export function derivePriority(
  predicted: LungClass,
  confidence: number,
  agreement: number,
): Priority {
  if (predicted === "malignant") return "high";
  if (predicted === "benign") return "attention";
  // Normal, but the branches disagreed or confidence is weak -> worth a look.
  if (agreement < 0.75 || confidence < 0.7) return "attention";
  return "routine";
}
