import { ShieldAlert } from "lucide-react";

/**
 * The regulatory statement. Present on every clinical surface, never dismissible.
 */
export function DisclaimerBar({ compact = false }: { compact?: boolean }) {
  return (
    <div className="flex items-start gap-2.5 rounded-[var(--radius-md)] border border-[var(--color-moderate-border)] bg-[var(--color-moderate-soft)] px-3.5 py-2.5">
      <ShieldAlert className="mt-0.5 size-4 shrink-0 text-[var(--color-moderate)]" aria-hidden />
      <p className="text-[12.5px] leading-relaxed text-[var(--color-ink-secondary)]">
        <span className="font-semibold text-[var(--color-moderate)]">
          AI-assisted research and clinical decision-support tool. Not a replacement for professional medical diagnosis.
        </span>
        {!compact && (
          <> Outputs are model predictions on a single 2D CT slice and require confirmation by a
          qualified clinician. Any cancer diagnosis requires histopathology.</>
        )}
      </p>
    </div>
  );
}
