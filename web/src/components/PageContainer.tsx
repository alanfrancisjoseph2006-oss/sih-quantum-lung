import { cn } from "@/lib/utils";

/**
 * Standard page padding and max width.
 *
 * `wide` is used by the radiology workspace, which needs the full viewport;
 * everything else is capped so tables and prose stay readable on 4K displays.
 */
export function PageContainer({
  children, className, wide = false,
}: { children: React.ReactNode; className?: string; wide?: boolean }) {
  return (
    <div className={cn("mx-auto w-full px-5 py-6 sm:px-6 lg:px-8", wide ? "max-w-none" : "max-w-[1400px]", className)}>
      {children}
    </div>
  );
}
