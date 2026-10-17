"use client";

/**
 * Application shell: collapsible sidebar, top bar, main workspace.
 *
 * The sidebar collapses to icons rather than disappearing, so the navigation
 * model stays stable — radiology software gets used by muscle memory.
 */

import * as React from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Activity, Bell, ChevronLeft, ChevronsUpDown, FileText, FlaskConical, LayoutDashboard, ListChecks, Menu, Microscope, Radar, Scan, Search, Settings, Stethoscope, Users,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui";

interface NavItem {
  href: string;
  label: string;
  icon: React.ElementType;
  group: "clinical" | "insight" | "system";
}

const NAV: NavItem[] = [
  { href: "/dashboard", label: "Overview", icon: LayoutDashboard, group: "clinical" },
  { href: "/dashboard/analyze", label: "Scan Analysis", icon: Scan, group: "clinical" },
  { href: "/dashboard/queue", label: "Priority Queue", icon: ListChecks, group: "clinical" },
  { href: "/dashboard/patients", label: "Patients", icon: Users, group: "clinical" },
  { href: "/dashboard/follow-ups", label: "Follow-ups", icon: Stethoscope, group: "clinical" },
  { href: "/dashboard/reports", label: "Reports", icon: FileText, group: "clinical" },
  { href: "/dashboard/analytics", label: "Model Analytics", icon: Activity, group: "insight" },
  { href: "/dashboard/domain-shift", label: "Domain Shift", icon: Radar, group: "insight" },
  { href: "/dashboard/technology", label: "Technology", icon: FlaskConical, group: "insight" },
  { href: "/dashboard/settings", label: "Settings", icon: Settings, group: "system" },
];

const GROUP_LABEL: Record<NavItem["group"], string> = {
  clinical: "Clinical",
  insight: "Insight",
  system: "System",
};

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = React.useState(false);
  const [mobileOpen, setMobileOpen] = React.useState(false);

  // Restore the sidebar preference; radiologists keep it collapsed on small
  // screens. Deferred a tick: localStorage is unavailable during SSR, and
  // setting state synchronously in an effect triggers a cascading render.
  React.useEffect(() => {
    const t = setTimeout(() => {
      try {
        const v = localStorage.getItem("ql.sidebar.collapsed");
        if (v) setCollapsed(v === "1");
      } catch { /* storage unavailable */ }
    }, 0);
    return () => clearTimeout(t);
  }, []);

  const toggle = () => {
    setCollapsed((c) => {
      try { localStorage.setItem("ql.sidebar.collapsed", c ? "0" : "1"); } catch {}
      return !c;
    });
  };

  // Close the mobile drawer on navigation. Guarded so it only updates state
  // when the drawer is actually open.
  React.useEffect(() => {
    if (!mobileOpen) return;
    const t = setTimeout(() => setMobileOpen(false), 0);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname]);

  const groups = ["clinical", "insight", "system"] as const;

  const sidebar = (
    <nav
      aria-label="Primary"
      className={cn(
        "flex h-full flex-col border-r border-[var(--color-line)] bg-[var(--color-surface)]",
        "transition-[width] duration-200 ease-out",
        collapsed ? "w-[68px]" : "w-[236px]",
      )}
    >
      {/* Brand */}
      <div className="flex h-14 items-center gap-2.5 border-b border-[var(--color-line)] px-4">
        <div className="flex size-7 shrink-0 items-center justify-center rounded-[var(--radius-md)] bg-[var(--color-primary)]">
          <Microscope className="size-4 text-white" aria-hidden />
        </div>
        {!collapsed && (
          <div className="min-w-0">
            <div className="truncate text-[13.5px] font-semibold tracking-[-0.01em]">QuantumLung</div>
            <div className="truncate text-[11px] text-[var(--color-ink-tertiary)]">Clinical Decision Support</div>
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-2.5 py-3">
        {groups.map((g) => {
          const items = NAV.filter((n) => n.group === g);
          if (!items.length) return null;
          return (
            <div key={g} className="mb-4 last:mb-0">
              {!collapsed && (
                <div className="mb-1 px-2.5 text-[10.5px] font-semibold uppercase tracking-[0.09em] text-[var(--color-ink-disabled)]">
                  {GROUP_LABEL[g]}
                </div>
              )}
              <ul className="space-y-0.5">
                {items.map(({ href, label, icon: Icon }) => {
                  const active = pathname === href ||
                    (href !== "/dashboard" && pathname.startsWith(href));
                  return (
                    <li key={href}>
                      <Link
                        href={href}
                        title={collapsed ? label : undefined}
                        aria-current={active ? "page" : undefined}
                        className={cn(
                          "flex items-center gap-2.5 rounded-[var(--radius-md)] px-2.5 py-2",
                          "text-[13.5px] font-medium transition-colors",
                          active
                            ? "bg-[var(--color-primary-soft)] text-[var(--color-primary)]"
                            : "text-[var(--color-ink-secondary)] hover:bg-[var(--color-surface-sunken)] hover:text-[var(--color-ink)]",
                          collapsed && "justify-center px-0",
                        )}
                      >
                        <Icon className="size-[17px] shrink-0" aria-hidden />
                        {!collapsed && <span className="truncate">{label}</span>}
                      </Link>
                    </li>
                  );
                })}
              </ul>
            </div>
          );
        })}
      </div>

      <div className="border-t border-[var(--color-line)] p-2.5">
        <button
          onClick={toggle}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          className={cn(
            "flex w-full items-center gap-2.5 rounded-[var(--radius-md)] px-2.5 py-2",
            "text-[13px] text-[var(--color-ink-tertiary)] transition-colors",
            "hover:bg-[var(--color-surface-sunken)] hover:text-[var(--color-ink)]",
            collapsed && "justify-center px-0",
          )}
        >
          <ChevronLeft className={cn("size-4 transition-transform", collapsed && "rotate-180")} aria-hidden />
          {!collapsed && "Collapse"}
        </button>
      </div>
    </nav>
  );

  return (
    <div className="flex h-screen overflow-hidden bg-[var(--color-canvas)]">
      {/* Desktop sidebar */}
      <div className="hidden md:block">{sidebar}</div>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 md:hidden">
          <div className="absolute inset-0 bg-[#101828]/40" onClick={() => setMobileOpen(false)} aria-hidden />
          <div className="absolute left-0 top-0 h-full animate-fade-up">{sidebar}</div>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar onMenu={() => setMobileOpen(true)} />
        <main className="flex-1 overflow-y-auto" id="main-content">
          {children}
        </main>
      </div>
    </div>
  );
}

function TopBar({ onMenu }: { onMenu: () => void }) {
  const router = useRouter();
  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b border-[var(--color-line)] bg-[var(--color-surface)] px-4">
      <button
        onClick={onMenu}
        aria-label="Open navigation"
        className="rounded-[var(--radius-md)] p-1.5 text-[var(--color-ink-secondary)] hover:bg-[var(--color-surface-sunken)] md:hidden"
      >
        <Menu className="size-5" aria-hidden />
      </button>

      {/* Organisation selector — single org in the prototype, but the control
          exists because multi-tenancy is part of the data model. */}
      <button
        className={cn(
          "hidden items-center gap-2 rounded-[var(--radius-md)] border border-[var(--color-line)]",
          "px-2.5 py-1.5 text-[13px] font-medium text-[var(--color-ink-secondary)]",
          "hover:bg-[var(--color-surface-sunken)] sm:flex",
        )}
        title="Organisation"
      >
        <span className="flex size-5 items-center justify-center rounded bg-[var(--color-primary-soft)] text-[10px] font-bold text-[var(--color-primary)]">
          DH
        </span>
        Demonstration Hospital
        <ChevronsUpDown className="size-3.5 text-[var(--color-ink-disabled)]" aria-hidden />
      </button>

      <div className="relative ml-auto hidden max-w-xs flex-1 sm:block">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-[var(--color-ink-disabled)]" aria-hidden />
        <input
          type="search"
          placeholder="Search patient ID…"
          aria-label="Search patient ID"
          className={cn(
            "h-9 w-full rounded-[var(--radius-md)] border border-[var(--color-line-strong)]",
            "bg-[var(--color-surface)] pl-8 pr-3 text-[13px]",
            "placeholder:text-[var(--color-ink-disabled)]",
          )}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              const v = (e.target as HTMLInputElement).value.trim();
              if (v) router.push(`/dashboard/patients?search=${encodeURIComponent(v)}`);
            }
          }}
        />
      </div>

      <Badge tone="neutral" className="ml-auto shrink-0 sm:ml-0">Research prototype</Badge>

      <button
        aria-label="Notifications"
        className="relative rounded-[var(--radius-md)] p-1.5 text-[var(--color-ink-secondary)] hover:bg-[var(--color-surface-sunken)]"
      >
        <Bell className="size-[18px]" aria-hidden />
      </button>

      <div className="flex items-center gap-2 border-l border-[var(--color-line)] pl-3">
        <div className="flex size-7 items-center justify-center rounded-full bg-[var(--color-surface-sunken)] text-[11px] font-semibold text-[var(--color-ink-secondary)]">
          DR
        </div>
        <div className="hidden leading-tight lg:block">
          <div className="text-[12.5px] font-medium">Demo Reviewer</div>
          <div className="text-[11px] text-[var(--color-ink-tertiary)]">Radiologist</div>
        </div>
      </div>
    </header>
  );
}
