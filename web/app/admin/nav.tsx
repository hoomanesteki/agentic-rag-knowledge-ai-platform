"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

// Shared backoffice top bar: brand plus tabs with an active state, so every admin view feels like
// one console.
const TABS = [
  { href: "/admin/analytics", label: "Overview" },
  { href: "/admin", label: "Review queue" },
  { href: "/admin/quality", label: "Quality" },
  { href: "/admin/health", label: "Health" },
  { href: "/admin/insights", label: "Insights" },
];

export function AdminNav() {
  const path = usePathname();
  return (
    <div className="ax-topbar">
      <Link href="/admin" className="ax-brand">
        Aster <span>Backoffice</span>
      </Link>
      <nav className="ax-tabs">
        {TABS.map((t) => (
          <Link key={t.href} href={t.href} className={path === t.href ? "on" : ""}>
            {t.label}
          </Link>
        ))}
      </nav>
      <Link href="/" className="ax-exit">
        View store &rarr;
      </Link>
    </div>
  );
}
