"use client";

import { useEffect, useState } from "react";

import { BarList, Donut, Funnel, Sparkbars } from "../charts";
import { AdminNav } from "../nav";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Analytics = {
  kpis: Record<string, number>;
  sessions_by_day: { day: string; count: number }[];
  top_searches: { term: string; count: number }[];
  top_questions: { question: string; count: number }[];
  top_products: { name: string; count: number }[];
  by_category: { label: string; pct: number }[];
  by_device: { label: string; pct: number }[];
  by_country: { label: string; pct: number }[];
  funnel: { step: string; count: number }[];
};
type Gap = { question: string; count: number };

const money = (n: number) => "$" + (n || 0).toLocaleString();
const num = (n: number) => (n || 0).toLocaleString();

export default function AnalyticsPage() {
  const [mounted, setMounted] = useState(false);
  const [a, setA] = useState<Analytics | null>(null);
  const [gaps, setGaps] = useState<Gap[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setMounted(true);
    const token = localStorage.getItem("skein_admin_token");
    if (!token) {
      setError("Sign in on the /admin page first.");
      return;
    }
    const headers = { Authorization: `Bearer ${token}` };
    fetch(`${API_BASE}/api/admin/analytics`, { headers })
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setA)
      .catch(() => setError("Could not load analytics (are you an admin?)."));
    fetch(`${API_BASE}/api/admin/gaps`, { headers })
      .then((r) => (r.ok ? r.json() : { gaps: [] }))
      .then((d) => setGaps(d.gaps || []))
      .catch(() => {});
  }, []);

  if (!mounted) return null;
  if (error)
    return (
      <main className="admin">
        <AdminNav />
        <p className="error">{error}</p>
      </main>
    );
  if (!a || !a.kpis)
    return (
      <main className="admin">
        <AdminNav />
        <p className="meta">
          {a ? "No analytics for this domain yet." : "Loading the dashboard..."}
        </p>
      </main>
    );

  const k = a.kpis;
  const kpis = [
    { label: "Sessions", value: num(k.sessions) },
    { label: "Visitors", value: num(k.visitors) },
    { label: "Orders", value: num(k.orders) },
    { label: "Revenue", value: money(k.revenue) },
    { label: "Conversion", value: `${k.conversion}%` },
    { label: "Avg order", value: money(k.aov) },
    { label: "Searches", value: num(k.searches) },
    { label: "Added to bag", value: num(k.add_to_cart) },
  ];

  return (
    <main className="admin">
      <AdminNav />
      <h1>Store overview</h1>
      <p className="meta">Last 30 days, {num(k.sessions)} simulated sessions.</p>

      <div className="stats">
        {kpis.map((kp) => (
          <div key={kp.label} className="stat">
            <div className="stat-label">{kp.label}</div>
            <div className="stat-value">{kp.value}</div>
          </div>
        ))}
      </div>

      <div className="ax-grid">
        <div className="ax-card ax-wide">
          <h3>Sessions per day</h3>
          <Sparkbars data={a.sessions_by_day} />
        </div>
        <div className="ax-card">
          <h3>Conversion funnel</h3>
          <Funnel steps={a.funnel} />
        </div>
        <div className="ax-card">
          <h3>Top searches</h3>
          <BarList items={a.top_searches.map((s) => ({ label: s.term, count: s.count }))} />
        </div>
        <div className="ax-card">
          <h3>What shoppers ask the assistant</h3>
          <BarList items={a.top_questions.map((s) => ({ label: s.question, count: s.count }))} />
        </div>
        <div className="ax-card">
          <h3>Most viewed products</h3>
          <BarList items={a.top_products.map((s) => ({ label: s.name, count: s.count }))} />
        </div>
        <div className="ax-card">
          <h3>Traffic by category</h3>
          <Donut slices={a.by_category.slice(0, 6)} />
        </div>
        <div className="ax-card">
          <h3>By device</h3>
          <Donut slices={a.by_device} />
        </div>
        <div className="ax-card">
          <h3>By country</h3>
          <Donut slices={a.by_country} />
        </div>
        <div className="ax-card ax-wide">
          <h3>Questions we could not answer well</h3>
          {gaps.length === 0 ? (
            <p className="meta">Nothing flagged. The assistant is answering confidently.</p>
          ) : (
            <BarList items={gaps.slice(0, 8).map((g) => ({ label: g.question, count: g.count }))} />
          )}
        </div>
      </div>
    </main>
  );
}
