"use client";

import { useEffect, useState } from "react";

import { Hint } from "../Hint";
import { AdminNav } from "../nav";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Bucket = {
  total: number;
  p95_latency_ms: number | null;
  error_rate: number;
  avg_cost: number | null;
  avg_grounding: number | null;
  throughput_per_min: number | null;
};

type Health = {
  overall: Bucket & { grounding_trend: { early: number | null; recent: number | null; delta: number | null } };
  by_language: Record<string, Bucket>;
};

function Row({ name, b }: { name: string; b: Bucket }) {
  return (
    <tr>
      <td>{name}</td>
      <td>{b.total}</td>
      <td>{b.p95_latency_ms ?? "-"}</td>
      <td>{b.throughput_per_min ?? "-"}</td>
      <td>{(b.error_rate * 100).toFixed(1)}%</td>
      <td>{b.avg_cost === null ? "-" : `$${b.avg_cost.toFixed(4)}`}</td>
      <td>{b.avg_grounding === null ? "-" : b.avg_grounding.toFixed(2)}</td>
    </tr>
  );
}

export default function HealthPage() {
  const [mounted, setMounted] = useState(false);
  const [data, setData] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setMounted(true);
    const token = localStorage.getItem("skein_admin_token");
    if (!token) {
      setError("Sign in on the /admin page first.");
      return;
    }
    fetch(`${API_BASE}/api/admin/health`, { headers: { Authorization: `Bearer ${token}` } })
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setData)
      .catch(() => setError("Could not load health (are you an admin?)."));
  }, []);

  if (!mounted) return null;
  if (error) return <main className="admin"><p className="error">{error}</p></main>;
  if (!data) return <main className="admin"><p>Loading...</p></main>;

  const trend = data.overall.grounding_trend;
  return (
    <main className="admin">
      <AdminNav />
      <h1>Platform health</h1>
      <p className="ax-intro">
        Live operational health of the assistant from recent request traces: how fast, how reliable,
        and how much each answer costs. Broken out by language, since retrieval quality can differ.
      </p>
      <table>
        <thead>
          <tr>
            <th>segment</th>
            <th>requests<Hint text="Business: how much real traffic this segment carried, so the other numbers have weight. Technical: count of answered turns in the recent window for this language segment." /></th>
            <th>p95 ms<Hint text="Business: the slow-case experience most shoppers never exceed; the number that decides if the assistant feels instant. Technical: 95th-percentile end-to-end latency, so 95 percent of answers were faster than this." /></th>
            <th>req/min<Hint text="Business: current load, which tells you whether to worry about a spike or a stall. Technical: requests per minute over the most recent 15-minute window." /></th>
            <th>errors<Hint text="Business: how often a shopper hit a degraded answer; the reliability promise. Technical: share of turns that failed or fell back, usually an upstream model hiccup absorbed by the fallback chain." /></th>
            <th>avg cost<Hint text="Business: what each answer costs to serve, the unit economics behind gross margin. Technical: average model spend per costed turn in USD; streaming turns omit token cost." /></th>
            <th>grounding<Hint text="Business: how much of each answer is backed by real sources, the trust and hallucination guardrail. Technical: mean source-backing of answered turns (0 to 1); watch for it drifting down over time." /></th>
          </tr>
        </thead>
        <tbody>
          <Row name="all" b={data.overall} />
          {Object.entries(data.by_language).map(([lang, b]) => (
            <Row key={lang} name={lang} b={b} />
          ))}
        </tbody>
      </table>
      {trend.delta !== null && (
        <p className="meta">
          Retrieval-quality trend: {trend.early} → {trend.recent} (delta {trend.delta})
        </p>
      )}
    </main>
  );
}
