"use client";

import { useEffect, useState } from "react";

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
      <table>
        <thead>
          <tr>
            <th>segment</th>
            <th>requests</th>
            <th>p95 ms</th>
            <th>req/min</th>
            <th>errors</th>
            <th>avg cost</th>
            <th>grounding</th>
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
