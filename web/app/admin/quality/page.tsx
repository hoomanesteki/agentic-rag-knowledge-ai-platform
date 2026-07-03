"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Bucket = {
  total: number;
  escalation_rate: number;
  abstain_rate: number;
  avg_grounding: number | null;
  thumbs_up: number;
  thumbs_down: number;
};

type Quality = { overall: Bucket; by_language: Record<string, Bucket> };

function Row({ name, b }: { name: string; b: Bucket }) {
  return (
    <tr>
      <td>{name}</td>
      <td>{b.total}</td>
      <td>{(b.escalation_rate * 100).toFixed(1)}%</td>
      <td>{(b.abstain_rate * 100).toFixed(1)}%</td>
      <td>{b.avg_grounding === null ? "-" : b.avg_grounding.toFixed(2)}</td>
      <td>
        {b.thumbs_up} / {b.thumbs_down}
      </td>
    </tr>
  );
}

function Stat({ label, value, pct, tone }: {
  label: string; value: string; pct: number; tone?: string;
}) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      <div className="bar">
        <div className={`bar-fill ${tone || ""}`} style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} />
      </div>
    </div>
  );
}

export default function QualityPage() {
  const [mounted, setMounted] = useState(false);
  const [data, setData] = useState<Quality | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setMounted(true);
    const token = localStorage.getItem("skein_admin_token");
    if (!token) {
      setError("Sign in on the /admin page first.");
      return;
    }
    fetch(`${API_BASE}/api/admin/quality`, { headers: { Authorization: `Bearer ${token}` } })
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setData)
      .catch(() => setError("Could not load quality (are you an admin?)."));
  }, []);

  if (!mounted) return null;
  if (error) return <main className="admin"><p className="error">{error}</p></main>;
  if (!data) return <main className="admin"><p>Loading...</p></main>;

  const o = data.overall;
  return (
    <main className="admin">
      <h1>Answer quality</h1>
      <div className="stats">
        <Stat label="Turns" value={String(o.total)} pct={100} />
        <Stat label="Grounding" value={o.avg_grounding === null ? "-" : o.avg_grounding.toFixed(2)}
          pct={(o.avg_grounding || 0) * 100} tone="good" />
        <Stat label="Abstained" value={`${(o.abstain_rate * 100).toFixed(0)}%`}
          pct={o.abstain_rate * 100} tone="warn" />
        <Stat label="Escalated" value={`${(o.escalation_rate * 100).toFixed(0)}%`}
          pct={o.escalation_rate * 100} tone="warn" />
        <Stat label="Helpful" value={`${o.thumbs_up} / ${o.thumbs_down}`}
          pct={o.thumbs_up + o.thumbs_down ? (o.thumbs_up / (o.thumbs_up + o.thumbs_down)) * 100 : 0}
          tone="good" />
      </div>
      <table>
        <thead>
          <tr>
            <th>segment</th>
            <th>turns</th>
            <th>escalated</th>
            <th>abstained</th>
            <th>grounding</th>
            <th>up / down</th>
          </tr>
        </thead>
        <tbody>
          <Row name="all" b={data.overall} />
          {Object.entries(data.by_language).map(([lang, b]) => (
            <Row key={lang} name={lang} b={b} />
          ))}
        </tbody>
      </table>
    </main>
  );
}
