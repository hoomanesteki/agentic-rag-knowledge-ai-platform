"use client";

import { useEffect, useState } from "react";

import { Hint } from "../Hint";
import { AdminNav } from "../nav";

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

function Stat({ label, value, pct, tone, hint }: {
  label: string; value: string; pct?: number; tone?: string; hint?: string;
}) {
  return (
    <div className="stat">
      <div className="stat-label">
        {label}
        {hint && <Hint text={hint} />}
      </div>
      <div className="stat-value">{value}</div>
      {pct !== undefined && (
        <div className="bar">
          <div className={`bar-fill ${tone || ""}`}
            style={{ width: `${Math.min(100, Math.max(0, pct))}%` }} />
        </div>
      )}
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
      <AdminNav />
      <h1>Answer quality</h1>
      <p className="ax-intro">
        How well the assistant is answering shoppers, from the same request traces the store runs
        on. Hover any <span className="ax-hint" style={{ cursor: "default" }}>?</span> for what a
        metric means.
      </p>
      <div className="stats">
        <Stat label="Turns" value={String(o.total)}
          hint="Total assistant answers in the recent traffic window." />
        <Stat label="Grounding" value={o.avg_grounding === null ? "-" : o.avg_grounding.toFixed(2)}
          pct={(o.avg_grounding || 0) * 100} tone="good"
          hint="How much of each answer is backed by the retrieved sources (0 to 1). Higher means less risk of the model making things up." />
        <Stat label="Abstained" value={`${(o.abstain_rate * 100).toFixed(0)}%`}
          pct={o.abstain_rate * 100} tone="warn"
          hint="Share of turns where the assistant declined to answer rather than guess, because retrieval was not confident. A safety feature; a spike means a knowledge gap." />
        <Stat label="Escalated" value={`${(o.escalation_rate * 100).toFixed(0)}%`}
          pct={o.escalation_rate * 100} tone="warn"
          hint="Share of turns handed to a human specialist. These land in the review queue." />
        <Stat label="Helpful" value={`${o.thumbs_up} / ${o.thumbs_down}`}
          pct={o.thumbs_up + o.thumbs_down ? (o.thumbs_up / (o.thumbs_up + o.thumbs_down)) * 100 : 0}
          tone="good"
          hint="Thumbs up versus thumbs down that shoppers left on answers." />
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
