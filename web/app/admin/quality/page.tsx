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

  return (
    <main className="admin">
      <h1>Answer quality</h1>
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
