"use client";

import { useEffect, useState } from "react";

import { Hint } from "../Hint";
import { AdminNav } from "../nav";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Domain = {
  domain: string;
  ontology: { entity_types: string[]; edges: { type: string; from: string; to: string }[] };
  metrics: { name: string; grain: string | null; dimensions: string[]; params: string[] }[];
  lineage: { medallion: { role: string; layers: string[]; pii_columns: string[]; metrics: string[] }[] };
  mlflow_url: string | null;
  langfuse_url: string | null;
};

type Gap = { question: string; count: number; lang?: string };

export default function InsightsPage() {
  const [mounted, setMounted] = useState(false);
  const [domain, setDomain] = useState<Domain | null>(null);
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
    fetch(`${API_BASE}/api/admin/domain`, { headers })
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setDomain)
      .catch(() => setError("Could not load domain views (are you an admin?)."));
    fetch(`${API_BASE}/api/admin/gaps`, { headers })
      .then((r) => (r.ok ? r.json() : { gaps: [] }))
      .then((d) => setGaps(d.gaps));
  }, []);

  if (!mounted) return null;
  if (error) return <main className="admin"><p className="error">{error}</p></main>;
  if (!domain) return <main className="admin"><p>Loading...</p></main>;

  return (
    <main className="admin">
      <AdminNav />
      <h1>Domain: {domain.domain}</h1>

      {(domain.mlflow_url || domain.langfuse_url) && (
        <div className="row">
          <span className="meta">Observability:</span>
          <Hint text="Business: proof the system is watched, not a black box; every answer and model run is traceable. Technical: Langfuse captures per-turn spans (retrieve, rerank, generate); MLflow logs eval runs and model versions." />
          {domain.langfuse_url && (
            <a className="ext" href={domain.langfuse_url} target="_blank" rel="noreferrer">
              Langfuse traces
            </a>
          )}
          {domain.mlflow_url && (
            <a className="ext" href={domain.mlflow_url} target="_blank" rel="noreferrer">
              MLflow runs
            </a>
          )}
        </div>
      )}

      <h2>Knowledge gaps
        <Hint text="Business: real shopper demand the assistant cannot yet meet, ranked by frequency; the highest-leverage content to add next. Technical: questions that triggered an abstain or low-grounding turn, deduped and counted from live traces." />
      </h2>
      <p className="ax-intro">
        Real questions shoppers asked that the assistant could not answer well, most frequent first.
        This is the worklist for what to teach it next: add a policy or product doc, and the gap
        closes. Not test queries, actual demand.
      </p>
      {gaps.length === 0 ? (
        <p>No unanswered questions yet.</p>
      ) : (
        <ul className="gap-list">
          {gaps.map((g) => (
            <li key={g.question}>
              <span className="gap-count">{g.count}×</span>
              <span className="gap-q">{g.question}</span>
              {g.lang && g.lang !== "en" && <span className="gap-lang">{g.lang}</span>}
            </li>
          ))}
        </ul>
      )}

      <h2>Ontology
        <Hint text="Business: the vocabulary of this domain, products, orders, policies and how they relate; swapping it re-skins the whole platform for a new industry. Technical: the entity types and typed edges declared in domain.yaml that shape the knowledge graph and retrieval." />
      </h2>
      <p className="meta">Entities: {domain.ontology.entity_types.join(", ")}</p>
      <ul>
        {domain.ontology.edges.map((e) => (
          <li key={e.type}>
            {e.from} <em>{e.type}</em> {e.to}
          </li>
        ))}
      </ul>

      <h2>Governed metrics
        <Hint text="Business: the numbers everyone agrees on, defined once so a metric means the same thing in the assistant, the dashboard, and a board deck. Technical: the semantic layer, each metric with its grain, dimensions, and params, computed from the gold medallion tables." />
      </h2>
      <ul>
        {domain.metrics.map((m) => (
          <li key={m.name}>
            <strong>{m.name}</strong> ({m.grain}) params: {m.params.join(", ") || "none"}
          </li>
        ))}
      </ul>

      <h2>Lineage
        <Hint text="Business: where every number comes from and how PII is protected on the way, the audit trail a regulator or buyer asks for. Technical: the medallion path (bronze to silver to gold), which columns are masked, and which governed metrics each layer serves." />
      </h2>
      <ul>
        {domain.lineage.medallion.map((r) => (
          <li key={r.role}>
            {r.layers.join(" → ")}
            {r.pii_columns.length > 0 && <span className="meta"> (PII: {r.pii_columns.join(", ")})</span>}
            {r.metrics.length > 0 && <span className="meta"> serves {r.metrics.join(", ")}</span>}
          </li>
        ))}
      </ul>
    </main>
  );
}
