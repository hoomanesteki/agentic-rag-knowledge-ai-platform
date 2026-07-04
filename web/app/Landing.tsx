"use client";

import { useEffect, useRef, useState } from "react";

import { API_BASE } from "./catalog";
import { setGate } from "./gate";
import { useTurnstile } from "./turnstile";

// Reveal children on scroll for a bit of motion without a library.
function useReveal() {
  const ref = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const els = ref.current?.querySelectorAll(".rv");
    if (!els) return;
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => e.isIntersecting && e.target.classList.add("in")),
      { threshold: 0.15 },
    );
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, []);
  return ref;
}

const FLOW = [
  { n: "01", t: "Sources", d: "Product catalog, descriptions, reviews, policies. Structured + unstructured, EN & FR.", tech: ["CSV", "JSONL"] },
  { n: "02", t: "Ingest & process", d: "Chunk, embed, and encode sparse BM25. A medallion lakehouse governs the metrics.", tech: ["Cohere embed-v4", "dbt", "DuckDB"] },
  { n: "03", t: "Stores", d: "Hybrid vectors, a knowledge graph of products and suppliers, governed metric tables.", tech: ["Qdrant", "Neo4j", "DuckDB"] },
  { n: "04", t: "Retrieve", d: "Hybrid dense + sparse search, reranked, fused with graph facts and governed metrics.", tech: ["Cohere rerank-v3.5", "BM25", "Graph"] },
  { n: "05", t: "Reason", d: "An agentic brain understands, dispatches specialists, reconciles, and gates for grounding. Cites every claim, abstains when unsure.", tech: ["LangGraph", "Groq · Llama 3.3 70B"] },
  { n: "06", t: "Observe & serve", d: "Every turn traced and evaluated. Streamed to a storefront with a voice assistant.", tech: ["Langfuse", "RAGAS", "FastAPI", "Next.js"] },
];

const STACK = [
  "Cohere embeddings", "Cohere rerank", "Groq · Llama 3.3 70B", "Whisper STT",
  "Qdrant", "Neo4j", "DuckDB", "LangChain", "LangGraph", "Langfuse", "MLflow",
  "RAGAS", "FastAPI", "Next.js",
];

const FEATURES = [
  { t: "Grounded & cited", d: "Every answer is built from retrieved context with inline citations, and it abstains instead of guessing." },
  { t: "Knowledge graph + metrics", d: "Beyond text: a product graph and governed metric tables answer relationship and number questions." },
  { t: "Talk to it", d: "A hands-free voice mode you can interrupt mid-answer, like a real conversation." },
  { t: "Domain-swappable", d: "One manifest drives the whole engine. Swap the pack and the same engine serves any domain a pack describes." },
];

export default function Landing({ onEnter }: { onEnter: () => void }) {
  const ref = useReveal();
  const [brand, setBrand] = useState("Aster");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const { token: captchaToken, widget: captchaWidget, reset: resetCaptcha } = useTurnstile();

  useEffect(() => {
    fetch(`${API_BASE}/api/catalog`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => d?.brand && setBrand(d.brand.split(" ")[0]))
      .catch(() => {});
  }, []);

  async function enter(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (busy) return;
    setError(null);
    setBusy(true);
    const data = new FormData(e.currentTarget);
    try {
      const res = await fetch(`${API_BASE}/api/gate-login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: data.get("username"),
          password: data.get("password"),
          turnstile_token: captchaToken || undefined,
        }),
      });
      if (res.ok) {
        setGate((await res.json()).access_token);
        onEnter();
      } else if (res.status === 401) {
        setError("That username or password is not right.");
        resetCaptcha();
      } else if (res.status === 403) {
        setError("Please complete the captcha and try again.");
        resetCaptcha();
      } else if (res.status === 429) {
        setError("Too many attempts. Please wait a moment.");
      } else {
        setError("Could not verify. Please try again.");
        resetCaptcha();
      }
    } catch {
      setError("Could not reach the server. Is the API running?");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="lp" ref={ref}>
      <header className="lp-nav">
        <div className="lp-brand">
          {brand}
          <span>.</span> <em>agentic RAG</em>
        </div>
        <a className="lp-cta-sm" href="#gate">
          See the demo
        </a>
      </header>

      <section className="lp-hero">
        <div className="rv">
          <div className="lp-kicker">Retrieval-augmented · knowledge graph · agentic</div>
          <h1>
            An assistant that actually <span>knows the catalog</span>.
          </h1>
          <p>
            A domain-swappable agentic RAG platform. Shown here as an apparel store, like Lululemon,
            but the same engine serves any domain a pack describes. Grounded, cited, and honest when
            it does not know.
          </p>
          <div className="lp-hero-cta">
            <a className="btn btn-primary" href="#gate">
              See the live demo
            </a>
            <a className="btn btn-ghost lp-ghost" href="#how">
              How it works
            </a>
          </div>
        </div>
        <div className="lp-orbwrap rv">
          <div className="lp-orb" />
          <div className="lp-orb-ring" />
        </div>
      </section>

      <section className="lp-features">
        {FEATURES.map((f) => (
          <div key={f.t} className="lp-feat rv">
            <h3>{f.t}</h3>
            <p>{f.d}</p>
          </div>
        ))}
      </section>

      <section className="lp-how" id="how">
        <div className="lp-sec-head rv">
          <div className="lp-kicker">The pipeline</div>
          <h2>From raw data to a cited answer</h2>
        </div>
        <div className="lp-flow">
          {FLOW.map((s, i) => (
            <div key={s.n} className="lp-stage rv" style={{ transitionDelay: `${i * 60}ms` }}>
              <div className="lp-stage-n">{s.n}</div>
              <h4>{s.t}</h4>
              <p>{s.d}</p>
              <div className="lp-tags">
                {s.tech.map((t) => (
                  <span key={t}>{t}</span>
                ))}
              </div>
              {i < FLOW.length - 1 && <div className="lp-arrow">→</div>}
            </div>
          ))}
        </div>
      </section>

      <section className="lp-stack rv">
        <div className="lp-kicker">Built with</div>
        <div className="lp-badges">
          {STACK.map((s) => (
            <span key={s} className="lp-badge">
              {s}
            </span>
          ))}
        </div>
      </section>

      <section className="lp-gate" id="gate">
        <div className="lp-gate-card rv">
          <div className="lp-kicker">Private demo</div>
          <h2>Enter the live demo</h2>
          <p>
            This demo is private. Use the username and password from the email you received (they are
            next to this link). A captcha keeps bots out.
          </p>
          <form className="lp-gate-form" onSubmit={enter}>
            <input name="username" placeholder="Username" aria-label="Username" autoComplete="username" />
            <input
              name="password"
              type="password"
              placeholder="Password"
              aria-label="Password"
              autoComplete="current-password"
            />
            {captchaWidget}
            <button type="submit" className="btn btn-primary" disabled={busy}>
              {busy ? "Verifying..." : "Enter demo"}
            </button>
            {error && <p className="err">{error}</p>}
          </form>
          <p className="lp-gate-note">
            No credentials? Reply to the email that shared this link to request access.
          </p>
        </div>
      </section>

      <footer className="lp-foot">
        <span>Built by Aaron (Hooman) Esteki</span>
        <span>Synthetic data · no real products or people</span>
      </footer>
    </div>
  );
}
