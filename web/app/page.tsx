"use client";

import { useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Citation = { n: number; id: string; doc_type?: string | null };

type FinalEvent = {
  type: "final";
  message_id?: string;
  tier: string;
  answer?: string;
  confidence?: number;
  grounding?: number;
  citations?: Citation[];
};

type StreamEvent =
  | { type: "token"; text: string }
  | FinalEvent
  | { type: "error"; message?: string };

export default function Home() {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [final, setFinal] = useState<FinalEvent | null>(null);
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim() || loading) return;
    setAnswer("");
    setFinal(null);
    setFeedback(null);
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      if (res.status === 429) {
        setAnswer("You are asking too fast. Please wait a moment and try again.");
        return;
      }
      if (!res.ok || !res.body) {
        setAnswer("Sorry, something went wrong. Please try again.");
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let received = false;
      try {
        for (;;) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const chunks = buffer.split(/\r?\n\r?\n/);
          buffer = chunks.pop() ?? "";
          for (const chunk of chunks) {
            const line = chunk.trim();
            if (!line.startsWith("data:")) continue;
            let event: StreamEvent;
            try {
              event = JSON.parse(line.slice(5).trim()) as StreamEvent;
            } catch {
              continue; // skip a malformed frame, keep the stream alive
            }
            received = true;
            if (event.type === "token") setAnswer((a) => a + event.text);
            else if (event.type === "final") setFinal(event);
            else if (event.type === "error") setAnswer("Sorry, something went wrong.");
          }
        }
      } finally {
        reader.cancel().catch(() => {});
      }
      if (!received) setAnswer("Sorry, no response was received. Please try again.");
    } catch {
      setAnswer("Could not reach the assistant. Is the API running?");
    } finally {
      setLoading(false);
    }
  }

  async function sendFeedback(verdict: "up" | "down") {
    if (!final?.message_id) return;
    setFeedback(verdict);
    try {
      await fetch(`${API_BASE}/api/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message_id: final.message_id, verdict }),
      });
    } catch {
      /* feedback is best-effort */
    }
  }

  // On a degraded/abstain final the streamed text may differ from the final answer; trust final.
  const shown = final?.answer ?? answer;

  return (
    <main className="wrap">
      <h1>Skein</h1>
      <p className="sub">
        Ask about the catalog. Every answer is grounded and cited, or it says it does not know.
      </p>

      <form className="ask" onSubmit={ask}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="What do customers say about sizing?"
          aria-label="Your question"
        />
        <button type="submit" disabled={loading}>
          {loading ? "..." : "Ask"}
        </button>
      </form>

      {shown && (
        <div className="answer" aria-live="polite">
          {shown}
        </div>
      )}

      {final && (
        <div className="meta">
          <span className={`tier tier-${final.tier}`}>{final.tier}</span>
          {typeof final.confidence === "number" && (
            <span>confidence {final.confidence.toFixed(2)}</span>
          )}
          {final.citations && final.citations.length > 0 && (
            <div className="cites">
              {final.citations.map((c) => (
                <span key={c.n} className="chip">
                  [{c.n}] {c.id}
                </span>
              ))}
            </div>
          )}
          {final.message_id && (
            <div className="thumbs">
              <button aria-label="Helpful" disabled={!!feedback} onClick={() => sendFeedback("up")}>
                &#128077;
              </button>
              <button
                aria-label="Not helpful"
                disabled={!!feedback}
                onClick={() => sendFeedback("down")}
              >
                &#128078;
              </button>
              {feedback && <span>thanks</span>}
            </div>
          )}
        </div>
      )}
    </main>
  );
}
