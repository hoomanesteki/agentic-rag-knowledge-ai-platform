"use client";

import { useEffect, useRef, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const SITE_KEY = process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY;

type Turnstile = {
  render: (el: HTMLElement, opts: Record<string, unknown>) => string;
  remove: (id: string) => void;
};

function turnstileApi(): Turnstile | undefined {
  return (window as unknown as { turnstile?: Turnstile }).turnstile;
}

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
  const [mounted, setMounted] = useState(false);
  const [token, setToken] = useState<string | null>(null);

  useEffect(() => {
    setMounted(true);
    setToken(localStorage.getItem("skein_token"));
  }, []);

  if (!mounted) return null; // avoid SSR/client hydration mismatch on the stored token

  function onToken(t: string) {
    localStorage.setItem("skein_token", t);
    setToken(t);
  }
  function signOut() {
    localStorage.removeItem("skein_token");
    setToken(null);
  }

  return (
    <main className="wrap">
      <h1>Skein</h1>
      <p className="sub">
        Grounded, cited answers over the catalog. When it is unsure, it says so.
      </p>
      {token ? <Chat token={token} onSignOut={signOut} /> : <Login onToken={onToken} />}
    </main>
  );
}

function Login({ onToken }: { onToken: (t: string) => void }) {
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [captchaToken, setCaptchaToken] = useState<string | null>(null);
  const captchaRef = useRef<HTMLDivElement | null>(null);
  const widgetId = useRef<string | undefined>(undefined);

  useEffect(() => {
    if (!SITE_KEY) return;
    function render() {
      const ts = turnstileApi();
      if (!ts || !captchaRef.current || widgetId.current !== undefined) return;
      widgetId.current = ts.render(captchaRef.current, {
        sitekey: SITE_KEY,
        callback: (t: string) => setCaptchaToken(t),
        "expired-callback": () => setCaptchaToken(null),
      });
    }
    if (turnstileApi()) {
      render(); // script already loaded (e.g. a remount after sign-out)
    } else {
      let script = document.querySelector("script[data-turnstile]") as HTMLScriptElement | null;
      if (!script) {
        script = document.createElement("script");
        script.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
        script.async = true;
        script.defer = true;
        script.setAttribute("data-turnstile", "1");
        document.head.appendChild(script);
      }
      script.addEventListener("load", render, { once: true });
    }
    return () => {
      const ts = turnstileApi();
      if (ts && widgetId.current !== undefined) {
        try {
          ts.remove(widgetId.current);
        } catch {
          /* widget already gone */
        }
        widgetId.current = undefined;
      }
    };
  }, []);

  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (busy) return;
    setError(null);
    setBusy(true);
    const data = new FormData(e.currentTarget);
    try {
      const res = await fetch(`${API_BASE}/api/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: data.get("username"),
          password: data.get("password"),
          turnstile_token: captchaToken || undefined,
        }),
      });
      if (res.ok) {
        onToken((await res.json()).access_token);
      } else if (res.status === 401) {
        setError("Wrong username or password.");
      } else if (res.status === 403) {
        setError("Captcha check failed. Please try again.");
      } else {
        setError("Could not sign in. Please try again.");
      }
    } catch {
      setError("Could not reach the server.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="login" onSubmit={submit}>
      <input name="username" placeholder="Username" aria-label="Username" autoComplete="username" />
      <input
        name="password"
        type="password"
        placeholder="Password"
        aria-label="Password"
        autoComplete="current-password"
      />
      {SITE_KEY && <div ref={captchaRef} />}
      <button type="submit" disabled={busy}>
        {busy ? "..." : "Sign in"}
      </button>
      {error && <p className="err">{error}</p>}
    </form>
  );
}

function Chat({ token, onSignOut }: { token: string; onSignOut: () => void }) {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [final, setFinal] = useState<FinalEvent | null>(null);
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);

  const authHeaders = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${token}`,
  };

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
        headers: authHeaders,
        body: JSON.stringify({ query }),
      });
      if (res.status === 401) {
        onSignOut();
        return;
      }
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
              continue;
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
        headers: authHeaders,
        body: JSON.stringify({ message_id: final.message_id, verdict }),
      });
    } catch {
      /* feedback is best-effort */
    }
  }

  const shown = final?.answer ?? answer;

  return (
    <>
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

      <button className="signout" onClick={onSignOut}>
        Sign out
      </button>
    </>
  );
}
