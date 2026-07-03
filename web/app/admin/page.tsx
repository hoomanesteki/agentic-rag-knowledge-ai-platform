"use client";

import { useEffect, useState } from "react";

import { useTurnstile } from "../turnstile";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Item = {
  id: string;
  question: string;
  route: string | null;
  domain: string | null;
  status: string;
};

// The login lives in its own component so it mounts together with the Turnstile widget div. If the
// hook ran on a parent that first renders null (the mounted gate), its one-shot effect would fire
// against a missing ref and the widget would never appear, making production admin login impossible.
function AdminLogin({ onToken }: { onToken: (t: string) => void }) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const { token: captchaToken, widget: captchaWidget, reset: resetCaptcha } = useTurnstile();

  async function login(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password, turnstile_token: captchaToken || undefined }),
      });
      if (res.status === 403) {
        setError("Captcha check failed. Please try again.");
        resetCaptcha();
        return;
      }
      if (!res.ok) {
        setError("Login failed.");
        resetCaptcha(); // the captcha token was consumed; the retry needs a fresh one
        return;
      }
      onToken((await res.json()).access_token);
    } catch {
      setError("Could not reach the server.");
    }
  }

  return (
    <main className="admin">
      <h1>Review queue</h1>
      <form onSubmit={login} className="login">
        <input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="admin" />
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="password"
        />
        {captchaWidget}
        <button type="submit">Sign in</button>
        {error && <p className="error">{error}</p>}
      </form>
    </main>
  );
}

export default function AdminPage() {
  const [mounted, setMounted] = useState(false);
  const [token, setToken] = useState<string | null>(null);
  const [items, setItems] = useState<Item[]>([]);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [flywheel, setFlywheel] = useState<string | null>(null);

  useEffect(() => {
    setMounted(true);
    setToken(localStorage.getItem("skein_admin_token"));
  }, []);

  const auth = { Authorization: `Bearer ${token}`, "Content-Type": "application/json" };

  function signOut() {
    localStorage.removeItem("skein_admin_token");
    setToken(null); // re-renders AdminLogin
  }

  async function loadQueue(t = token) {
    setError(null);
    const res = await fetch(`${API_BASE}/api/admin/queue`, {
      headers: { Authorization: `Bearer ${t}` },
    });
    if (res.status === 401) {
      signOut(); // the token expired; send the operator back to the login instead of a dead page
      return;
    }
    if (res.status === 403) {
      setError("This account is not an admin.");
      return;
    }
    if (!res.ok) {
      setError("Could not load the queue.");
      return;
    }
    setItems((await res.json()).items);
  }

  useEffect(() => {
    if (token) loadQueue();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  function onToken(t: string) {
    localStorage.setItem("skein_admin_token", t);
    setToken(t);
  }

  async function claim(id: string) {
    const res = await fetch(`${API_BASE}/api/admin/queue/${id}/claim`, {
      method: "POST",
      headers: auth,
    });
    if (!res.ok) setError("Could not claim (someone else took it).");
    loadQueue();
  }

  async function answer(id: string) {
    const body = JSON.stringify({ answer: answers[id] || "" });
    const res = await fetch(`${API_BASE}/api/admin/queue/${id}/answer`, {
      method: "POST",
      headers: auth,
      body,
    });
    if (res.ok) loadQueue();
    else setError("Could not submit the answer (claim it first, or it was taken).");
  }

  async function runFlywheel() {
    setFlywheel("Running...");
    const res = await fetch(`${API_BASE}/api/admin/flywheel`, { method: "POST", headers: auth });
    if (res.status === 401) {
      signOut();
      return;
    }
    if (!res.ok) {
      setFlywheel("Flywheel failed.");
      return;
    }
    const r = await res.json();
    setFlywheel(`Indexed ${r.indexed} answer(s), grew eval by ${r.grown}. Suggested gate: ${r.threshold.suggested}.`);
  }

  if (!mounted) return null;

  if (!token) return <AdminLogin onToken={onToken} />;

  return (
    <main className="admin">
      <h1>Review queue ({items.length})</h1>
      <div className="row">
        <button onClick={runFlywheel}>Run flywheel</button>
        <button className="signout" onClick={signOut}>Sign out</button>
        {flywheel && <span className="meta">{flywheel}</span>}
      </div>
      {error && <p className="error">{error}</p>}
      {items.length === 0 && <p>No open questions. The system is confident right now.</p>}
      {items.map((it) => (
        <div key={it.id} className="queue-item">
          <p className="q">{it.question}</p>
          <p className="meta">
            {it.domain} · {it.route || "unrouted"}
          </p>
          <div className="row">
            {/* Claim is always offered: on an open item it locks it, on a stale claimed one it
                takes it over (the server rejects a live claim by someone else). */}
            <button onClick={() => claim(it.id)}>Claim</button>
          </div>
          {it.status === "claimed" && (
            <>
              <textarea
                value={answers[it.id] || ""}
                onChange={(e) => setAnswers({ ...answers, [it.id]: e.target.value })}
                placeholder="Write the verified answer"
              />
              <div className="row">
                <button onClick={() => answer(it.id)}>Answer and close</button>
              </div>
            </>
          )}
        </div>
      ))}
    </main>
  );
}
