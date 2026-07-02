"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Item = {
  id: string;
  question: string;
  route: string | null;
  domain: string | null;
  status: string;
};

export default function AdminPage() {
  const [mounted, setMounted] = useState(false);
  const [token, setToken] = useState<string | null>(null);
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [items, setItems] = useState<Item[]>([]);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [flywheel, setFlywheel] = useState<string | null>(null);

  useEffect(() => {
    setMounted(true);
    setToken(localStorage.getItem("skein_admin_token"));
  }, []);

  const auth = { Authorization: `Bearer ${token}`, "Content-Type": "application/json" };

  async function loadQueue(t = token) {
    setError(null);
    const res = await fetch(`${API_BASE}/api/admin/queue`, {
      headers: { Authorization: `Bearer ${t}` },
    });
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

  async function login(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const res = await fetch(`${API_BASE}/api/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    if (!res.ok) {
      setError("Login failed.");
      return;
    }
    const t = (await res.json()).access_token;
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
    if (!res.ok) {
      setFlywheel("Flywheel failed.");
      return;
    }
    const r = await res.json();
    setFlywheel(`Indexed ${r.indexed} answer(s), grew eval by ${r.grown}. Suggested gate: ${r.threshold.suggested}.`);
  }

  if (!mounted) return null;

  if (!token) {
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
          <button type="submit">Sign in</button>
          {error && <p className="error">{error}</p>}
        </form>
      </main>
    );
  }

  return (
    <main className="admin">
      <h1>Review queue ({items.length})</h1>
      <div className="row">
        <button onClick={runFlywheel}>Run flywheel</button>
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
          {it.status === "open" ? (
            <div className="row">
              <button onClick={() => claim(it.id)}>Claim</button>
            </div>
          ) : (
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
