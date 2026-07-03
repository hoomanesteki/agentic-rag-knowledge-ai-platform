"use client";

import { useEffect, useRef, useState } from "react";

import { API_BASE } from "./catalog";
import { useTurnstile } from "./turnstile";

type Citation = { n: number; id: string; doc_type?: string | null };
type Suggestion = { text: string; lang: string; kind: string };

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

type Message = {
  role: "me" | "bot";
  text: string;
  final?: FinalEvent;
  feedback?: "up" | "down";
};

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve((reader.result as string).split(",")[1] || "");
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

export default function ChatWidget({
  open,
  setOpen,
  seed,
}: {
  open: boolean;
  setOpen: (v: boolean) => void;
  seed?: string | null;
}) {
  const [mounted, setMounted] = useState(false);
  const [token, setToken] = useState<string | null>(null);

  useEffect(() => {
    setMounted(true);
    setToken(localStorage.getItem("skein_token"));
  }, []);

  function onToken(t: string) {
    localStorage.setItem("skein_token", t);
    setToken(t);
  }
  function signOut() {
    localStorage.removeItem("skein_token");
    setToken(null);
  }

  if (!mounted) return null;

  return (
    <>
      {!open && (
        <button className="fab" onClick={() => setOpen(true)} aria-label="Open the Aster assistant">
          <span className="dot" />
          &#128172;
        </button>
      )}
      {open && (
        <section className="panel" role="dialog" aria-label="Aster assistant">
          <header className="panel-hdr">
            <div className="avatar">A</div>
            <div>
              <div className="t">Aster Assistant</div>
              <div className="s">Grounded answers, always cited</div>
            </div>
            <button className="x" onClick={() => setOpen(false)} aria-label="Close">
              &times;
            </button>
          </header>
          {token ? (
            <Conversation token={token} onSignOut={signOut} seed={seed} />
          ) : (
            <Login onToken={onToken} />
          )}
        </section>
      )}
    </>
  );
}

function Login({ onToken }: { onToken: (t: string) => void }) {
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const { token: captchaToken, widget: captchaWidget, reset: resetCaptcha } = useTurnstile();

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
        resetCaptcha();
      } else if (res.status === 403) {
        setError("Captcha check failed. Please try again.");
        resetCaptcha();
      } else {
        setError("Could not sign in. Please try again.");
        resetCaptcha();
      }
    } catch {
      setError("Could not reach the server.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="login" onSubmit={submit}>
      <p>Sign in to chat with the assistant about products, sizing, and policies.</p>
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
        {busy ? "..." : "Sign in"}
      </button>
      {error && <p className="err">{error}</p>}
    </form>
  );
}

function Conversation({
  token,
  onSignOut,
  seed,
}: {
  token: string;
  onSignOut: () => void;
  seed?: string | null;
}) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<HTMLDivElement | null>(null);
  const seededRef = useRef<string | null>(null);

  const authHeaders = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${token}`,
  };

  useEffect(() => {
    fetch(`${API_BASE}/api/suggestions`, { headers: { Authorization: `Bearer ${token}` } })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => d && setSuggestions(d.suggestions || []))
      .catch(() => {});
  }, [token]);

  useEffect(() => {
    // keep the newest message in view as tokens stream in
    streamRef.current?.scrollTo({ top: streamRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    // a product the shopper clicked in the store seeds the input, once
    if (seed && seededRef.current !== seed) {
      seededRef.current = seed;
      setInput(seed);
    }
  }, [seed]);

  async function send(q: string) {
    if (!q.trim() || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "me", text: q }, { role: "bot", text: "" }]);
    setLoading(true);
    const patchBot = (fn: (m: Message) => Message) =>
      setMessages((all) => {
        const copy = [...all];
        for (let i = copy.length - 1; i >= 0; i--) {
          if (copy[i].role === "bot") {
            copy[i] = fn(copy[i]);
            break;
          }
        }
        return copy;
      });
    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: authHeaders,
        body: JSON.stringify({ query: q }),
      });
      if (res.status === 401) {
        onSignOut();
        return;
      }
      if (res.status === 429) {
        patchBot((b) => ({ ...b, text: "You are asking too fast. Please wait a moment." }));
        return;
      }
      if (!res.ok || !res.body) {
        patchBot((b) => ({ ...b, text: "Sorry, something went wrong. Please try again." }));
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
            if (event.type === "token") patchBot((b) => ({ ...b, text: b.text + event.text }));
            else if (event.type === "final")
              patchBot((b) => ({ ...b, text: event.answer ?? b.text, final: event }));
            else if (event.type === "error")
              patchBot((b) => ({ ...b, text: "Sorry, something went wrong." }));
          }
        }
      } finally {
        reader.cancel().catch(() => {});
      }
      if (!received) patchBot((b) => ({ ...b, text: "No response received. Please try again." }));
    } catch {
      patchBot((b) => ({ ...b, text: "Could not reach the assistant. Is the API running?" }));
    } finally {
      setLoading(false);
    }
  }

  async function sendFeedback(idx: number, verdict: "up" | "down") {
    const msg = messages[idx];
    if (!msg?.final?.message_id || msg.feedback) return;
    setMessages((all) => all.map((m, i) => (i === idx ? { ...m, feedback: verdict } : m)));
    try {
      await fetch(`${API_BASE}/api/feedback`, {
        method: "POST",
        headers: authHeaders,
        body: JSON.stringify({ message_id: msg.final.message_id, verdict }),
      });
    } catch {
      /* best effort */
    }
  }

  async function toggleMic() {
    if (recording) {
      recorderRef.current?.stop();
      return;
    }
    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices ||
      typeof MediaRecorder === "undefined"
    ) {
      return;
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      return;
    }
    try {
      const recorder = new MediaRecorder(stream);
      const chunks: BlobPart[] = [];
      recorder.ondataavailable = (e) => chunks.push(e.data);
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        setRecording(false);
        recorderRef.current = null;
        const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
        try {
          const res = await fetch(`${API_BASE}/api/transcribe`, {
            method: "POST",
            headers: authHeaders,
            body: JSON.stringify({ audio_base64: await blobToBase64(blob), mime: blob.type }),
          });
          if (res.ok) setInput((await res.json()).text);
        } catch {
          /* transcription is best-effort */
        }
      };
      recorderRef.current = recorder;
      recorder.start();
      setRecording(true);
    } catch {
      stream.getTracks().forEach((t) => t.stop());
    }
  }

  const empty = messages.length === 0;

  return (
    <>
      <div className="stream" ref={streamRef}>
        {empty && (
          <div className="greet">
            <div className="big">Hi, I&apos;m your Aster assistant.</div>
            How can I help you today? Ask about a product, sizing, shipping, or what to wear.
          </div>
        )}
        {empty && suggestions.length > 0 && (
          <div className="sugg">
            {suggestions.slice(0, 5).map((s, i) => (
              <button key={`${i}-${s.text}`} type="button" onClick={() => send(s.text)}>
                {s.text}
              </button>
            ))}
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} style={{ display: "contents" }}>
            <div className={`msg ${m.role}${m.role === "bot" && !m.text ? " think" : ""}`}>
              {m.text || (loading ? "Thinking..." : "")}
            </div>
            {m.final && (
              <div className="meta-row">
                <span className={`tier tier-${m.final.tier}`}>{m.final.tier}</span>
                {m.final.citations?.slice(0, 4).map((c) => (
                  <span key={c.n} className="cite">
                    [{c.n}] {c.id}
                  </span>
                ))}
                {m.final.message_id && (
                  <span className="thumbs">
                    <button
                      aria-label="Helpful"
                      disabled={!!m.feedback}
                      onClick={() => sendFeedback(i, "up")}
                    >
                      &#128077;
                    </button>
                    <button
                      aria-label="Not helpful"
                      disabled={!!m.feedback}
                      onClick={() => sendFeedback(i, "down")}
                    >
                      &#128078;
                    </button>
                  </span>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <button
          type="button"
          className={recording ? "icon-btn rec" : "icon-btn"}
          onClick={toggleMic}
          aria-label={recording ? "Stop recording" : "Speak"}
        >
          {recording ? "■" : "🎤"}
        </button>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about a product, sizing, weather..."
          aria-label="Your question"
        />
        <button type="submit" className="icon-btn send" disabled={loading} aria-label="Send">
          &#8593;
        </button>
      </form>
    </>
  );
}
