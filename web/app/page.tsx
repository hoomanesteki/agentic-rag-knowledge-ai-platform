"use client";

import { useEffect, useRef, useState } from "react";

import { useTurnstile } from "./turnstile";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
        resetCaptcha(); // the token was consumed; a fresh one is needed for the retry
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
      <input name="username" placeholder="Username" aria-label="Username" autoComplete="username" />
      <input
        name="password"
        type="password"
        placeholder="Password"
        aria-label="Password"
        autoComplete="current-password"
      />
      {captchaWidget}
      <button type="submit" disabled={busy}>
        {busy ? "..." : "Sign in"}
      </button>
      {error && <p className="err">{error}</p>}
    </form>
  );
}

type SpeechRecognitionEventLike = {
  results: { [i: number]: { [j: number]: { transcript: string } } };
};
type SpeechRecognitionLike = {
  lang: string;
  onresult: (e: SpeechRecognitionEventLike) => void;
  onerror: () => void;
  start: () => void;
};

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve((reader.result as string).split(",")[1] || "");
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

function Chat({ token, onSignOut }: { token: string; onSignOut: () => void }) {
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [final, setFinal] = useState<FinalEvent | null>(null);
  const [loading, setLoading] = useState(false);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const recorderRef = useRef<MediaRecorder | null>(null);

  const authHeaders = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${token}`,
  };

  useEffect(() => {
    // starter prompts for the active domain, so the first screen guides instead of a blank box
    fetch(`${API_BASE}/api/suggestions`, { headers: { Authorization: `Bearer ${token}` } })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => d && setSuggestions(d.suggestions || []))
      .catch(() => {});
  }, [token]);

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    send(query);
  }

  async function send(q: string) {
    if (!q.trim() || loading) return;
    setQuery(q);
    setAnswer("");
    setFinal(null);
    setFeedback(null);
    setLoading(true);
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

  function speechFallback() {
    // browser live speech recognition, the fallback when MediaRecorder or the server STT is down
    const SR =
      (window as unknown as { SpeechRecognition?: new () => SpeechRecognitionLike }).SpeechRecognition ||
      (window as unknown as { webkitSpeechRecognition?: new () => SpeechRecognitionLike })
        .webkitSpeechRecognition;
    if (!SR) {
      setAnswer("Voice input is not available in this browser. Please type your question.");
      return;
    }
    const rec = new SR();
    // follow the browser locale so French users get French recognition, not forced en-US
    rec.lang = (typeof navigator !== "undefined" && navigator.language) || "en-US";
    rec.onresult = (e: SpeechRecognitionEventLike) => setQuery(e.results[0][0].transcript);
    rec.onerror = () => setAnswer("Could not capture voice. Please type your question.");
    rec.start();
  }

  async function toggleMic() {
    if (recording) {
      recorderRef.current?.stop();
      return;
    }
    if (typeof navigator === "undefined" || !navigator.mediaDevices ||
        typeof MediaRecorder === "undefined") {
      speechFallback(); // no MediaRecorder: use live Web Speech instead
      return;
    }
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setAnswer("Microphone unavailable or permission denied.");
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
          if (res.ok) setQuery((await res.json()).text);
          else speechFallback(); // server STT down: fall back to live Web Speech
        } catch {
          speechFallback();
        }
      };
      recorderRef.current = recorder;
      recorder.start();
      setRecording(true);
    } catch {
      stream.getTracks().forEach((t) => t.stop()); // do not leave the mic hot
      speechFallback();
    }
  }

  function speak() {
    const text = final?.answer ?? answer;
    const synth = typeof window !== "undefined" ? window.speechSynthesis : undefined;
    if (text && synth) {
      synth.cancel();
      synth.speak(new SpeechSynthesisUtterance(text));
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
        <button
          type="button"
          className={recording ? "mic recording" : "mic"}
          onClick={toggleMic}
          aria-label={recording ? "Stop recording" : "Ask by voice"}
        >
          {recording ? "Stop" : "Mic"}
        </button>
        <button type="submit" disabled={loading}>
          {loading ? "..." : "Ask"}
        </button>
      </form>

      {!shown && !loading && suggestions.length > 0 && (
        <div className="starters">
          <p className="hint">
            Try one of these, or ask your own. Every answer is grounded and cited, or it honestly
            says it does not know.
          </p>
          <div className="chips">
            {suggestions.map((s) => (
              <button
                key={s.text}
                type="button"
                className="chip-suggest"
                onClick={() => send(s.text)}
              >
                {s.text}
              </button>
            ))}
          </div>
        </div>
      )}

      {loading && !shown && <div className="thinking">Thinking...</div>}

      {shown && (
        <div className="answer" aria-live="polite">
          {shown}
          <button type="button" className="speak" onClick={speak} aria-label="Read the answer aloud">
            Speak
          </button>
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

      {final && suggestions.length > 0 && (
        <div className="followups">
          <span className="hint">Ask another</span>
          {suggestions.slice(0, 4).map((s) => (
            <button
              key={s.text}
              type="button"
              className="chip-suggest small"
              onClick={() => send(s.text)}
            >
              {s.text}
            </button>
          ))}
        </div>
      )}

      <button className="signout" onClick={onSignOut}>
        Sign out
      </button>
    </>
  );
}
