"use client";

import { useEffect, useRef, useState } from "react";

import { API_BASE, fetchCatalog, Product, swatchStyle } from "./catalog";
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
  recs?: Product[];
  followups?: string[];
};

function cap(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

// Products the assistant named in its answer, matched by catalog name, so we can show them as cards.
function recsFromAnswer(text: string, products: Product[]): Product[] {
  if (!text) return [];
  const low = text.toLowerCase();
  const seen = new Set<string>();
  const hits: Product[] = [];
  for (const p of products) {
    const full = p.name.toLowerCase();
    const short = full.replace(/^aster\s+/, "");
    if ((low.includes(full) || low.includes(short)) && !seen.has(p.id)) {
      seen.add(p.id);
      hits.push(p);
    }
  }
  return hits.slice(0, 4);
}

// Follow-ups that track the conversation: product-specific when it recommended something, and a
// gentle "here is what I can answer" when it did not know, so the shopper always has a next step.
function followupsFor(final: FinalEvent, text: string, recs: Product[], suggestions: Suggestion[]): string[] {
  const unsure =
    final.tier !== "auto" ||
    /(don't|do not|couldn't|could not|cannot) (have|find|see)|not sure|no information|don't know/i.test(text);
  if (unsure || recs.length === 0) return suggestions.slice(0, 3).map((s) => s.text);
  const p = recs[0];
  const short = p.name.replace(/^Aster\s+/, "");
  const byCat: Record<string, string> = {
    jackets: `What should I layer under the ${short} when it's cold?`,
    leggings: `Is the ${short} squat proof?`,
    bras: `Is the ${short} high support?`,
    tops: `Is the ${short} good for hot yoga?`,
    bottoms: `Is the ${short} okay for travel?`,
    shorts: `Does the ${short} have pockets?`,
    bags: `What fits in the ${short}?`,
    hoodies: `Does the ${short} run oversized?`,
    accessories: `Is the ${short} warm enough for winter?`,
  };
  const out = [`Does the ${short} run true to size?`];
  if (byCat[p.category]) out.push(byCat[p.category]);
  out.push("What is your return policy?");
  return out;
}

// Minimal shape of the browser SpeechRecognition, which handles endpointing (knows when you stop
// talking) so the voice loop stays conversational without us building silence detection.
type SpeechResult = { results: { [i: number]: { [j: number]: { transcript: string } } } };
type Recognizer = {
  lang: string;
  interimResults: boolean;
  maxAlternatives: number;
  onresult: (e: SpeechResult) => void;
  onerror: () => void;
  onend: () => void;
  start: () => void;
  abort: () => void;
};
function makeRecognizer(): Recognizer | null {
  if (typeof window === "undefined") return null;
  const SR =
    (window as unknown as { SpeechRecognition?: new () => Recognizer }).SpeechRecognition ||
    (window as unknown as { webkitSpeechRecognition?: new () => Recognizer }).webkitSpeechRecognition;
  return SR ? new SR() : null;
}
// Speak text and resolve when the voice finishes, so the loop waits before listening again.
function speakAsync(text: string): Promise<void> {
  return new Promise((resolve) => {
    const synth = typeof window !== "undefined" ? window.speechSynthesis : undefined;
    if (!text || !synth) {
      resolve();
      return;
    }
    synth.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.onend = () => resolve();
    u.onerror = () => resolve();
    synth.speak(u);
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
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [products, setProducts] = useState<Product[]>([]);
  const [voiceOn, setVoiceOn] = useState(false);
  const [voiceState, setVoiceState] = useState<"greeting" | "listening" | "thinking" | "speaking">(
    "greeting",
  );
  const [heard, setHeard] = useState("");
  const streamRef = useRef<HTMLDivElement | null>(null);
  const seededRef = useRef<string | null>(null);
  const recogRef = useRef<Recognizer | null>(null);
  const voiceLiveRef = useRef(false); // lets stopVoice break the async loop

  const authHeaders = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${token}`,
  };

  useEffect(() => {
    fetch(`${API_BASE}/api/suggestions`, { headers: { Authorization: `Bearer ${token}` } })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => d && setSuggestions(d.suggestions || []))
      .catch(() => {});
    fetchCatalog().then(setProducts); // so answers can surface product cards
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

  async function send(q: string): Promise<string> {
    if (!q.trim() || loading) return "";
    setInput("");
    setMessages((m) => [...m, { role: "me", text: q }, { role: "bot", text: "" }]);
    setLoading(true);
    let result = "";
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
        return "";
      }
      if (res.status === 429) {
        const m = "You are asking too fast. Please wait a moment.";
        patchBot((b) => ({ ...b, text: m }));
        return m;
      }
      if (!res.ok || !res.body) {
        patchBot((b) => ({ ...b, text: "Sorry, something went wrong. Please try again." }));
        return "";
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let received = false;
      let acc = "";
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
            if (event.type === "token") {
              acc += event.text;
              patchBot((b) => ({ ...b, text: b.text + event.text }));
            } else if (event.type === "final") {
              const fe = event;
              const answer = fe.answer ?? acc;
              result = answer;
              const recs = recsFromAnswer(answer, products);
              patchBot((b) => ({
                ...b,
                text: fe.answer ?? b.text,
                final: fe,
                recs,
                followups: followupsFor(fe, answer, recs, suggestions),
              }));
            } else if (event.type === "error")
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
    return result;
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

  // Listen for one utterance and resolve with the transcript. The browser recognizer decides when
  // the speaker has stopped, which keeps the loop hands-free.
  function listenOnce(): Promise<string> {
    return new Promise((resolve) => {
      const rec = makeRecognizer();
      if (!rec) {
        resolve("");
        return;
      }
      rec.lang = (typeof navigator !== "undefined" && navigator.language) || "en-US";
      rec.interimResults = false;
      rec.maxAlternatives = 1;
      let said = "";
      rec.onresult = (e) => {
        said = e.results[0][0].transcript;
      };
      rec.onerror = () => resolve("");
      rec.onend = () => resolve(said);
      recogRef.current = rec;
      try {
        rec.start();
      } catch {
        resolve("");
      }
    });
  }

  // The conversation loop: greet, then listen -> answer -> speak, over and over, until Stop.
  async function startVoice() {
    if (!makeRecognizer()) {
      setMessages((m) => [
        ...m,
        { role: "bot", text: "Voice chat is not supported in this browser. Please type instead." },
      ]);
      return;
    }
    setVoiceOn(true);
    voiceLiveRef.current = true;
    setHeard("");
    setVoiceState("greeting");
    await speakAsync("Hi, I'm your Aster assistant. How can I help you today?");
    let quiet = 0;
    while (voiceLiveRef.current) {
      setVoiceState("listening");
      const said = await listenOnce();
      if (!voiceLiveRef.current) break;
      if (!said.trim()) {
        // nothing heard (silence, or the mic was blocked): give up after a few tries, do not spin
        if (++quiet >= 3) {
          await speakAsync("I did not catch that. Tap the mic when you are ready.");
          break;
        }
        continue;
      }
      quiet = 0;
      setHeard(said);
      setVoiceState("thinking");
      const answer = await send(said);
      if (!voiceLiveRef.current) break;
      setVoiceState("speaking");
      await speakAsync(answer);
    }
    voiceLiveRef.current = false;
    setVoiceOn(false);
  }

  function stopVoice() {
    voiceLiveRef.current = false;
    recogRef.current?.abort();
    if (typeof window !== "undefined") window.speechSynthesis?.cancel();
    setVoiceOn(false);
  }

  useEffect(() => {
    // stop the mic and any speech if the widget unmounts mid-conversation
    return () => {
      voiceLiveRef.current = false;
      recogRef.current?.abort();
      if (typeof window !== "undefined") window.speechSynthesis?.cancel();
    };
  }, []);

  const empty = messages.length === 0;
  const voiceLabel =
    voiceState === "greeting"
      ? "Say hello..."
      : voiceState === "listening"
        ? "Listening..."
        : voiceState === "thinking"
          ? "Thinking..."
          : "Speaking...";

  if (voiceOn) {
    return (
      <div className="voice">
        <div>
          <div
            className={`orb ${voiceState === "listening" ? "listening" : ""}${
              voiceState === "speaking" ? " speaking" : ""
            }`}
          />
          <div className="state">{voiceLabel}</div>
          {heard && <div className="said">&ldquo;{heard}&rdquo;</div>}
        </div>
        <button className="btn btn-primary" onClick={stopVoice}>
          Stop
        </button>
      </div>
    );
  }

  return (
    <>
      <div className="stream" ref={streamRef}>
        {empty && (
          <div className="greet">
            <div className="big">Hi, I&apos;m your Aster assistant.</div>
            How can I help you today? Ask about a product, sizing, shipping, or what to wear, or tap
            the mic to talk.
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
                    {m.feedback && <span>thanks</span>}
                  </span>
                )}
              </div>
            )}
            {m.recs && m.recs.length > 0 && (
              <div className="recs">
                {m.recs.map((p) => (
                  <button
                    key={p.id}
                    className="rec"
                    onClick={() => send(`Tell me more about the ${p.name}.`)}
                  >
                    <div className="sw" style={swatchStyle(p.color)} />
                    <div className="rb">
                      <div className="rn">{p.name.replace(/^Aster /, "")}</div>
                      <div className="rp">
                        {cap(p.category)}
                        {p.price != null ? ` · $${p.price.toFixed(0)}` : ""}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}
            {m.followups && m.followups.length > 0 && (
              <div className="sugg">
                {m.followups.map((f, k) => (
                  <button key={k} type="button" onClick={() => send(f)}>
                    {f}
                  </button>
                ))}
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
          className="icon-btn"
          onClick={startVoice}
          aria-label="Start a voice conversation"
          title="Talk to the assistant"
        >
          🎙
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
