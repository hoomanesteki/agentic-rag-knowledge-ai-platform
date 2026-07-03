"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import Avatar, { AvatarState } from "./Avatar";
import { API_BASE, fetchStore, PageContext, Product, swatchStyle } from "./catalog";
import { Markdown } from "./markdown";
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

// Topic-narrowing follow-ups: after a policy answer, drill into that same topic rather than
// jumping to random suggestions, so each question can go deeper.
const TOPICS: { test: RegExp; qs: string[] }[] = [
  { test: /return|refund|exchange/i, qs: ["How long do refunds take?", "Can I exchange for another size?", "What if my item arrived damaged?"] },
  { test: /ship|deliver|arrive|warehouse|express/i, qs: ["When is shipping free?", "How fast is express shipping?", "How long to ship to Toronto?"] },
  { test: /\bsize|fit|true to size|runs small|measurement/i, qs: ["Where is the size chart?", "Do the leggings run small?", "What if I'm between sizes?"] },
  { test: /\bpay|payment|afterpay|installment|visa|paypal/i, qs: ["Can I pay in installments?", "Do you take Apple Pay?", "Do you sell gift cards?"] },
  { test: /wash|care|dry|fabric softener|merino/i, qs: ["Can I tumble dry it?", "How do I wash merino?", "Does washing affect the warranty?"] },
  { test: /warranty|defect|guarantee/i, qs: ["What does the warranty cover?", "How do I make a claim?", "What is your return policy?"] },
  { test: /member|circle|points|loyalty/i, qs: ["How do I earn points?", "What are the member perks?", "Is membership free?"] },
  { test: /discount|student|promo|adjust/i, qs: ["Do you have a student discount?", "Can I get a price adjustment?", "Do you run sales?"] },
  { test: /store|pickup|location|gastown|yorkville|soho/i, qs: ["Where are your stores?", "Can I pick up in store?", "Do stores take returns?"] },
];

// Follow-ups that track the conversation: product-specific when it recommended something, topic
// follow-ups after a policy answer, and starter suggestions when it was unsure.
function followupsFor(final: FinalEvent, text: string, recs: Product[], suggestions: Suggestion[]): string[] {
  const unsure =
    final.tier !== "auto" ||
    /(don't|do not|couldn't|could not|cannot) (have|find|see)|not sure|no information|don't know/i.test(text);
  if (!unsure && recs.length > 0) {
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
    out.push("Is it in stock in my size?");
    return out;
  }
  if (!unsure) {
    for (const t of TOPICS) if (t.test.test(text)) return t.qs;
  }
  return suggestions.slice(0, 3).map((s) => s.text);
}

// Minimal shape of the browser SpeechRecognition. We run it continuously so the shopper can
// interrupt the assistant mid-sentence (barge-in), like ChatGPT voice.
type SpeechAlt = { transcript: string };
type SpeechRes = { [j: number]: SpeechAlt; isFinal: boolean };
type SpeechResult = { resultIndex: number; results: { [i: number]: SpeechRes; length: number } };
type Recognizer = {
  lang: string;
  interimResults: boolean;
  continuous: boolean;
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
  context = null,
}: {
  open: boolean;
  setOpen: (v: boolean) => void;
  seed?: string | null;
  context?: PageContext;
}) {
  const [mounted, setMounted] = useState(false);
  const [token, setToken] = useState<string | null>(null);
  const [brand, setBrand] = useState("");
  const [products, setProducts] = useState<Product[]>([]);

  useEffect(() => {
    setMounted(true);
    const existing = localStorage.getItem("skein_token");
    if (existing) {
      setToken(existing);
    } else {
      // frictionless demo: mint a demo token so the visitor can just ask, no password wall. If the
      // server is in production this 404s and we fall back to the sign-in form below.
      fetch(`${API_BASE}/api/demo-login`, { method: "POST" })
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => {
          if (d?.access_token) {
            localStorage.setItem("skein_token", d.access_token);
            setToken(d.access_token);
          }
        })
        .catch(() => {});
    }
    fetchStore().then((s) => {
      setBrand(s.brand);
      setProducts(s.products);
    });
  }, []);

  const short = brand.split(" ")[0] || "Shopping"; // the assistant names itself from the pack

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
        <button
          className="fab"
          onClick={() => setOpen(true)}
          aria-label={`Open the ${short} assistant`}
        >
          <span className="dot" />
          &#128172;
        </button>
      )}
      {open && (
        <section className="panel" role="dialog" aria-label={`${short} assistant`}>
          <header className="panel-hdr">
            <Avatar state="idle" size={38} />
            <div>
              <div className="t">{short} Assistant</div>
              <div className="s">
                <span className="online-dot" /> Online now
              </div>
            </div>
            <button className="x" onClick={() => setOpen(false)} aria-label="Close">
              &times;
            </button>
          </header>
          {token ? (
            <Conversation
              token={token}
              onSignOut={signOut}
              seed={seed}
              brand={short}
              products={products}
              context={context}
            />
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
  brand,
  products,
  context,
}: {
  token: string;
  onSignOut: () => void;
  seed?: string | null;
  brand: string;
  products: Product[];
  context?: PageContext;
}) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [voiceOn, setVoiceOn] = useState(false);
  const [voiceState, setVoiceState] = useState<"greeting" | "listening" | "thinking" | "speaking">(
    "greeting",
  );
  const [heard, setHeard] = useState("");
  const [name, setName] = useState("");
  const streamRef = useRef<HTMLDivElement | null>(null);
  const seededRef = useRef<string | null>(null);
  const recogRef = useRef<Recognizer | null>(null);
  const voiceLiveRef = useRef(false); // lets stopVoice break the async loop
  const processingRef = useRef(false); // handling an utterance (thinking or speaking)
  const speakingRef = useRef(false); // currently speaking, so a new utterance can barge in
  const utterRef = useRef(0); // id of the current utterance, to drop superseded ones
  const lastSpokenRef = useRef(""); // what we are saying, to ignore the mic hearing our own voice

  const authHeaders = {
    "Content-Type": "application/json",
    Authorization: `Bearer ${token}`,
  };

  useEffect(() => {
    fetch(`${API_BASE}/api/suggestions`, { headers: { Authorization: `Bearer ${token}` } })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => d && setSuggestions(d.suggestions || []))
      .catch(() => {});
    const saved = localStorage.getItem("aster_name");
    if (saved) setName(saved);
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
    // light personalization: remember a name the shopper offers, and greet back if that is all
    // they said, without bothering the model
    const nameHit = /(?:my name is|call me|i am|i'm)\s+([a-zA-Z][a-zA-Z'-]{1,19})\b/i.exec(q);
    if (nameHit && !/looking|not|from|trying|wondering|here|interested|sure|okay|good/i.test(nameHit[1])) {
      const nm = nameHit[1][0].toUpperCase() + nameHit[1].slice(1).toLowerCase();
      localStorage.setItem("aster_name", nm);
      setName(nm);
      if (q.trim().length <= nameHit[0].length + 4) {
        setInput("");
        setMessages((m) => [
          ...m,
          { role: "me", text: q },
          { role: "bot", text: `Nice to meet you, ${nm}. How can I help you shop today?` },
        ]);
        return "";
      }
    }
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

  // Handle one spoken utterance. If we are mid-answer, the shopper is interrupting (barge-in), so
  // we cut the speech and take the new question. Superseded utterances drop out via the id check.
  async function handleUtterance(text: string) {
    if (!voiceLiveRef.current) return;
    if (processingRef.current && !speakingRef.current) return; // busy thinking, ignore
    if (speakingRef.current) {
      // the mic may just be hearing our own voice; ignore an echo of what we are saying
      if (lastSpokenRef.current.toLowerCase().includes(text.toLowerCase()) && text.length > 4) return;
      if (typeof window !== "undefined") window.speechSynthesis?.cancel(); // barge in
    }
    const myId = ++utterRef.current;
    processingRef.current = true;
    speakingRef.current = false;
    setHeard(text);
    setVoiceState("thinking");
    const answer = await send(text);
    if (!voiceLiveRef.current || myId !== utterRef.current) return; // stopped or superseded
    speakingRef.current = true;
    lastSpokenRef.current = answer;
    setVoiceState("speaking");
    await speakAsync(answer);
    if (myId !== utterRef.current) return; // a newer utterance took over while we spoke
    speakingRef.current = false;
    processingRef.current = false;
    if (voiceLiveRef.current) setVoiceState("listening");
  }

  // Continuous recognition, so the shopper can talk any time, even over the answer.
  function startListening() {
    const rec = makeRecognizer();
    if (!rec) return;
    rec.lang = (typeof navigator !== "undefined" && navigator.language) || "en-US";
    rec.interimResults = true;
    rec.continuous = true;
    rec.maxAlternatives = 1;
    rec.onresult = (e) => {
      let finalText = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        if (e.results[i].isFinal) finalText += e.results[i][0].transcript;
      }
      if (finalText.trim()) handleUtterance(finalText.trim());
    };
    rec.onerror = () => {};
    rec.onend = () => {
      if (voiceLiveRef.current) {
        try {
          rec.start(); // browsers stop after a pause; keep the mic alive until Stop
        } catch {
          /* already started */
        }
      }
    };
    recogRef.current = rec;
    try {
      rec.start();
    } catch {
      /* already started */
    }
  }

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
    processingRef.current = false;
    speakingRef.current = false;
    utterRef.current = 0;
    setHeard("");
    setVoiceState("greeting");
    speakingRef.current = true;
    const greeting = `Hi${name ? " " + name : ""}, I'm your ${brand} assistant. How can I help you today?`;
    lastSpokenRef.current = greeting;
    await speakAsync(greeting);
    speakingRef.current = false;
    if (!voiceLiveRef.current) return;
    setVoiceState("listening");
    startListening();
  }

  function stopVoice() {
    voiceLiveRef.current = false;
    utterRef.current++; // invalidate any in-flight utterance
    processingRef.current = false;
    speakingRef.current = false;
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
    const avatarState: AvatarState =
      voiceState === "speaking" ? "speaking" : voiceState === "thinking" ? "thinking" : "idle";
    return (
      <div className="voice">
        <div>
          <div className={`voice-ring ${voiceState}`}>
            <Avatar state={avatarState} size={120} />
          </div>
          <div className="state">{voiceLabel}</div>
          {heard && <div className="said">&ldquo;{heard}&rdquo;</div>}
          <div className="voice-hint">You can talk any time — even to interrupt.</div>
        </div>
        <button className="btn btn-primary" onClick={stopVoice}>
          Stop
        </button>
      </div>
    );
  }

  // context-aware openers, so the first prompts match the page the shopper came from
  const short2 = (s: string) => s.replace(/^Aster /, "");
  const ctxSuggest: string[] =
    context?.kind === "product"
      ? [`Is the ${short2(context.name)} good for me?`, "Does it run true to size?", "Is it in stock?"]
      : context?.kind === "category"
        ? [`Best ${context.category.replace(/s$/, "")} for cold weather?`, `Show me ${context.category} under $100`]
        : context?.kind === "help"
          ? [`Tell me more about ${context.topic}`]
          : [];
  const ctxLabel =
    context?.kind === "product"
      ? `Viewing ${short2(context.name)}`
      : context?.kind === "category"
        ? `Browsing ${cap(context.category)}`
        : context?.kind === "help"
          ? `Help · ${cap(context.topic)}`
          : "";

  return (
    <>
      <div className="stream" ref={streamRef}>
        {empty && (
          <div className="greet">
            {ctxLabel && <div className="ctx-chip">{ctxLabel}</div>}
            <div className="big">
              {name ? `Hi ${name}, ` : "Hi, "}I&apos;m your {brand} assistant.
            </div>
            {context?.kind === "product"
              ? `Ask me anything about the ${short2(context.name)}, or tap the mic to talk.`
              : "How can I help you today? Ask about a product, sizing, shipping, or what to wear, or tap the mic to talk."}
          </div>
        )}
        {empty && (
          <div className="sugg">
            {[...ctxSuggest, ...suggestions.map((s) => s.text)].slice(0, 5).map((t, i) => (
              <button key={`${i}-${t}`} type="button" onClick={() => send(t)}>
                {t}
              </button>
            ))}
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} style={{ display: "contents" }}>
            {m.role === "bot" && !m.text && loading ? (
              <div className="msg bot think typing">
                <Avatar state="thinking" size={22} />
                <span className="dots">
                  <i /><i /><i />
                </span>
              </div>
            ) : (
              <div className={`msg ${m.role}`}>
                {m.role === "bot" ? <Markdown text={m.text} /> : m.text}
              </div>
            )}
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
                  <Link key={p.id} className="rec" href={`/product/${p.id}`}>
                    <div className="sw" style={swatchStyle(p.color)} />
                    <div className="rb">
                      <div className="rn">{p.name.replace(/^Aster /, "")}</div>
                      <div className="rp">
                        {cap(p.category)}
                        {p.price != null ? ` · $${p.price.toFixed(0)}` : ""}
                      </div>
                    </div>
                  </Link>
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
