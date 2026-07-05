"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import Avatar, { AvatarState } from "./Avatar";
import { API_BASE, fetchStore, PageContext, Product } from "./catalog";
import ImageTile from "./ImageTile";
import { Markdown } from "./markdown";

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
  agent?: boolean; // from Sara, the human agent, after an escalation
};

const AGENT_INTRO =
  "Hi, I'm Sara, Aster's AI care specialist. 👋 I've got you from here. Tell me what's going on, " +
  "and if it's about an order I'll pull it up right away.";

function cap(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

// Is a JWT still valid (not within 30s of expiry)? A cached-but-expired token would otherwise be
// trusted on load, boot the chat, then 401 on the first message and dead-end at "Connecting...".
// An unparseable token is trusted (the server will 401 it, which triggers a clean reconnect).
function tokenAlive(t: string): boolean {
  try {
    const payload = JSON.parse(atob((t.split(".")[1] || "").replace(/-/g, "+").replace(/_/g, "/")));
    return typeof payload.exp !== "number" || payload.exp * 1000 > Date.now() + 30_000;
  } catch {
    return true;
  }
}

// Turn an internal citation id (CK33, graph:Product:P007, metric:...) into a human-readable source.
function citeLabel(c: Citation): string {
  const id = c.id || "";
  if (id.startsWith("graph:")) return "Product data";
  if (id.startsWith("metric:")) return "Store metric";
  if (/^CK/i.test(id)) return "Store policy";
  if (/^(CATR|CAT)/i.test(id)) return "Catalog";
  if (/^OC/i.test(id)) return "Buying guide";
  if (/^(PD|PC)/i.test(id)) return "Product info";
  if (/^OD/i.test(id) || c.doc_type === "order") return "Order record";
  if (/^(R\d|RV|RG)/i.test(id)) return "Customer review";
  if (c.doc_type === "review") return "Customer review";
  if (c.doc_type === "product") return "Product info";
  if (c.doc_type === "guide") return "Store info";
  return id;
}

// Collapse citations to unique human-readable sources (many CK ids all read as "Store policy").
function dedupeCites(cites?: Citation[]): Citation[] {
  if (!cites) return [];
  const seen = new Set<string>();
  const out: Citation[] = [];
  for (const c of cites) {
    const label = citeLabel(c);
    if (seen.has(label)) continue;
    seen.add(label);
    out.push(c);
    if (out.length >= 4) break;
  }
  return out;
}

// Products the assistant named in its answer, matched by catalog name, so we can show them as cards.
function recsFromAnswer(text: string, products: Product[]): Product[] {
  if (!text) return [];
  const low = text.toLowerCase();
  const seen = new Set<string>();
  const hits: { p: Product; at: number }[] = [];
  for (const p of products) {
    const full = p.name.toLowerCase();
    const short = full.replace(/^aster\s+/, "");
    const idxs = [low.indexOf(full), low.indexOf(short)].filter((x) => x >= 0);
    if (idxs.length && !seen.has(p.id)) {
      seen.add(p.id);
      hits.push({ p, at: Math.min(...idxs) });
    }
  }
  // order the cards by where each product first appears in the answer, so the card row matches the
  // order of the bulleted recommendations rather than catalog order
  return hits.sort((a, b) => a.at - b.at).slice(0, 4).map((h) => h.p);
}

// Topic-narrowing follow-ups: after a policy answer, drill into that same topic rather than
// jumping to random suggestions, so each question can go deeper.
// Topic threads: each keeps its own set of narrowing follow-ups, so a conversation about returns
// keeps suggesting return questions instead of jumping to leggings.
const TOPIC_MAP: Record<string, string[]> = {
  returns: ["How long do refunds take?", "Can I exchange for another size?", "What if my item arrived damaged?"],
  shipping: ["When is shipping free?", "How fast is express shipping?", "How long to ship to my city?"],
  sizing: ["Where is the size chart?", "Do the leggings run small?", "What if I'm between sizes?"],
  payment: ["Can I pay in installments?", "Do you take Apple Pay?", "Do you sell gift cards?"],
  care: ["Can I tumble dry it?", "How do I wash merino?", "Does washing affect the warranty?"],
  warranty: ["What does the warranty cover?", "How do I make a claim?", "How do I return a faulty item?"],
  membership: ["How do I earn points?", "What are the member perks?", "Is membership free?"],
  discounts: ["Do you have a student discount?", "Can I get a price adjustment?", "Do you run sales?"],
  stores: ["Where are your stores?", "Can I pick up in store?", "Do stores take returns?"],
  order: ["Where is my order?", "Can I change my order?", "How do I track my order?"],
  gift: ["A gift for her?", "A gift for him?", "Any gifts under $50?"],
};
const TOPIC_PATTERNS: [string, RegExp][] = [
  ["order", /\b(my order|track|order status|didn'?t (get|receive)|not arriv|where is my|hasn'?t arrived)\b/i],
  ["returns", /\b(returns?|refunds?|exchanges?)\b/i],
  ["warranty", /\b(warranty|defect|guarantee|damaged|faulty|broken)\b/i],
  ["membership", /\b(member|membership|circle|points|loyalty|rewards?)\b/i],
  ["payment", /\b(pay|payment|afterpay|installments?|visa|paypal|klarna)\b/i],
  ["care", /\b(wash|care|dry|fabric softener|merino|shrink)\b/i],
  ["discounts", /\b(discount|student|promo|price adjust|sale|coupon)\b/i],
  ["stores", /\b(store|pickup|pick up|location|in person)\b/i],
  ["gift", /\b(gift|present)\b|for my (girlfriend|boyfriend|wife|husband|mom|dad|friend)/i],
  ["shipping", /\b(ship|shipping|shipped|deliver|delivery|arrives?|express|warehouse)\b/i],
  ["sizing", /\b(size|sizes|sizing|fit|true to size|runs small|measurement)\b/i],
];

function detectTopic(text: string): string {
  for (const [key, rx] of TOPIC_PATTERNS) if (rx.test(text)) return key;
  return "";
}

// After recommending products, offer refinement follow-ups (narrow by color / price / use / a
// specific pick), not generic policy questions. Relevant to what the shopper is actually browsing.
function productFollowups(p: Product, recs: Product[]): string[] {
  const short = p.name.replace(/^Aster\s+/, "");
  const useByCat: Record<string, string> = {
    jackets: "Which is warmest for winter?",
    leggings: "Which is best for the gym?",
    bras: "Which has the most support?",
    tops: "Which is best for hot weather?",
    bottoms: "Which is best for travel?",
    shorts: "Which has pockets?",
    bags: "Which is best for the gym?",
    hoodies: "Which is the coziest?",
    accessories: "Which is warmest?",
  };
  const out = [`Do you have the ${short} in another color?`];
  out.push(recs.length > 1 ? (useByCat[p.category] || "Which do you recommend most?") : "Show me similar options");
  out.push("Any under $100?");
  return out;
}

// Sticky, narrowing follow-ups. A HARD policy topic (order, returns, shipping...) always wins.
// Otherwise a product recommendation gets refinement follow-ups (so "size and color for socks"
// suggests colors/alternatives, not the size chart). Soft topics and suggestions are the fallback.
const HARD_TOPICS = new Set([
  "order", "returns", "warranty", "shipping", "payment", "membership", "discounts", "stores",
]);
function followupsFor(
  query: string,
  final: FinalEvent,
  recs: Product[],
  suggestions: Suggestion[],
  lastTopic: string,
): { followups: string[]; topic: string } {
  const unsure = final.tier !== "auto";
  const detected = detectTopic(query);
  if (detected && HARD_TOPICS.has(detected)) return { followups: TOPIC_MAP[detected], topic: detected };
  if (!unsure && recs.length > 0) return { followups: productFollowups(recs[0], recs), topic: "" };
  if (detected) return { followups: TOPIC_MAP[detected], topic: detected };
  if (lastTopic && TOPIC_MAP[lastTopic]) return { followups: TOPIC_MAP[lastTopic], topic: lastTopic };
  return { followups: suggestions.slice(0, 3).map((s) => s.text), topic: lastTopic };
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
  onerror: (e: { error?: string }) => void;
  onend: () => void;
  start: () => void;
  abort: () => void;
};

// Strip markdown and emoji so the voice speaks clean text (no "star star", "bracket one", or the
// voice trying to pronounce an emoji).
function plainSpeak(text: string): string {
  return text
    .replace(/\s*\[\d+(?:\s*,\s*\d+)*\](?:\s*(?:,\s*)?(?:and\s+|&\s+)?\[\d+(?:\s*,\s*\d+)*\])*/g, "")
    .replace(/\s+([,.;:!?])/g, "$1")
    .replace(/([,;:])(?:\s*[,;:])+/g, "$1")
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/^\s*[*-]\s+/gm, "")
    .replace(/^\s*\d+[.)]\s+/gm, "")
    .replace(
      /[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\u{2190}-\u{21FF}\u{2B00}-\u{2BFF}\u{FE00}-\u{FE0F}\u{200D}]/gu,
      "",
    )
    .replace(/\s+/g, " ")
    .trim();
}
// Significant words, for a fuzzy echo check that survives punctuation and casing differences.
function normWords(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2);
}
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
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(guard);
      resolve();
    };
    // Chrome's speechSynthesis can stall without firing onend (long or remote-voice utterances), which
    // would leave voiceState stuck at "Speaking..." forever. Resolve on a duration-based guard too.
    const guard = setTimeout(finish, Math.min(30_000, 2_000 + 65 * text.length));
    u.onend = finish;
    u.onerror = finish;
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

  // One-time setup: mark mounted and load the storefront (brand + products).
  useEffect(() => {
    setMounted(true);
    fetchStore().then((s) => {
      setBrand(s.brand);
      setProducts(s.products);
    });
  }, []);

  // Connect / reconnect whenever there is no valid token. Keyed on `token`, so a 401 that calls
  // signOut() (token -> null) automatically restarts login instead of dead-ending at "Connecting".
  // Frictionless demo: silently mint a demo token. This converges on the first load even when the
  // API is still warming up (uvicorn --reload, or a scaled-to-zero deploy) and under React
  // StrictMode's double-mounted effect: the token is cached the instant it arrives (so a discarded
  // mount can't waste it), the UI converges to whatever is cached, and retries are short and bounded
  // (~0.6s, then 2s) rather than an exponential backoff that reads as a hang. The orphaned StrictMode
  // request is aborted so only one call is ever in flight.
  useEffect(() => {
    if (token) return;
    let alive = true;
    const controller = new AbortController();
    const connect = (tries = 0) => {
      if (!alive) return;
      const cached = localStorage.getItem("skein_token");
      if (cached && tokenAlive(cached)) {
        setToken(cached);
        return;
      }
      if (cached) localStorage.removeItem("skein_token"); // expired: drop it and mint a fresh one
      fetch(`${API_BASE}/api/demo-login`, { method: "POST", signal: controller.signal })
        .then((r) => {
          if (r.status === 404) {
            // Production: demo-login is disabled, so the gate token expired. Send the visitor back
            // to the landing gate to sign in again instead of polling a 404 forever.
            localStorage.removeItem("aster_gate");
            localStorage.removeItem("skein_token");
            window.location.assign("/");
            return null;
          }
          return r.ok ? r.json() : null;
        })
        .then((d) => {
          if (d?.access_token) localStorage.setItem("skein_token", d.access_token); // cache first
          if (d?.name) localStorage.setItem("skein_name", d.name); // greet the shopper back by name
          const now = localStorage.getItem("skein_token");
          if (!alive) return;
          if (now) setToken(now);
          else setTimeout(() => connect(tries + 1), tries < 8 ? 500 + Math.random() * 300 : 2000);
        })
        .catch((e) => {
          if (e?.name === "AbortError" || !alive) return;
          setTimeout(() => connect(tries + 1), tries < 8 ? 500 + Math.random() * 300 : 2000);
        });
    };
    connect();
    return () => {
      alive = false;
      controller.abort();
    };
  }, [token]);

  // If the storefront failed to load on a cold first paint (fetchStore swallows errors and returns
  // empty), refetch once the token confirms the API is up, so the greeting names the brand and the
  // product rec cards render instead of staying blank for the whole session.
  useEffect(() => {
    if (!token || products.length) return;
    let alive = true;
    fetchStore().then((s) => {
      if (!alive) return;
      if (s.brand) setBrand(s.brand);
      if (s.products.length) setProducts(s.products);
    });
    return () => {
      alive = false;
    };
  }, [token, products.length]);

  const short = brand.split(" ")[0] || "Shopping"; // the assistant names itself from the pack

  // one-shot manual reconnect, so a stuck visitor never has to refresh the page
  function forceConnect() {
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

  function signOut() {
    localStorage.removeItem("skein_token");
    setToken(null);
  }
  // Note: the remembered name (skein_name) is kept across an expiry-triggered signOut so the
  // reconnected session still greets the returning shopper; a fresh demo-login refreshes it.

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
            <div className="chat-connecting">
              <Avatar state="thinking" size={34} />
              <p>Connecting you to the assistant...</p>
              <button className="chip" type="button" onClick={forceConnect}>
                Retry
              </button>
            </div>
          )}
        </section>
      )}
    </>
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
  const [messages, setMessages] = useState<Message[]>(() => {
    // restore the session's history at mount, so closing and reopening the chat keeps it. A lazy
    // initializer avoids the empty-state save effect from clobbering the saved history.
    try {
      const raw = typeof window !== "undefined" ? localStorage.getItem("aster_chat") : null;
      const parsed = raw ? JSON.parse(raw) : null;
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  });
  const [loading, setLoading] = useState(false);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [voiceOn, setVoiceOn] = useState(false);
  const [voiceState, setVoiceState] = useState<"greeting" | "listening" | "thinking" | "speaking">(
    "greeting",
  );
  const [heard, setHeard] = useState("");
  const [name, setName] = useState(
    () => (typeof localStorage !== "undefined" && localStorage.getItem("skein_name")) || "",
  );
  const [agentMode, setAgentMode] = useState<boolean>(() => {
    // restore agent mode with the history, so a refresh mid-handoff does not silently turn the
    // "human" specialist back into the assistant
    try {
      return typeof window !== "undefined" && localStorage.getItem("aster_agent") === "1";
    } catch {
      return false;
    }
  });
  const [micOn, setMicOn] = useState(true);
  const streamRef = useRef<HTMLDivElement | null>(null);
  const seededRef = useRef<string | null>(null);
  const lastTopicRef = useRef(""); // sticky conversation topic, so follow-ups narrow and stay put
  const recogRef = useRef<Recognizer | null>(null);
  const voiceLiveRef = useRef(false); // lets stopVoice break the async loop
  const processingRef = useRef(false); // handling an utterance (thinking or speaking)
  const speakingRef = useRef(false); // currently speaking, so a new utterance can barge in
  const utterRef = useRef(0); // id of the current utterance, to drop superseded ones
  const lastSpokenRef = useRef(""); // what we are saying, to ignore the mic hearing our own voice
  const spokeEndRef = useRef(0); // when we stopped speaking, so a lagging echo tail is still caught
  const micDeadRef = useRef(false); // mic denied/unavailable, so do not restart the recognizer
  const micOnRef = useRef(true); // the shopper muted the mic without leaving voice mode
  const handlerRef = useRef<(t: string) => void>(() => {}); // latest handleUtterance, avoids stale closure
  // premium (ElevenLabs) voice playback + lip-sync
  const audioElRef = useRef<HTMLAudioElement | null>(null); // the currently playing answer audio
  const audioCtxRef = useRef<AudioContext | null>(null); // reused Web Audio context for the analyser
  const speakSeqRef = useRef(0); // monotonic token: bumped on every speak() and every stopSpeaking()
  const speakDoneRef = useRef<null | (() => void)>(null); // resolves the play promise on barge-in
  const [speakLevel, setSpeakLevel] = useState(0); // 0..1 mouth openness, drives the avatar lips
  const agentModeRef = useRef(false); // mirror of agentMode, read inside already-running voice closures
  const bargedRef = useRef(false); // an interim barge-in fired: accept the next final utterance as-is

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
    // persist the running conversation (history restored by the useState initializer above)
    try {
      if (messages.length) localStorage.setItem("aster_chat", JSON.stringify(messages.slice(-40)));
    } catch {
      /* storage full: skip persisting */
    }
  }, [messages]);

  useEffect(() => {
    agentModeRef.current = agentMode; // keep the ref in sync for already-running voice closures
    try {
      localStorage.setItem("aster_agent", agentMode ? "1" : "0");
    } catch {
      /* ignore */
    }
  }, [agentMode]);

  function clearChat() {
    setMessages([]);
    setAgentMode(false);
    agentModeRef.current = false;
    lastTopicRef.current = "";
    try {
      localStorage.removeItem("aster_chat");
      localStorage.setItem("aster_agent", "0");
    } catch {
      /* ignore */
    }
  }

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
    // light personalization: remember a name the shopper explicitly gives, and greet back if that
    // is all they said. Only the explicit forms, so "I'm between sizes" never becomes a name.
    const nameHit = /(?:my name is|call me)\s+([a-zA-Z][a-zA-Z'-]{1,19})\b/i.exec(q);
    // words that follow "call me" but are not a name ("call me asap", "call me back later",
    // "my name is not important"), so we never greet someone as "Asap" or persist "Not"
    const notAName =
      /^(an?|the|me|you|i|it|when|what|where|why|how|who|if|back|now|soon|later|today|tonight|tomorrow|anytime|whenever|asap|about|that|this|please|maybe|not|no|yes|ok|okay|sir|maam|madam|dude|guys?|again|here|there)$/i;
    if (nameHit && !notAName.test(nameHit[1])) {
      const nm = nameHit[1][0].toUpperCase() + nameHit[1].slice(1).toLowerCase();
      localStorage.setItem("aster_name", nm);
      setName(nm);
      if (q.trim().length <= nameHit[0].length + 4) {
        const reply = `Nice to meet you, ${nm}. How can I help you shop today?`;
        setInput("");
        setMessages((m) => [...m, { role: "me", text: q }, { role: "bot", text: reply }]);
        return reply; // return the text so voice mode speaks it
      }
    }
    // human handoff: a short bare request ("human", "agent", "human plz") or a verb+person phrase.
    // The length guard keeps "a gift for a person who loves yoga" from being an escalation.
    const qt = q.trim();
    // a question about the assistant's nature ("are you human?", "is this a bot?") is answered
    // honestly by the backend, not treated as a request to be transferred to a person
    const asksNature = /^(are|is|am)\s+(you|this|it|i|u)\b/i.test(qt);
    const shortHuman =
      qt.split(/\s+/).length <= 4 &&
      /\b(human|agent|representative|real person|advisor|operator|manager|specialist)\b/i.test(q);
    // A verb+person phrase in EITHER order, so "talk to a human" and "a human I can talk to" both
    // escalate. NOT "get" (too generic: "get someone a gift", "get the most support") and NOT the
    // vague nouns "someone/somebody/support" that ordinary shopping uses. The gap is tight so
    // "connect me with a plan that has good support" cannot match.
    const verbHuman =
      /\b(talk|speak|chat|connect|transfer|reach|escalate)\b.{0,20}\b(human|person|agent|representative|rep|advisor|operator|manager|supervisor|specialist)\b/i.test(
        q,
      ) ||
      /\b(human|person|agent|representative|rep|advisor|operator|manager|supervisor|specialist)\b.{0,20}\b(talk|speak|chat|connect|transfer|reach|escalate)\b/i.test(
        q,
      );
    // An explicit refusal must not escalate. Two shapes: (1) a negated desire/handoff
    // ("I don't want to talk to an agent", "no need for a human", "rather not speak to a person");
    // (2) a bare negation directly on the human noun with no affirmative request ("no human", "not
    // a human please"). A leading discourse "No, I want to talk to a human" is NOT a refusal: it
    // carries an affirmative verbHuman, so it still escalates.
    const refusesHuman =
      /\b(don'?t|do not|never|without|no need|rather not|no thanks?)\b[^.?!]{0,24}\b(human|person|agent|representative|rep|advisor|operator|manager|specialist)\b/i.test(
        q,
      ) ||
      (/\b(no|not)\b[^.?!]{0,12}\b(human|person|agent|representative|rep|advisor|operator|manager|specialist)\b/i.test(
        q,
      ) &&
        !verbHuman);
    if (!agentMode && !asksNature && !refusesHuman && (shortHuman || verbHuman)) {
      setInput("");
      setMessages((m) => [...m, { role: "me", text: q }, { role: "bot", agent: true, text: AGENT_INTRO }]);
      setAgentMode(true);
      agentModeRef.current = true; // sync now so the intro is spoken in Sara's voice, not Aria's
      return AGENT_INTRO; // return the intro so voice mode speaks it
    }
    setInput("");
    setMessages((m) => [...m, { role: "me", text: q }, { role: "bot", text: "", agent: agentMode }]);
    setLoading(true);
    // Only fold the current product into the query when the shopper clearly means "it" (a pronoun)
    // and is not naming another category. Otherwise "a jacket for LA" while viewing a waist pack
    // would keep retrieving that same product.
    let qSend = q;
    // never fold product context in agent mode: after escalating from a product page, "when will
    // it arrive?" is about the shopper's order, not the product they were viewing
    if (context?.kind === "product" && !agentMode) {
      const nm = context.name.replace(/^Aster /i, "");
      const pronoun = /\b(it|its|it's|this|that|the one|the item|the product)\b/i.test(q);
      // Domain agnostic: the query "names another item" if it mentions any catalog category
      // other than the one in context. Categories come from the live catalog, not a hardcoded
      // vocabulary, so the engine stays domain neutral (and the leak linter stays green).
      const stem = (s: string) => s.toLowerCase().replace(/s$/, "");
      const ctxCat = stem(context.category || "");
      const otherThing = products.some((pp) => {
        const c = stem(pp.category || "");
        // "top"/"bottom" double as ordinary words ("top pick", "loose at the bottom"), so never
        // treat those category stems as the shopper naming a different item
        return (
          !!c && c !== ctxCat && c !== "top" && c !== "bottom" && new RegExp(`\\b${c}s?\\b`, "i").test(q)
        );
      });
      if (pronoun && !otherThing && !q.toLowerCase().includes(nm.toLowerCase())) {
        qSend = `${q} (about the Aster ${nm})`;
      }
    }
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
    // prior turns, so the assistant can resolve follow-ups ("which is cheaper") and multi-turn
    // verification (an email then a name) instead of treating each message as brand new
    const history = messages
      .filter((mm) => mm.text && mm.text.trim())
      .slice(-6)
      .map((mm) => ({ role: mm.role === "me" ? "user" : "assistant", content: mm.text }));
    // Watchdog so a cold or wedged API that accepts the socket but never sends bytes can't leave the
    // typing dots (and voice "Thinking...") spinning forever: abort if no first byte in 30s, and if
    // the stream then goes idle for 60s. An abort lands in the catch below ("Could not reach...").
    const ctrl = new AbortController();
    let watchdog = setTimeout(() => ctrl.abort(), 30_000);
    const armIdle = () => {
      clearTimeout(watchdog);
      watchdog = setTimeout(() => ctrl.abort(), 60_000);
    };
    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: "POST",
        headers: authHeaders,
        signal: ctrl.signal,
        body: JSON.stringify({
          query: qSend,
          ...(agentMode ? { persona: "agent" } : {}),
          ...(voiceLiveRef.current ? { concise: true } : {}), // short, speakable reply in voice mode
          ...(history.length ? { history } : {}),
        }),
      });
      if (res.status === 401) {
        // The token expired. Clearing it triggers an automatic reconnect (the connect effect is
        // keyed on the token), so tell the shopper rather than dropping their message silently.
        const m = "Your session expired, reconnecting. Please resend that.";
        patchBot((b) => ({ ...b, text: m }));
        onSignOut();
        return m;
      }
      if (res.status === 429) {
        const m = "You are asking too fast. Please wait a moment.";
        patchBot((b) => ({ ...b, text: m }));
        return m;
      }
      if (!res.ok || !res.body) {
        const m = "Sorry, something went wrong. Please try again.";
        patchBot((b) => ({ ...b, text: m }));
        return m; // return the text so voice mode speaks it instead of going silent
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
          armIdle(); // reset the idle timeout on each chunk received
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
              const fu = followupsFor(q, fe, recs, suggestions, lastTopicRef.current);
              lastTopicRef.current = fu.topic; // remember the thread for the next turn
              patchBot((b) => ({
                ...b,
                text: fe.answer ?? b.text,
                final: fe,
                recs,
                followups: fu.followups,
              }));
            } else if (event.type === "error") {
              result = "Sorry, something went wrong.";
              patchBot((b) => ({ ...b, text: result }));
            }
          }
        }
      } finally {
        reader.cancel().catch(() => {});
      }
      if (!received) {
        result = "No response received. Please try again.";
        patchBot((b) => ({ ...b, text: result }));
      }
    } catch {
      result = "Could not reach the assistant. Is the API running?";
      patchBot((b) => ({ ...b, text: result }));
    } finally {
      clearTimeout(watchdog);
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

  function escalate() {
    if (agentMode) return;
    setAgentMode(true);
    setMessages((m) => [...m, { role: "bot", agent: true, text: AGENT_INTRO }]);
  }
  function endAgent() {
    setAgentMode(false);
    agentModeRef.current = false;
    setMessages((m) => [
      ...m,
      { role: "bot", text: "You're back with Aria, your shopping assistant. 😊 What else can I help you find?" },
    ]);
  }

  // Handle one spoken utterance. If we are mid-answer, the shopper is interrupting (barge-in), so
  // Speak an answer. Try the premium server voice (ElevenLabs via /api/tts) with amplitude
  // lip-sync; on 204 (no key configured) or any error, fall back to the browser's built-in voice.
  async function speak(text: string, persona?: string): Promise<void> {
    const clean = plainSpeak(text);
    if (!clean) return;
    const seq = ++speakSeqRef.current; // this call owns `seq`; stopSpeaking() bumps it to cancel
    // Timeout so a cold/hung /api/tts can't freeze the greeting at "Say hello..." or lock voice in
    // "Speaking...": abort after 8s and fall through to the browser voice (the catch handles it).
    const ttsCtrl = new AbortController();
    const ttsTimer = setTimeout(() => ttsCtrl.abort(), 8_000);
    try {
      const res = await fetch(`${API_BASE}/api/tts`, {
        method: "POST",
        headers: authHeaders,
        signal: ttsCtrl.signal,
        body: JSON.stringify({ text: clean.slice(0, 1200), persona: persona || null }),
      });
      if (speakSeqRef.current !== seq) {
        clearTimeout(ttsTimer);
        return; // barged in during the fetch: do not play stale audio
      }
      if (res.ok && res.status !== 204) {
        const blob = await res.blob(); // keep the timer armed across the body read (shares the signal)
        clearTimeout(ttsTimer);
        const played = await playWithLipSync(blob, seq);
        if (played || speakSeqRef.current !== seq) return; // spoke it, or superseded: no double voice
      } else {
        clearTimeout(ttsTimer);
      }
    } catch {
      clearTimeout(ttsTimer);
      /* network/blocked/timeout: fall through to the browser voice */
    }
    if (speakSeqRef.current !== seq) return;
    // browser fallback (204 / error / premium playback failed): the browser voice gives no
    // amplitude, so fake a gentle mouth oscillation while it speaks, then close it
    let t = 0;
    const iv = window.setInterval(() => {
      if (speakSeqRef.current !== seq) return;
      t += 0.4;
      setSpeakLevel(0.2 + 0.22 * Math.abs(Math.sin(t)));
    }, 90);
    try {
      await speakAsync(clean);
    } finally {
      clearInterval(iv);
      if (speakSeqRef.current === seq) setSpeakLevel(0);
    }
  }

  // Play the answer MP3 with amplitude lip-sync. Resolves true only if audio actually played, so the
  // caller can fall back to the browser voice on failure. All per-play nodes are local and cleaned
  // up on every exit path; shared lip-sync state is only touched while this play still owns `seq`.
  function playWithLipSync(blob: Blob, seq: number): Promise<boolean> {
    return new Promise((resolve) => {
      if (speakSeqRef.current !== seq) {
        resolve(false);
        return;
      }
      const url = URL.createObjectURL(blob);
      const audio = new Audio(url);
      audioElRef.current = audio;
      let srcNode: MediaElementAudioSourceNode | null = null;
      let analyser: AnalyserNode | null = null;
      let rafId = 0;
      let played = false;
      let settled = false;
      const finish = (ok: boolean) => {
        if (settled) return;
        settled = true;
        if (rafId) cancelAnimationFrame(rafId);
        try {
          srcNode?.disconnect();
          analyser?.disconnect();
        } catch {
          /* already torn down */
        }
        try {
          URL.revokeObjectURL(url);
        } catch {
          /* ignore */
        }
        if (speakSeqRef.current === seq) setSpeakLevel(0); // only reset the mouth if still our turn
        if (audioElRef.current === audio) audioElRef.current = null;
        if (speakDoneRef.current === onBargeIn) speakDoneRef.current = null;
        resolve(ok && played);
      };
      const onBargeIn = () => finish(false);
      speakDoneRef.current = onBargeIn; // stopSpeaking() calls this to cut us off instantly
      audio.onended = () => finish(true);
      audio.onerror = () => finish(false);
      (async () => {
        try {
          const Ctx =
            window.AudioContext ||
            (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
          let ctx = audioCtxRef.current;
          if (!ctx) {
            ctx = new Ctx();
            audioCtxRef.current = ctx;
          }
          if (ctx.state === "suspended") await ctx.resume(); // await, or the first answer is silent
          if (speakSeqRef.current !== seq) {
            finish(false);
            return;
          }
          srcNode = ctx.createMediaElementSource(audio);
          analyser = ctx.createAnalyser();
          analyser.fftSize = 256;
          srcNode.connect(analyser);
          analyser.connect(ctx.destination);
          const data = new Uint8Array(analyser.frequencyBinCount);
          let last = 0;
          const tick = () => {
            if (speakSeqRef.current !== seq || settled) return;
            analyser!.getByteTimeDomainData(data);
            let sum = 0;
            for (let i = 0; i < data.length; i++) {
              const v = (data[i] - 128) / 128;
              sum += v * v;
            }
            const level = Math.min(1, Math.sqrt(sum / data.length) * 2.4);
            const now = performance.now();
            if (now - last > 45) {
              setSpeakLevel(Math.round(level * 100) / 100); // throttle to ~22fps
              last = now;
            }
            rafId = requestAnimationFrame(tick);
          };
          rafId = requestAnimationFrame(tick);
        } catch {
          /* Web Audio unavailable/exhausted: play the element raw (still audible, no lip-sync) */
        }
        try {
          await audio.play();
          played = true;
        } catch {
          finish(false); // autoplay blocked / decode error: let the caller use the browser voice
        }
      })();
    });
  }

  // Stop any speech immediately (barge-in / stop): cancels the current speak (via the seq bump),
  // the browser voice, and the audio element, resolves a pending play promise, and closes the mouth.
  function stopSpeaking() {
    speakSeqRef.current++; // invalidate any in-flight or playing speak()
    if (typeof window !== "undefined") window.speechSynthesis?.cancel();
    const a = audioElRef.current;
    if (a) {
      try {
        a.pause();
      } catch {
        /* ignore */
      }
      audioElRef.current = null;
    }
    setSpeakLevel(0);
    const done = speakDoneRef.current;
    speakDoneRef.current = null;
    if (done) done(); // resolves the pending play promise so its handleUtterance can continue
  }

  // we cut the speech and take the new question. Superseded utterances drop out via the id check.
  async function handleUtterance(text: string) {
    if (!voiceLiveRef.current || !micOnRef.current) return;
    // if the interim path already detected a real interruption, accept this final utterance as-is
    // rather than dropping it on the echo/length guards below (which would swallow "stop"/"wait")
    const barged = bargedRef.current;
    bargedRef.current = false;
    if (barged) {
      stopSpeaking(); // ensure the audio is cut, then take this question
    } else {
      if (processingRef.current && !speakingRef.current) return; // busy thinking, ignore
      if (speakingRef.current) {
        // the mic may just be hearing our own voice; ignore an utterance that mostly echoes what we
        // are saying, and only treat a genuinely new one as a barge-in
        const spoken = new Set(normWords(lastSpokenRef.current));
        const words = normWords(text);
        const overlap = words.length ? words.filter((w) => spoken.has(w)).length / words.length : 0;
        if (words.length < 2) return; // a single word while speaking is almost always echo/noise
        if (overlap > 0.6) return;
        stopSpeaking(); // barge in: cut the premium audio (or browser voice) instantly
      } else if (Date.now() - spokeEndRef.current < 1500) {
        // recognition can finalize the tail of our own answer just after we stop speaking; drop an
        // utterance in that brief window if it mostly matches what we just said
        const spoken = new Set(normWords(lastSpokenRef.current));
        const words = normWords(text);
        const overlap = words.length ? words.filter((w) => spoken.has(w)).length / words.length : 0;
        if (words.length < 2 || overlap > 0.6) return;
      }
    }
    const myId = ++utterRef.current;
    processingRef.current = true;
    speakingRef.current = false;
    setHeard(text);
    setVoiceState("thinking");
    const answer = await send(text);
    if (!voiceLiveRef.current || myId !== utterRef.current) return; // stopped or superseded
    // note: a mute does NOT stop the assistant from speaking; muting only stops listening to you
    speakingRef.current = true;
    const spoken = plainSpeak(answer);
    lastSpokenRef.current = spoken;
    setVoiceState("speaking");
    await speak(spoken, agentModeRef.current ? "agent" : undefined); // ref, not stale closure state
    if (myId !== utterRef.current) return; // a newer utterance took over while we spoke
    speakingRef.current = false;
    spokeEndRef.current = Date.now(); // start the echo-tail window
    processingRef.current = false;
    if (voiceLiveRef.current) setVoiceState("listening");
  }
  handlerRef.current = handleUtterance; // keep the recognizer callback on the latest closure

  // Continuous recognition, so the shopper can talk any time, even over the answer.
  function startListening() {
    const rec = makeRecognizer();
    if (!rec) return;
    rec.lang = (typeof navigator !== "undefined" && navigator.language) || "en-US";
    rec.interimResults = true;
    rec.continuous = true;
    rec.maxAlternatives = 1;
    // Track consecutive 'network' errors (Chrome's speech service offline) so onend does not spin
    // restarting the recognizer with zero backoff, stuck at "Listening..." while nothing is heard.
    let netErr = 0;
    let hadNetError = false;
    rec.onresult = (e) => {
      let finalText = "";
      let interim = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const t = e.results[i][0].transcript;
        interim += t;
        if (e.results[i].isFinal) finalText += t;
      }
      if (interim.trim()) netErr = 0; // the service is working again; clear the error streak
      // barge-in the instant the shopper starts speaking: if the assistant is talking and this is
      // not just the mic hearing its own voice, stop speaking so they can talk
      if (speakingRef.current && micOnRef.current && interim.trim()) {
        const spoken = new Set(normWords(lastSpokenRef.current));
        const words = normWords(interim);
        const overlap = words.length ? words.filter((w) => spoken.has(w)).length / words.length : 0;
        if (words.length >= 1 && overlap < 0.5 && typeof window !== "undefined") {
          stopSpeaking();
          bargedRef.current = true; // so the final transcript ("stop"/"wait") is not dropped later
        }
      }
      if (finalText.trim()) handlerRef.current(finalText.trim());
    };
    rec.onerror = (e) => {
      const err = e?.error || "";
      // permission or hardware problems are fatal: stop, do not spin restarting the recognizer
      if (err === "not-allowed" || err === "service-not-allowed" || err === "audio-capture") {
        micDeadRef.current = true;
        stopVoice();
        setMessages((m) => [
          ...m,
          { role: "bot", text: "I couldn't reach the microphone. Please allow mic access, or type your question." },
        ]);
      } else if (err === "network") {
        // The browser's speech service is unreachable. Back off, and after a few tries give up
        // (with a message) instead of an invisible zero-backoff restart loop.
        hadNetError = true;
        netErr += 1;
        if (netErr >= 3) {
          micDeadRef.current = true;
          stopVoice();
          setMessages((m) => [
            ...m,
            { role: "bot", text: "Voice isn't reachable right now, so I've switched off the mic. You can type your question, or try voice again in a moment." },
          ]);
        }
      }
      // no-speech / aborted are benign: onend restarts normally
    };
    rec.onend = () => {
      if (!voiceLiveRef.current || micDeadRef.current) return;
      const delay = hadNetError ? 1500 : 0; // browsers stop after a pause; keep the mic alive
      hadNetError = false;
      window.setTimeout(() => {
        if (!voiceLiveRef.current || micDeadRef.current) return;
        try {
          rec.start();
        } catch {
          /* already started */
        }
      }, delay);
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
    micDeadRef.current = false;
    micOnRef.current = true;
    setMicOn(true);
    bargedRef.current = false;
    // note: utterRef is NOT reset here (stopVoice already bumped it); resetting to 0 would let a
    // stale in-flight turn from the previous session match the new session's first id and hijack it
    setHeard("");
    setVoiceState("greeting");
    speakingRef.current = true;
    // greet in the right voice: if the shopper was already handed to the human specialist, keep
    // it Sara, not Aria
    const greeting = agentModeRef.current
      ? `Hi${name ? " " + name : ""}, Sara here. How can I help?`
      : `Hi${name ? " " + name : ""}, I'm Aria. What can I help you find?`;
    lastSpokenRef.current = greeting;
    await speak(greeting, agentModeRef.current ? "agent" : undefined);
    speakingRef.current = false;
    spokeEndRef.current = Date.now();
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
    stopSpeaking();
    setVoiceOn(false);
  }

  function toggleMute() {
    // mute only your microphone (so background noise does not interrupt); the assistant keeps
    // talking, and you just listen
    const on = !micOnRef.current;
    micOnRef.current = on;
    setMicOn(on);
  }

  useEffect(() => {
    // stop the mic and any speech if the widget unmounts mid-conversation, and release Web Audio so
    // repeated open/close does not leak running AudioContexts (browsers cap them at ~6). These are
    // data refs (not DOM nodes); reading their live values at unmount is exactly the intent.
    return () => {
      voiceLiveRef.current = false;
      // eslint-disable-next-line react-hooks/exhaustive-deps
      speakSeqRef.current++;
      recogRef.current?.abort();
      if (typeof window !== "undefined") window.speechSynthesis?.cancel();
      audioElRef.current?.pause();
      speakDoneRef.current?.(); // revokes the object URL, cancels the RAF, disconnects nodes
      speakDoneRef.current = null;
      audioCtxRef.current?.close().catch(() => {});
      audioCtxRef.current = null;
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
    const lastBot = [...messages].reverse().find((m) => m.role === "bot" && m.text);
    return (
      <div className="voice">
        <div style={{ width: "100%" }}>
          <div className={`voice-ring ${voiceState}`}>
            <Avatar state={avatarState} size={104} level={speakLevel} />
          </div>
          <div className="state">
            {!micOn && voiceState !== "speaking" ? "Your mic is muted" : voiceLabel}
          </div>
          {/* live transcript, like ChatGPT voice: see what you said and the answer as it streams */}
          {(heard || lastBot) && (
            <div className="voice-live">
              {heard && <div className="vl-said">&ldquo;{heard}&rdquo;</div>}
              {lastBot && (
                <div className="vl-ans">
                  <Markdown text={lastBot.text} products={products} />
                </div>
              )}
              {lastBot?.recs && lastBot.recs.length > 0 && (
                <div className="recs">
                  {lastBot.recs.map((p) => (
                    <Link key={p.id} className="rec" href={`/product/${p.id}`}>
                      <ImageTile category={p.category} color={p.color} name={p.name} className="rec-img" />
                      <div className="rb">
                        <div className="rn">{p.name.replace(/^Aster /, "")}</div>
                        <div className="rp">{p.price != null ? `$${p.price.toFixed(0)}` : ""}</div>
                      </div>
                    </Link>
                  ))}
                </div>
              )}
            </div>
          )}
          <div className="voice-hint">You can talk any time, even to interrupt.</div>
        </div>
        <div className="voice-controls">
          <button
            className={micOn ? "icon-btn" : "icon-btn muted"}
            onClick={toggleMute}
            aria-label={micOn ? "Mute microphone" : "Unmute microphone"}
            title={micOn ? "Mute" : "Unmute"}
          >
            {micOn ? "🎤" : "🔇"}
          </button>
          <button className="btn btn-primary" onClick={stopVoice}>
            End
          </button>
        </div>
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
      {agentMode && (
        <div className="agent-banner">
          <span>
            <b>Sara</b> · Aster care specialist
          </span>
          <button onClick={endAgent}>Back to Aria</button>
        </div>
      )}
      {!agentMode && messages.length > 0 && (
        <div className="chat-toolbar">
          <button className="chat-clear" onClick={clearChat}>
            Clear chat
          </button>
        </div>
      )}
      <div
        className="stream"
        ref={streamRef}
        role="log"
        aria-live="polite"
        aria-atomic="false"
        aria-label="Conversation"
      >
        {empty && (
          <div className="greet">
            {ctxLabel && <div className="ctx-chip">{ctxLabel}</div>}
            <div className="big">
              {name ? `Welcome back, ${name}! ` : "Hi, "}I&apos;m {brand ? "Aria" : "your assistant"}. 😊
            </div>
            {context?.kind === "product"
              ? `Any questions about the ${short2(context.name)}? I can help with the fit, colors, sizing, stock, or how it wears.`
              : context?.kind === "category"
                ? `Looking at ${cap(context.category)}? I can help you find the right one, compare options, or check sizing. What matters most to you?`
                : "How can I help you today? Ask about a product, sizing, shipping, gifts, or what to wear, or tap the mic to talk."}
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
              <>
                {m.agent && <div className="agent-tag">Sara · care specialist</div>}
                <div className={`msg ${m.role}${m.agent ? " agent" : ""}`}>
                  {m.role === "bot" ? <Markdown text={m.text} products={products} /> : m.text}
                </div>
              </>
            )}
            {m.final && (
              <div className="meta-row">
                <span className={`tier tier-${m.final.tier}`}>{m.final.tier}</span>
                {dedupeCites(m.final.citations).map((c) => (
                  <span key={c.n} className="cite" title={c.id}>
                    {citeLabel(c)}
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
            {i === messages.length - 1 && m.final?.tier === "abstain" && !agentMode && (
              <button className="escalate-btn" type="button" onClick={escalate}>
                💬 Talk to a human agent
              </button>
            )}
            {m.recs && m.recs.length > 0 && (
              <div className="recs">
                {m.recs.map((p) => (
                  <Link key={p.id} className="rec" href={`/product/${p.id}`}>
                    <ImageTile category={p.category} color={p.color} name={p.name} className="rec-img" />
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
      <a
        className="chat-credit"
        href="https://esteki.ca/"
        target="_blank"
        rel="noopener noreferrer"
      >
        Created by esteki.ca
      </a>
    </>
  );
}
