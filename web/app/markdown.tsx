import Link from "next/link";
import { ReactNode } from "react";

// Minimal, safe markdown for assistant answers: **bold**, bullet lists (* or -), and numbered
// lists. No raw HTML is ever injected; everything is built as React nodes, so it cannot execute
// markup from the model or the context.

type Prod = { id: string; name: string };

// Turn product names inside a text run into links to the product page, so the shopper can click the
// name in the sentence (not just the card below). Longest names first, so a full brand-prefixed name
// wins over its short form. Matches the full catalog name and a distinctive multi-word short form
// (the brand prefix dropped); single-word short forms are skipped so common words never become links.
function linkTargets(products?: Prod[]): { label: string; id: string }[] {
  if (!products || products.length === 0) return [];
  const out: { label: string; id: string }[] = [];
  const seen = new Set<string>();
  for (const p of products) {
    const full = p.name.trim();
    const short = full.replace(/^Aster\s+/i, "").trim();
    for (const label of [full, short]) {
      const key = label.toLowerCase();
      const distinctive = label === full || /\s/.test(label); // skip single-word short forms
      if (label.length >= 5 && distinctive && !seen.has(key)) {
        seen.add(key);
        out.push({ label, id: p.id });
      }
    }
  }
  return out.sort((a, b) => b.label.length - a.label.length);
}

function linkify(text: string, key: string, targets: { label: string; id: string }[]): ReactNode[] {
  if (targets.length === 0) return [text];
  const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const rx = new RegExp(`\\b(${targets.map((t) => esc(t.label)).join("|")})\\b`, "gi");
  const out: ReactNode[] = [];
  let last = 0;
  let i = 0;
  let m: RegExpExecArray | null;
  while ((m = rx.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const hit = targets.find((t) => t.label.toLowerCase() === m![1].toLowerCase());
    if (hit) {
      out.push(
        <Link key={`${key}-a${i++}`} href={`/product/${hit.id}`} className="prod-link">
          {m[1]}
        </Link>,
      );
    } else {
      out.push(m[1]);
    }
    last = m.index + m[1].length;
    if (rx.lastIndex === m.index) rx.lastIndex++; // guard against a zero-width match
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function inline(text: string, key: string, targets: { label: string; id: string }[]): ReactNode[] {
  const out: ReactNode[] = [];
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  parts.forEach((p, i) => {
    if (p.startsWith("**") && p.endsWith("**")) {
      out.push(<strong key={`${key}-b${i}`}>{linkify(p.slice(2, -2), `${key}-b${i}`, targets)}</strong>);
    } else if (p) {
      out.push(...linkify(p, `${key}-t${i}`, targets));
    }
  });
  return out;
}

export function Markdown({ text, products }: { text: string; products?: Prod[] }) {
  // Strip inline citation markers ([1], [2, 3]) from the visible text: a shopper should read a
  // human answer, not footnotes. The sources still show as chips below the message (from the
  // citations array), and the raw text is kept elsewhere for product-matching and grounding.
  // Strip whole runs of citation markers, including the commas/"and" between separate ones
  // ("[2], [3], and [9]"), then tidy any punctuation the removal left dangling, so an order lookup
  // that cites many sources doesn't render as ", , , and".
  const clean = text
    .replace(/\s*\[\d+(?:\s*,\s*\d+)*\](?:\s*(?:,\s*)?(?:and\s+|&\s+)?\[\d+(?:\s*,\s*\d+)*\])*/g, "")
    .replace(/\s+([,.;:!?])/g, "$1")
    .replace(/([,;:])(?:\s*[,;:])+/g, "$1")
    .replace(/\s{2,}/g, " ");
  const targets = linkTargets(products);
  const lines = clean.split(/\r?\n/);
  const blocks: ReactNode[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;

  const flush = () => {
    if (!list) return;
    const items = list.items.map((it, i) => (
      <li key={i}>{inline(it, `li${blocks.length}-${i}`, targets)}</li>
    ));
    blocks.push(list.ordered ? <ol key={`ol${blocks.length}`}>{items}</ol> : <ul key={`ul${blocks.length}`}>{items}</ul>);
    list = null;
  };

  for (const raw of lines) {
    const line = raw.trim();
    const bullet = /^[*-]\s+(.*)/.exec(line);
    const numbered = /^\d+[.)]\s+(.*)/.exec(line);
    if (bullet) {
      if (!list || list.ordered) flush();
      list = list || { ordered: false, items: [] };
      list.items.push(bullet[1]);
    } else if (numbered) {
      if (!list || !list.ordered) flush();
      list = list || { ordered: true, items: [] };
      list.items.push(numbered[1]);
    } else {
      flush();
      if (line) blocks.push(<p key={`p${blocks.length}`}>{inline(line, `p${blocks.length}`, targets)}</p>);
    }
  }
  flush();
  return <>{blocks}</>;
}
