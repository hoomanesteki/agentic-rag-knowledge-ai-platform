import { ReactNode } from "react";

// Minimal, safe markdown for assistant answers: **bold**, bullet lists (* or -), and numbered
// lists. No raw HTML is ever injected; everything is built as React nodes, so it cannot execute
// markup from the model or the context.
function inline(text: string, key: string): ReactNode[] {
  const out: ReactNode[] = [];
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  parts.forEach((p, i) => {
    if (p.startsWith("**") && p.endsWith("**")) {
      out.push(<strong key={`${key}-b${i}`}>{p.slice(2, -2)}</strong>);
    } else if (p) {
      out.push(p);
    }
  });
  return out;
}

export function Markdown({ text }: { text: string }) {
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
  const lines = clean.split(/\r?\n/);
  const blocks: ReactNode[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;

  const flush = () => {
    if (!list) return;
    const items = list.items.map((it, i) => <li key={i}>{inline(it, `li${blocks.length}-${i}`)}</li>);
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
      if (line) blocks.push(<p key={`p${blocks.length}`}>{inline(line, `p${blocks.length}`)}</p>);
    }
  }
  flush();
  return <>{blocks}</>;
}
