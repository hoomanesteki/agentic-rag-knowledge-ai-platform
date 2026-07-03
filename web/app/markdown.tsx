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
  const lines = text.split(/\r?\n/);
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
