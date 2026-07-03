"use client";

// Small, dependency-free charts (SVG + CSS) for the manager dashboard.
const PALETTE = ["#1f6f5c", "#37b899", "#e0a72e", "#d97706", "#3d5a78", "#5a6b7b", "#a1a5ac", "#b06a3b"];

export function BarList({ items }: { items: { label: string; count: number }[] }) {
  const max = Math.max(1, ...items.map((i) => i.count));
  return (
    <div className="barlist">
      {items.map((i) => (
        <div key={i.label} className="barlist-row">
          <span className="bl-label" title={i.label}>
            {i.label}
          </span>
          <span className="bl-track">
            <span className="bl-fill" style={{ width: `${(i.count / max) * 100}%` }} />
          </span>
          <span className="bl-count">{i.count}</span>
        </div>
      ))}
    </div>
  );
}

export function Donut({ slices }: { slices: { label: string; pct: number }[] }) {
  const r = 52;
  const c = 2 * Math.PI * r;
  let offset = 0;
  return (
    <div className="donut-wrap">
      <svg viewBox="0 0 140 140" className="donut">
        <g transform="translate(70,70) rotate(-90)">
          {slices.map((s, i) => {
            const len = (s.pct / 100) * c;
            const el = (
              <circle
                key={s.label}
                r={r}
                fill="none"
                stroke={PALETTE[i % PALETTE.length]}
                strokeWidth="20"
                strokeDasharray={`${len} ${c - len}`}
                strokeDashoffset={-offset}
              />
            );
            offset += len;
            return el;
          })}
        </g>
      </svg>
      <div className="donut-legend">
        {slices.map((s, i) => (
          <div key={s.label} className="dl-row">
            <span className="dl-dot" style={{ background: PALETTE[i % PALETTE.length] }} />
            <span className="dl-label">{s.label}</span>
            <span className="dl-pct">{s.pct}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function Funnel({ steps }: { steps: { step: string; count: number }[] }) {
  const top = Math.max(1, steps[0]?.count || 1);
  return (
    <div className="funnel">
      {steps.map((s, i) => {
        const pct = Math.round((s.count / top) * 100);
        const conv = i > 0 ? Math.round((s.count / (steps[i - 1].count || 1)) * 100) : 100;
        return (
          <div key={s.step} className="funnel-row">
            <div className="fn-head">
              <span>{s.step}</span>
              <span className="fn-count">
                {s.count.toLocaleString()} {i > 0 && <em>({conv}%)</em>}
              </span>
            </div>
            <div className="fn-track">
              <div className="fn-fill" style={{ width: `${pct}%`, background: PALETTE[i % PALETTE.length] }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function Sparkbars({ data }: { data: { day: string; count: number }[] }) {
  const max = Math.max(1, ...data.map((d) => d.count));
  return (
    <div className="sparkbars">
      {data.map((d, i) => (
        <span
          key={i}
          className="sb-bar"
          style={{ height: `${(d.count / max) * 100}%` }}
          title={`${d.day}: ${d.count}`}
        />
      ))}
    </div>
  );
}
