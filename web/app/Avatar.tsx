"use client";

// A stylized, friendly assistant avatar drawn in SVG (no external assets). It animates by state:
// idle blinks and gently breathes, thinking bobs with a soft pulse, speaking moves the mouth.
export type AvatarState = "idle" | "thinking" | "speaking";

export default function Avatar({ state = "idle", size = 34 }: { state?: AvatarState; size?: number }) {
  return (
    <span className={`ava ava-${state}`} style={{ width: size, height: size }} aria-hidden="true">
      <svg viewBox="0 0 64 64" width={size} height={size}>
        <defs>
          <clipPath id="ava-c">
            <circle cx="32" cy="32" r="32" />
          </clipPath>
          <linearGradient id="ava-bg" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#2a8e77" />
            <stop offset="1" stopColor="#14140f" />
          </linearGradient>
        </defs>
        <g clipPath="url(#ava-c)">
          <rect width="64" height="64" fill="url(#ava-bg)" />
          {/* shoulders */}
          <path d="M10 64 Q32 45 54 64 Z" fill="#f2f1ec" />
          {/* neck */}
          <rect x="29" y="42" width="6" height="8" fill="#eec0a0" />
          {/* hair: light, open bob framing the face */}
          <path className="ava-hair" d="M17 34 Q16 13 32 12 Q48 13 47 34 Q47 45 43 49 L43 30 Q43 21 32 21 Q21 21 21 30 L21 49 Q17 45 17 34 Z" fill="#c98f52" />
          {/* face */}
          <ellipse className="ava-face" cx="32" cy="31" rx="13.5" ry="15" fill="#f4cdaa" />
          {/* side-swept fringe */}
          <path d="M20 30 Q23 18 33 19 Q44 20 44 30 Q40 24 31 24 Q25 24 20 30 Z" fill="#bd8347" />
          {/* brows */}
          <g className="ava-brows" stroke="#a5763f" strokeWidth="1.4" strokeLinecap="round">
            <path d="M23 27 Q26 25.5 29 27" fill="none" />
            <path d="M35 27 Q38 25.5 41 27" fill="none" />
          </g>
          {/* eyes */}
          <g className="ava-eyes" fill="#40342b">
            <ellipse cx="26.5" cy="31" rx="2" ry="2.6" />
            <ellipse cx="37.5" cy="31" rx="2" ry="2.6" />
          </g>
          {/* blush */}
          <circle cx="23.5" cy="36" r="2.1" fill="#f0a488" opacity="0.55" />
          <circle cx="40.5" cy="36" r="2.1" fill="#f0a488" opacity="0.55" />
          {/* mouth */}
          <path className="ava-mouth" d="M28 39 Q32 42 36 39" stroke="#b5615a" strokeWidth="1.6" fill="none" strokeLinecap="round" />
        </g>
      </svg>
      <span className="ava-dots">
        <i /><i /><i />
      </span>
    </span>
  );
}
