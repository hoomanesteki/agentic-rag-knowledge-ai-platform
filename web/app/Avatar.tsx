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
          {/* shoulders / collar */}
          <path d="M12 64 Q32 46 52 64 Z" fill="#f2f1ec" />
          <path d="M28 52 h8 v8 h-8 Z" fill="#e7e5df" />
          {/* hair back */}
          <path d="M15 34 Q15 12 32 12 Q49 12 49 34 L49 46 Q49 40 44 40 L20 40 Q15 40 15 46 Z" fill="#3a2c28" />
          {/* face */}
          <ellipse className="ava-face" cx="32" cy="32" rx="14" ry="15" fill="#f2c8a8" />
          {/* fringe */}
          <path d="M18 28 Q20 18 32 18 Q44 18 46 28 Q40 24 32 24 Q24 24 18 28 Z" fill="#3a2c28" />
          {/* eyes */}
          <g className="ava-eyes" fill="#2a211e">
            <ellipse cx="26" cy="32" rx="2.1" ry="2.6" />
            <ellipse cx="38" cy="32" rx="2.1" ry="2.6" />
          </g>
          {/* blush */}
          <circle cx="24" cy="37" r="2.2" fill="#eeae90" opacity="0.6" />
          <circle cx="40" cy="37" r="2.2" fill="#eeae90" opacity="0.6" />
          {/* mouth */}
          <path className="ava-mouth" d="M28 40 Q32 43 36 40" stroke="#a85a52" strokeWidth="1.6" fill="none" strokeLinecap="round" />
        </g>
      </svg>
      <span className="ava-dots">
        <i /><i /><i />
      </span>
    </span>
  );
}
