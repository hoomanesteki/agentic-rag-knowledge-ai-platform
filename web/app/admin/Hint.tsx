"use client";

// A small "?" badge that reveals a plain-language explanation on hover or focus, so every metric
// in the back office says what it means and why it matters. Keyboard focusable for accessibility.
export function Hint({ text }: { text: string }) {
  return (
    <span className="ax-hint" tabIndex={0} role="note" aria-label={text}>
      ?<span className="ax-hint-bubble">{text}</span>
    </span>
  );
}
