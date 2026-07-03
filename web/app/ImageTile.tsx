import { colorHex } from "./catalog";

// A clean garment silhouette in the product's real color on a soft neutral ground, so every
// product reads like a catalog shot without stock photography. Swap for real photos later by
// pointing this at an <img> when a file exists.
const SHAPES: Record<string, React.ReactNode> = {
  tops: <path d="M35 26 L26 31 L18 44 L27 51 L34 45 L34 96 L66 96 L66 45 L73 51 L82 44 L74 31 L65 26 Q57 33 50 33 Q43 33 35 26 Z" />,
  bras: (
    <>
      <path d="M27 46 Q50 35 73 46 L71 60 Q50 72 29 60 Z" />
      <path d="M34 47 L40 31 M66 47 L60 31" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
    </>
  ),
  jackets: (
    <>
      <path d="M35 26 L26 31 L18 44 L27 51 L34 45 L34 96 L66 96 L66 45 L73 51 L82 44 L74 31 L65 26 Q57 33 50 33 Q43 33 35 26 Z" />
      <path d="M50 33 L50 96 M43 30 L50 40 L57 30" fill="none" stroke="rgba(0,0,0,.18)" strokeWidth="1.6" />
    </>
  ),
  hoodies: (
    <>
      <path d="M35 30 L26 34 L18 47 L27 54 L34 48 L34 96 L66 96 L66 48 L73 54 L82 47 L74 34 L65 30 Z" />
      <path d="M38 30 Q50 16 62 30 Q56 36 50 36 Q44 36 38 30 Z" />
      <rect x="40" y="70" width="20" height="13" rx="2" fill="none" stroke="rgba(0,0,0,.18)" strokeWidth="1.6" />
    </>
  ),
  leggings: <path d="M33 22 L67 22 L64 100 L53 100 L50 52 L47 100 L36 100 Z" />,
  bottoms: <path d="M32 22 L68 22 L65 100 L53 100 L50 52 L47 100 L35 100 Z" />,
  shorts: <path d="M33 30 L67 30 L65 66 L54 66 L50 48 L46 66 L35 66 Z" />,
  bags: (
    <>
      <rect x="30" y="42" width="40" height="50" rx="5" />
      <path d="M40 42 Q40 27 50 27 Q60 27 60 42" fill="none" stroke="currentColor" strokeWidth="4" />
    </>
  ),
  accessories: (
    <>
      <path d="M30 62 Q30 34 50 34 Q70 34 70 62 Z" />
      <rect x="27" y="58" width="46" height="9" rx="4" />
    </>
  ),
};

export default function ImageTile({
  category,
  color,
  className = "",
}: {
  category: string;
  color: string | null;
  className?: string;
}) {
  const hex = colorHex(color);
  const shape = SHAPES[category] || SHAPES.tops;
  return (
    <div className={`imgtile ${className}`}>
      <svg viewBox="0 0 100 120" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
        <g fill={hex} stroke="rgba(0,0,0,.10)" strokeWidth="1" style={{ color: hex }}>
          {shape}
        </g>
      </svg>
    </div>
  );
}
