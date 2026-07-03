import { colorHex } from "./catalog";

// A clean garment silhouette in the product's real color on a soft neutral ground, so every
// product reads like a catalog shot without stock photography. Accessories and bags pick a shape
// from the product name (a glove is not a beanie), so the picture matches the item.
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
  // bags
  tote: (
    <>
      <rect x="30" y="42" width="40" height="50" rx="5" />
      <path d="M40 42 Q40 27 50 27 Q60 27 60 42" fill="none" stroke="currentColor" strokeWidth="4" />
    </>
  ),
  backpack: (
    <>
      <rect x="30" y="34" width="40" height="54" rx="12" />
      <rect x="40" y="48" width="20" height="24" rx="4" fill="none" stroke="rgba(0,0,0,.2)" strokeWidth="1.6" />
      <path d="M40 36 Q40 24 50 24 Q60 24 60 36" fill="none" stroke="currentColor" strokeWidth="3" />
    </>
  ),
  duffel: (
    <>
      <rect x="20" y="46" width="60" height="30" rx="15" />
      <path d="M38 46 Q38 37 50 37 Q62 37 62 46" fill="none" stroke="currentColor" strokeWidth="4" />
    </>
  ),
  sling: (
    <>
      <rect x="38" y="50" width="30" height="34" rx="9" />
      <path d="M40 52 L66 28" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
    </>
  ),
  beltbag: (
    <>
      <rect x="30" y="54" width="40" height="22" rx="11" />
      <path d="M30 60 L15 52 M70 60 L85 52" fill="none" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
    </>
  ),
  // accessories
  beanie: (
    <>
      <path d="M30 62 Q30 34 50 34 Q70 34 70 62 Z" />
      <rect x="27" y="58" width="46" height="9" rx="4" />
    </>
  ),
  cap: (
    <>
      <path d="M28 56 Q28 34 50 34 Q70 34 72 54 Z" />
      <path d="M70 54 Q90 54 90 63 L70 63 Z" />
    </>
  ),
  socks: (
    <>
      <path d="M40 24 L54 24 L54 60 Q54 74 42 74 L32 74 Q24 74 24 66 Q24 60 32 58 L40 55 Z" />
      <path d="M40 24 L54 24 L54 33 L40 33 Z" fill="none" stroke="rgba(0,0,0,.18)" strokeWidth="1.6" />
    </>
  ),
  gloves: (
    <>
      <path d="M36 46 Q36 34 48 34 L58 34 Q66 34 66 44 L66 64 Q66 78 52 78 L48 78 Q36 78 36 64 Z" />
      <path d="M64 46 Q76 44 76 52 Q76 60 66 60 Z" />
    </>
  ),
  scarf: (
    <>
      <path d="M34 28 L46 28 L46 74 L40 88 L34 74 Z" />
      <path d="M54 28 L66 28 L66 74 L60 88 L54 74 Z" />
      <rect x="34" y="28" width="32" height="9" rx="2" />
    </>
  ),
  headband: <path d="M22 54 Q50 34 78 54 L78 63 Q50 46 22 63 Z" />,
};

function shapeKey(category: string, name: string): string {
  const n = (name || "").toLowerCase();
  if (category === "accessories") {
    if (/sock/.test(n)) return "socks";
    if (/glove/.test(n)) return "gloves";
    if (/\bcap\b/.test(n)) return "cap";
    if (/scarf/.test(n)) return "scarf";
    if (/headband/.test(n)) return "headband";
    return "beanie";
  }
  if (category === "bags") {
    if (/backpack/.test(n)) return "backpack";
    if (/duffel/.test(n)) return "duffel";
    if (/sling/.test(n)) return "sling";
    if (/belt bag/.test(n)) return "beltbag";
    return "tote";
  }
  return category;
}

export default function ImageTile({
  category,
  color,
  name = "",
  className = "",
}: {
  category: string;
  color: string | null;
  name?: string;
  className?: string;
}) {
  const hex = colorHex(color);
  const shape = SHAPES[shapeKey(category, name)] || SHAPES.tops;
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
