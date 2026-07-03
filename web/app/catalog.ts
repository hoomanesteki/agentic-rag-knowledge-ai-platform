// Shared storefront types and helpers. The product "photo" is a color swatch derived from the
// product's real color, so the store looks intentional without stock imagery.

export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Product = {
  id: string;
  name: string;
  category: string;
  price: number | null;
  color: string | null;
  sizes: string[];
};

// Named colors from the catalog mapped to a base hex. Unknowns fall back to a neutral.
const COLORS: Record<string, string> = {
  black: "#1d1d20",
  "storm blue": "#3d5a78",
  "heather grey": "#a1a5ac",
  "heather gray": "#a1a5ac",
  slate: "#5a6b7b",
  oatmeal: "#d7ccb8",
  charcoal: "#3a3f46",
  navy: "#2a3550",
  olive: "#5f6b4a",
  sand: "#cbb997",
};

export function colorHex(color: string | null): string {
  if (!color) return "#8b8f96";
  return COLORS[color.trim().toLowerCase()] || "#8b8f96";
}

// A soft top-lit gradient of the product color, so each tile reads as a premium swatch.
export function swatchStyle(color: string | null): React.CSSProperties {
  const base = colorHex(color);
  return { background: `radial-gradient(120% 120% at 30% 20%, ${base}dd, ${base})` };
}

export async function fetchCatalog(): Promise<Product[]> {
  try {
    const res = await fetch(`${API_BASE}/api/catalog`);
    if (!res.ok) return [];
    const data = await res.json();
    return (data.products || []) as Product[];
  } catch {
    return [];
  }
}
