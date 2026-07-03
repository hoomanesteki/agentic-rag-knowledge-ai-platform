"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";

import { useCart } from "./cart";
import { colorHex, fetchStore, Product } from "./catalog";
import { useChat } from "./ChatProvider";
import { isGated } from "./gate";
import ImageTile from "./ImageTile";
import Landing from "./Landing";
import StoreFooter from "./StoreFooter";
import StoreHeader from "./StoreHeader";

function cap(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

export default function Page() {
  const [gated, setGated] = useState<boolean | null>(null);
  useEffect(() => setGated(isGated()), []);
  if (gated === null) return null; // brief hold to avoid a flash before we know
  if (!gated) return <Landing onEnter={() => setGated(true)} />;
  return (
    <Suspense fallback={null}>
      <Home />
    </Suspense>
  );
}

function Home() {
  const params = useSearchParams();
  const gender = params.get("g") || "all";
  const q = (params.get("q") || "").trim();
  const [products, setProducts] = useState<Product[]>([]);
  const [brand, setBrand] = useState("");
  const [cat, setCat] = useState("all");
  const [limit, setLimit] = useState(12);
  const { openWith, setContext } = useChat();

  useEffect(() => {
    fetchStore().then((s) => {
      setProducts(s.products);
      setBrand(s.brand);
    });
  }, []);

  useEffect(() => {
    setCat("all");
    setLimit(12);
  }, [gender, q]); // reset when switching Women/Men or searching

  const forGender = useMemo(
    () =>
      gender === "all"
        ? products
        : products.filter((p) => p.gender === gender || p.gender === "unisex"),
    [products, gender],
  );
  const cats = useMemo(() => {
    const set = new Set(forGender.map((p) => p.category).filter(Boolean));
    return ["all", ...Array.from(set).sort()];
  }, [forGender]);
  const searchResults = useMemo(() => {
    if (!q) return null;
    const t = q.toLowerCase();
    return products.filter(
      (p) =>
        p.name.toLowerCase().includes(t) ||
        p.category.toLowerCase().includes(t) ||
        (p.weather || "").toLowerCase().includes(t) ||
        (p.gender || "").toLowerCase().includes(t) ||
        (p.colors || [p.color || ""]).some((c) => c.toLowerCase().includes(t)),
    );
  }, [q, products]);
  const filtered =
    searchResults ?? (cat === "all" ? forGender : forGender.filter((p) => p.category === cat));
  const shown = filtered.slice(0, limit);

  useEffect(() => {
    setContext(!q && cat !== "all" ? { kind: "category", category: cat, gender } : null);
  }, [cat, gender, q, setContext]);

  const title = q
    ? `Results for "${q}"`
    : gender === "women"
      ? "Women"
      : gender === "men"
        ? "Men"
        : "Top picks";

  return (
    <>
      <div className="promo">Free shipping over $150 · Ships from our Vancouver studio · Canada &amp; US</div>
      <StoreHeader brand={brand} active={gender} />

      <section className="hero rise">
        <h1>Move well. Rain or shine.</h1>
        <p>
          Technical apparel for studio days, wet commutes, and everything between. Not sure what to
          pick? Ask the assistant, it knows the whole catalog and cites every answer.
        </p>
        <div className="cta">
          <a className="btn btn-primary" href="#shop">
            Shop the collection
          </a>
          <button className="btn btn-ghost" onClick={() => openWith(null, null)}>
            Ask the assistant
          </button>
        </div>
      </section>

      <main className="shell" id="shop">
        <h2 className="section-title">{!q && cat !== "all" ? cap(cat) : title}</h2>
        <p className="section-sub">
          {filtered.length} piece{filtered.length === 1 ? "" : "s"}
          {filtered.length > shown.length ? ` · showing ${shown.length}` : ""}
        </p>
        {!q && (
          <div className="chips-row">
            {cats.map((c) => (
              <button key={c} className={c === cat ? "chip on" : "chip"} onClick={() => setCat(c)}>
                {c === "all" ? "All" : c}
              </button>
            ))}
          </div>
        )}

        {products.length === 0 ? (
          <p className="section-sub">Loading the collection...</p>
        ) : filtered.length === 0 ? (
          <p className="section-sub">
            No matches{q ? ` for "${q}"` : ""}. <Link href="/">Back to all products</Link>.
          </p>
        ) : (
          <>
            <div className="grid">
              {shown.map((p) => (
                <ProductCard key={p.id} p={p} />
              ))}
            </div>
            {filtered.length > shown.length && (
              <div className="see-more">
                <button className="btn btn-ghost" onClick={() => setLimit((n) => n + 12)}>
                  See more ({filtered.length - shown.length} more)
                </button>
              </div>
            )}
          </>
        )}
      </main>

      <StoreFooter brand={brand} />
    </>
  );
}

function ProductCard({ p }: { p: Product }) {
  const { add } = useCart();
  const [added, setAdded] = useState(false);
  const quickAdd = (e: React.MouseEvent) => {
    e.preventDefault();
    add({ id: p.id, name: p.name, price: p.price || 0, color: p.color || "", size: p.sizes[0] || "OS" });
    setAdded(true);
    setTimeout(() => setAdded(false), 1200);
  };
  return (
    <article className="card rise">
      <Link href={`/product/${p.id}`} className="card-link">
        <ImageTile category={p.category} color={p.color} />
        <div className="body">
          <p className="name">{p.name.replace(/^Aster /, "")}</p>
          <div className="sub">
            <span>{cap(p.category)}</span>
            <span className="price">{p.price != null ? `$${p.price.toFixed(0)}` : ""}</span>
          </div>
          <div className="swatches">
            {(p.colors || [p.color || ""]).slice(0, 4).map((c) => (
              <span key={c} className="sw-dot" style={{ background: colorHex(c) }} title={c} />
            ))}
          </div>
        </div>
      </Link>
      <button className="add" onClick={quickAdd}>
        {added ? "Added ✓" : "Quick add"}
      </button>
    </article>
  );
}
