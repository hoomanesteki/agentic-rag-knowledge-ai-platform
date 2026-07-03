"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { useCart } from "../../cart";
import { colorHex, fetchProduct, fetchStore, ProductDetail } from "../../catalog";
import { useChat } from "../../ChatProvider";
import { useRequireGate } from "../../gate";
import ImageTile from "../../ImageTile";
import StoreFooter from "../../StoreFooter";
import StoreHeader from "../../StoreHeader";

function cap(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

export default function ProductPage({ params }: { params: { id: string } }) {
  const gateOk = useRequireGate();
  const [p, setP] = useState<ProductDetail | null>(null);
  const [missing, setMissing] = useState(false);
  const [brand, setBrand] = useState("");
  const [size, setSize] = useState("");
  const [color, setColor] = useState("");
  const [added, setAdded] = useState(false);
  const { add } = useCart();
  const { openWith, setContext } = useChat();

  useEffect(() => {
    fetchProduct(params.id).then((d) => {
      if (!d) {
        setMissing(true);
        return;
      }
      setP(d);
      setSize(d.sizes[0] || "OS");
      setColor(d.color || (d.colors && d.colors[0]) || "");
    });
    fetchStore().then((s) => setBrand(s.brand));
  }, [params.id]);

  useEffect(() => {
    if (p) setContext({ kind: "product", id: p.id, name: p.name, category: p.category });
    return () => setContext(null);
  }, [p, setContext]);

  if (!gateOk) return null; // redirecting to the landing gate
  if (missing) {
    return (
      <>
        <StoreHeader brand={brand} />
        <main className="pdp">
          <p className="section-sub">
            That product could not be found. <Link href="/">Back to the store</Link>.
          </p>
        </main>
      </>
    );
  }
  if (!p) {
    return (
      <>
        <StoreHeader brand={brand} />
        <main className="pdp">
          <p className="section-sub">Loading...</p>
        </main>
      </>
    );
  }

  const short = p.name.replace(/^Aster /, "");
  const stock = p.stock ?? 0;
  const stockLabel = stock <= 0 ? "Out of stock" : stock < 10 ? `Low stock, ${stock} left` : "In stock";

  const addToCart = () => {
    add({ id: p.id, name: p.name, price: p.price || 0, color, size });
    setAdded(true);
    setTimeout(() => setAdded(false), 1400);
  };

  return (
    <>
      <StoreHeader brand={brand} />
      <main className="pdp">
        <div className="pdp-media">
          <ImageTile category={p.category} color={color} className="pdp-tile" />
        </div>
        <div className="pdp-info">
          <Link href="/" className="back">
            ← Back to store
          </Link>
          <h1>{short}</h1>
          <div className="pdp-meta">
            {cap(p.category)}
            {p.gender && p.gender !== "unisex" ? ` · ${cap(p.gender)}` : ""}
            {p.weather ? ` · ${cap(p.weather)}` : ""}
          </div>
          <div className="pdp-price">{p.price != null ? `$${p.price.toFixed(0)}` : ""}</div>
          <p className={`pdp-stock ${stock <= 0 ? "out" : stock < 10 ? "low" : "ok"}`}>{stockLabel}</p>
          {p.description && <p className="pdp-desc">{p.description}</p>}

          {p.colors && p.colors.length > 0 && (
            <div className="pdp-opts">
              <div className="opt-label">Color: {color}</div>
              <div className="opt-row">
                {p.colors.map((c) => (
                  <button
                    key={c}
                    className={`sw-pick ${c === color ? "on" : ""}`}
                    style={{ background: colorHex(c) }}
                    onClick={() => setColor(c)}
                    aria-label={c}
                    title={c}
                  />
                ))}
              </div>
            </div>
          )}

          <div className="pdp-opts">
            <div className="opt-label">Size</div>
            <div className="opt-row">
              {p.sizes.map((s) => (
                <button
                  key={s}
                  className={`size-pick ${s === size ? "on" : ""}`}
                  onClick={() => setSize(s)}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>

          <div className="pdp-actions">
            <button className="btn btn-primary" onClick={addToCart} disabled={stock <= 0}>
              {added ? "Added to bag ✓" : stock <= 0 ? "Out of stock" : "Add to bag"}
            </button>
            <button
              className="btn btn-ghost"
              onClick={() =>
                openWith(`Is the ${short} a good pick for me?`, {
                  kind: "product",
                  id: p.id,
                  name: p.name,
                  category: p.category,
                })
              }
            >
              Ask about this
            </button>
          </div>

          {p.reviews && p.reviews.length > 0 && (
            <div className="pdp-reviews">
              <h3>What customers say</h3>
              {p.reviews.map((r, i) => (
                <div key={i} className="review">
                  <span className="stars">{"★".repeat(r.rating || 5)}</span>
                  <span>{r.text}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
      <StoreFooter brand={brand} />
    </>
  );
}
