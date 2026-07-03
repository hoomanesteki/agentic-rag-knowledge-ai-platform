"use client";

import { useEffect, useMemo, useState } from "react";

import ChatWidget from "./ChatWidget";
import { fetchStore, Product, swatchStyle } from "./catalog";

export default function Home() {
  const [products, setProducts] = useState<Product[]>([]);
  const [brand, setBrand] = useState("");
  const [cat, setCat] = useState("all");
  const [open, setOpen] = useState(false);
  const [seed, setSeed] = useState<string | null>(null);

  useEffect(() => {
    fetchStore().then((s) => {
      setProducts(s.products);
      setBrand(s.brand);
    });
  }, []);

  const short = brand.split(" ")[0] || "Store"; // a compact logo mark from the brand name

  const cats = useMemo(() => {
    const set = new Set(products.map((p) => p.category).filter(Boolean));
    return ["all", ...Array.from(set).sort()];
  }, [products]);

  const shown = cat === "all" ? products : products.filter((p) => p.category === cat);

  function askAbout(p: Product) {
    setSeed(`Tell me about the ${p.name}. Is it good for me?`);
    setOpen(true);
  }

  return (
    <>
      <header className="hdr">
        <div className="hdr-in">
          <div className="brand">
            {short}
            <span>.</span>
          </div>
          <nav className="nav">
            {cats.map((c) => (
              <button key={c} className={c === cat ? "on" : ""} onClick={() => setCat(c)}>
                {c === "all" ? "All" : c}
              </button>
            ))}
          </nav>
          <span className="mini">Synthetic demo store</span>
        </div>
      </header>

      <section className="hero rise">
        <h1>Move well. Rain or shine.</h1>
        <p>
          Technical apparel for studio days, wet commutes, and everything between. Not sure what to
          pick? Ask our assistant, it knows the whole catalog and always cites its answer.
        </p>
        <div className="cta">
          <a className="btn btn-primary" href="#shop">
            Shop the collection
          </a>
          <button
            className="btn btn-ghost"
            onClick={() => {
              setSeed(null);
              setOpen(true);
            }}
          >
            Ask the assistant
          </button>
        </div>
      </section>

      <main className="shell" id="shop">
        <h2 className="section-title">{cat === "all" ? "New arrivals" : cap(cat)}</h2>
        <p className="section-sub">
          {shown.length} piece{shown.length === 1 ? "" : "s"} in the collection
        </p>
        <div className="chips-row">
          {cats.map((c) => (
            <button key={c} className={c === cat ? "chip on" : "chip"} onClick={() => setCat(c)}>
              {c === "all" ? "All" : c}
            </button>
          ))}
        </div>

        {products.length === 0 ? (
          <p className="section-sub">
            The catalog is loading, or the store has not been built yet. Run the data pipeline to
            populate it.
          </p>
        ) : (
          <div className="grid">
            {shown.map((p) => (
              <article key={p.id} className="card rise" onClick={() => askAbout(p)}>
                <div className="swatch" style={swatchStyle(p.color)}>
                  {p.color && <span className="tag">{p.color}</span>}
                </div>
                <div className="body">
                  <p className="name">{p.name}</p>
                  <div className="sub">
                    <span>{cap(p.category)}</span>
                    <span className="price">{p.price != null ? `$${p.price.toFixed(0)}` : ""}</span>
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}
      </main>

      <footer className="foot">
        <div className="foot-in">
          <span>
            {brand ? `${brand}, a` : "A"} synthetic demo brand. No real products or people.
          </span>
          <a href="/admin">Backoffice</a>
        </div>
      </footer>

      <ChatWidget open={open} setOpen={setOpen} seed={seed} />
    </>
  );
}

function cap(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}
