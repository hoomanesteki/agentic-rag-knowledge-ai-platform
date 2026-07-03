"use client";

import Link from "next/link";

import { useCart } from "./cart";
import { useChat } from "./ChatProvider";

export default function StoreHeader({ brand, active }: { brand: string; active?: string }) {
  const { count } = useCart();
  const { openWith } = useChat();
  const short = (brand || "Aster").split(" ")[0];
  const genders = [
    { key: "women", label: "Women" },
    { key: "men", label: "Men" },
    { key: "all", label: "All" },
  ];
  return (
    <header className="hdr">
      <div className="hdr-in">
        <Link href="/" className="brand">
          {short}
          <span>.</span>
        </Link>
        <nav className="nav">
          {genders.map((g) => (
            <Link
              key={g.key}
              href={g.key === "all" ? "/" : `/?g=${g.key}`}
              className={active === g.key ? "on" : ""}
            >
              {g.label}
            </Link>
          ))}
        </nav>
        <button
          className="chip"
          onClick={() => openWith(null, null)}
          style={{ marginLeft: "auto" }}
        >
          Ask
        </button>
        <Link href="/cart" className="cart-btn" aria-label="Cart">
          &#128722;
          {count > 0 && <span className="cart-count">{count}</span>}
        </Link>
      </div>
    </header>
  );
}
