"use client";

import Link from "next/link";

// Shared footer for every store page. Brand text comes from the API (never hardcoded), so the
// engine stays domain agnostic.
export default function StoreFooter({ brand }: { brand?: string }) {
  const name = brand || "";
  const short = name.split(" ")[0] || "Store";
  return (
    <footer className="sfoot">
      <div className="sfoot-in">
        <div className="sfoot-brand-col">
          <div className="sfoot-brand">
            {short}
            <span>.</span>
          </div>
          <p>
            Technical apparel for studio days and wet commutes. A domain-swappable RAG demo on
            synthetic data, no real products or people.
          </p>
        </div>
        <div className="sfoot-col">
          <h4>Shop</h4>
          <Link href="/?g=women">Women</Link>
          <Link href="/?g=men">Men</Link>
          <Link href="/">New arrivals</Link>
        </div>
        <div className="sfoot-col">
          <h4>Help</h4>
          <Link href="/help">Shipping &amp; returns</Link>
          <Link href="/help">Sizing</Link>
          <Link href="/cart">Your bag</Link>
        </div>
        <div className="sfoot-col">
          <h4>More</h4>
          <a href="/admin">Backoffice</a>
          <a href="https://esteki.ca/" target="_blank" rel="noopener noreferrer">
            Built by esteki.ca
          </a>
        </div>
      </div>
      <div className="sfoot-bottom">
        <span>{name ? `${name} (demo)` : "Demo store"}. Ships from Vancouver to Canada and the US.</span>
        <span>Free shipping over $150</span>
      </div>
    </footer>
  );
}
