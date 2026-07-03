"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { FREE_SHIP, useCart } from "../cart";
import { fetchStore } from "../catalog";
import { useRequireGate } from "../gate";
import StoreHeader from "../StoreHeader";

export default function CartPage() {
  const gateOk = useRequireGate();
  const { lines, subtotal, setQty, remove } = useCart();
  const [brand, setBrand] = useState("");
  useEffect(() => {
    fetchStore().then((s) => setBrand(s.brand));
  }, []);
  if (!gateOk) return null;

  const toFree = Math.max(0, FREE_SHIP - subtotal);
  const shipping = subtotal >= FREE_SHIP || subtotal === 0 ? 0 : 8;

  return (
    <>
      <StoreHeader brand={brand} />
      <main className="cartpage">
        <h1>Your bag</h1>
        {lines.length === 0 ? (
          <p className="section-sub">
            Your bag is empty. <Link href="/">Shop the collection</Link>.
          </p>
        ) : (
          <div className="cart-grid">
            <div className="cart-lines">
              {lines.map((l) => (
                <div key={`${l.id}-${l.size}`} className="cart-line">
                  <div className="cl-info">
                    <div className="cl-name">{l.name.replace(/^Aster /, "")}</div>
                    <div className="cl-meta">
                      {l.color} · Size {l.size}
                    </div>
                  </div>
                  <div className="cl-qty">
                    <button onClick={() => setQty(l.id, l.size, l.qty - 1)} aria-label="Decrease">
                      −
                    </button>
                    <span>{l.qty}</span>
                    <button onClick={() => setQty(l.id, l.size, l.qty + 1)} aria-label="Increase">
                      +
                    </button>
                  </div>
                  <div className="cl-price">${(l.price * l.qty).toFixed(0)}</div>
                  <button className="cl-remove" onClick={() => remove(l.id, l.size)} aria-label="Remove">
                    &times;
                  </button>
                </div>
              ))}
            </div>
            <aside className="cart-summary">
              <div className="cs-row">
                <span>Subtotal</span>
                <span>${subtotal.toFixed(0)}</span>
              </div>
              <div className="cs-row">
                <span>Shipping</span>
                <span>{shipping === 0 ? "Free" : `$${shipping}`}</span>
              </div>
              {toFree > 0 && (
                <div className="cs-free">Add ${toFree.toFixed(0)} more for free shipping</div>
              )}
              <div className="cs-row cs-total">
                <span>Total</span>
                <span>${(subtotal + shipping).toFixed(0)}</span>
              </div>
              <Link href="/checkout" className="btn btn-primary cs-checkout">
                Checkout
              </Link>
              <Link href="/" className="cs-continue">
                Continue shopping
              </Link>
            </aside>
          </div>
        )}
      </main>
    </>
  );
}
