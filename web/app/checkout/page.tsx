"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { FREE_SHIP, useCart } from "../cart";
import { fetchStore } from "../catalog";
import { useRequireGate } from "../gate";
import StoreHeader from "../StoreHeader";

export default function CheckoutPage() {
  const gateOk = useRequireGate();
  const { lines, subtotal, clear } = useCart();
  const [brand, setBrand] = useState("");
  const [done, setDone] = useState<string | null>(null);
  useEffect(() => {
    fetchStore().then((s) => setBrand(s.brand));
  }, []);
  if (!gateOk) return null;

  const shipping = subtotal >= FREE_SHIP || subtotal === 0 ? 0 : 8;

  function placeOrder(e: React.FormEvent) {
    e.preventDefault();
    const order = "AS" + Math.floor(100000 + Math.random() * 900000);
    clear();
    setDone(order);
  }

  if (done) {
    return (
      <>
        <StoreHeader brand={brand} />
        <main className="checkout done">
          <div className="confirm">
            <div className="check">✓</div>
            <h1>Order confirmed</h1>
            <p>
              Thanks for your order. Confirmation <strong>{done}</strong> is on its way to your inbox.
            </p>
            <p className="section-sub">
              Ships from our Vancouver studio. You will get tracking when it leaves the warehouse.
            </p>
            <Link href="/" className="btn btn-primary">
              Keep shopping
            </Link>
          </div>
        </main>
      </>
    );
  }

  return (
    <>
      <StoreHeader brand={brand} />
      <main className="checkout">
        <h1>Checkout</h1>
        {lines.length === 0 ? (
          <p className="section-sub">
            Your bag is empty. <Link href="/">Shop the collection</Link>.
          </p>
        ) : (
          <form className="co-grid" onSubmit={placeOrder}>
            <div className="co-fields">
              <h3>Contact</h3>
              <input required type="email" placeholder="Email" aria-label="Email" />
              <h3>Shipping address</h3>
              <input required placeholder="Full name" aria-label="Full name" />
              <input required placeholder="Address" aria-label="Address" />
              <div className="co-2">
                <input required placeholder="City" aria-label="City" />
                <input required placeholder="Postal / ZIP" aria-label="Postal code" />
              </div>
              <select aria-label="Country" defaultValue="CA">
                <option value="CA">Canada</option>
                <option value="US">United States</option>
              </select>
              <h3>Payment</h3>
              <input required placeholder="Card number" aria-label="Card number" inputMode="numeric" />
              <div className="co-2">
                <input required placeholder="MM / YY" aria-label="Expiry" />
                <input required placeholder="CVC" aria-label="CVC" />
              </div>
              <p className="co-note">Demo checkout — no real payment is taken.</p>
            </div>
            <aside className="co-summary">
              <h3>Order</h3>
              {lines.map((l) => (
                <div key={`${l.id}-${l.size}-${l.color}`} className="co-line">
                  <span>
                    {l.name.replace(/^Aster /, "")} × {l.qty}
                  </span>
                  <span>${(l.price * l.qty).toFixed(0)}</span>
                </div>
              ))}
              <div className="co-line">
                <span>Shipping</span>
                <span>{shipping === 0 ? "Free" : `$${shipping}`}</span>
              </div>
              <div className="co-line co-total">
                <span>Total</span>
                <span>${(subtotal + shipping).toFixed(0)}</span>
              </div>
              <button type="submit" className="btn btn-primary co-pay">
                Place order
              </button>
            </aside>
          </form>
        )}
      </main>
    </>
  );
}
