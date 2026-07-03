"use client";

import { useEffect, useState } from "react";

import { fetchStore } from "../catalog";
import { useChat } from "../ChatProvider";
import { useRequireGate } from "../gate";
import StoreFooter from "../StoreFooter";
import StoreHeader from "../StoreHeader";

const TOPICS = [
  {
    topic: "shipping",
    title: "Shipping & delivery",
    body: "We ship from our Vancouver, BC studio to every province in Canada and every state in the US (including Alaska and Hawaii). Standard shipping is free over $150, otherwise a flat $8, and arrives in 2–8 business days depending on distance. Express is a flat $15 (1–2 days). We do not ship outside Canada and the US.",
  },
  {
    topic: "returns",
    title: "Returns & exchanges",
    body: "Return unworn items with tags within 30 days for a full refund. Returns are free with a prepaid label. Exchanges ship as soon as the carrier scans your return, at no extra charge. Refunds land in 5–7 business days.",
  },
  {
    topic: "sizing",
    title: "Sizing & fit",
    body: "Sizes run XS–XL, with one-size for bags and accessories. Most pieces run true to size; leggings are compressive, so size up for a looser feel. Every product page has a measurement chart and reviews often mention fit.",
  },
  {
    topic: "payment",
    title: "Payment",
    body: "We accept Visa, Mastercard, American Express, Discover, Apple Pay, Google Pay, PayPal, and Shop Pay, plus interest-free installments with Afterpay over $50.",
  },
  {
    topic: "care",
    title: "Product care",
    body: "Machine wash cold with like colors and hang dry or tumble dry low. Skip fabric softener on performance fabrics, since it reduces moisture wicking. Merino can be washed cold on a wool cycle.",
  },
  {
    topic: "warranty",
    title: "Quality guarantee",
    body: "Every piece is covered against manufacturing defects like seams and zippers, and we repair or replace. Normal wear and misuse are not covered.",
  },
  {
    topic: "membership",
    title: "Aster Circle membership",
    body: "Free to join. Earn 1 point per dollar, get early access to new drops, member shipping offers, and a birthday reward. Redeem points for discounts at checkout.",
  },
  {
    topic: "support",
    title: "Contact & support",
    body: "Chat and email 7 days a week, 8am–8pm Eastern. If we can't resolve something, we escalate to a human specialist who follows up within one business day.",
  },
];

export default function HelpPage() {
  const gateOk = useRequireGate();
  const [brand, setBrand] = useState("");
  const { openWith } = useChat();
  useEffect(() => {
    fetchStore().then((s) => setBrand(s.brand));
  }, []);
  if (!gateOk) return null;

  return (
    <>
      <StoreHeader brand={brand} />
      <main className="help">
        <h1>Help &amp; policies</h1>
        <p className="section-sub">
          Everything the assistant knows, in one place. Have a specific question? Just ask.
        </p>
        <div className="help-grid">
          {TOPICS.map((t) => (
            <section key={t.topic} className="help-card">
              <h3>{t.title}</h3>
              <p>{t.body}</p>
              <button
                className="chip"
                onClick={() => openWith(`Tell me more about ${t.topic}`, { kind: "help", topic: t.topic })}
              >
                Ask about {t.topic}
              </button>
            </section>
          ))}
        </div>
      </main>
      <StoreFooter brand={brand} />
    </>
  );
}
