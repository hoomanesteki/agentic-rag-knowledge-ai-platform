"use client";

import { ReactNode, useEffect, useState } from "react";

import { fetchStore } from "./catalog";
import { useChat } from "./ChatProvider";
import { useRequireGate } from "./gate";
import StoreFooter from "./StoreFooter";
import StoreHeader from "./StoreHeader";

// Shared shell for the static info pages (shipping, payment, size guide, membership, contact).
// Keeps the gate, header, footer, and the "ask the assistant" hook in one place so each page is
// just its content. Brand text still comes from the API, so the engine stays domain agnostic.
export default function InfoPage({
  title,
  intro,
  ask,
  children,
}: {
  title: string;
  intro?: string;
  ask?: string; // if set, renders a button that opens the assistant with this prompt
  children: ReactNode;
}) {
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
      <main className="help info">
        <h1>{title}</h1>
        {intro && <p className="section-sub">{intro}</p>}
        {children}
        {ask && (
          <div className="info-ask">
            <button className="btn btn-ghost" onClick={() => openWith(ask, null)}>
              Ask the assistant
            </button>
          </div>
        )}
      </main>
      <StoreFooter brand={brand} />
    </>
  );
}
