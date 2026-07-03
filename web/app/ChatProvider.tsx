"use client";

import { usePathname } from "next/navigation";
import { createContext, useContext, useEffect, useState, ReactNode } from "react";

import type { PageContext } from "./catalog";
import ChatWidget from "./ChatWidget";
import { GATE_EVENT, isGated } from "./gate";

type ChatCtx = {
  open: boolean;
  setOpen: (v: boolean) => void;
  seed: string | null;
  context: PageContext;
  setContext: (c: PageContext) => void;
  openWith: (seed?: string | null, context?: PageContext) => void;
};

const Ctx = createContext<ChatCtx | null>(null);

export function ChatProvider({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const [seed, setSeed] = useState<string | null>(null);
  const [context, setContext] = useState<PageContext>(null);
  const [gated, setGated] = useState(false);
  const pathname = usePathname();
  const onAdmin = pathname?.startsWith("/admin");

  useEffect(() => {
    // only show the assistant once the visitor is inside the demo, never on the public landing page
    setGated(isGated());
    const onGate = () => setGated(true);
    window.addEventListener(GATE_EVENT, onGate);
    return () => window.removeEventListener(GATE_EVENT, onGate);
  }, []);

  const openWith: ChatCtx["openWith"] = (s = null, c = null) => {
    setSeed(s);
    if (c !== null) setContext(c);
    setOpen(true);
  };

  return (
    <Ctx.Provider value={{ open, setOpen, seed, context, setContext, openWith }}>
      {children}
      {gated && !onAdmin && (
        <ChatWidget open={open} setOpen={setOpen} seed={seed} context={context} />
      )}
    </Ctx.Provider>
  );
}

export function useChat(): ChatCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useChat must be used inside ChatProvider");
  return c;
}
