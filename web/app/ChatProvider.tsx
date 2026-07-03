"use client";

import { usePathname } from "next/navigation";
import { createContext, useContext, useState, ReactNode } from "react";

import type { PageContext } from "./catalog";
import ChatWidget from "./ChatWidget";

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
  const pathname = usePathname();
  const onAdmin = pathname?.startsWith("/admin");

  const openWith: ChatCtx["openWith"] = (s = null, c = null) => {
    setSeed(s);
    if (c !== null) setContext(c);
    setOpen(true);
  };

  return (
    <Ctx.Provider value={{ open, setOpen, seed, context, setContext, openWith }}>
      {children}
      {!onAdmin && <ChatWidget open={open} setOpen={setOpen} seed={seed} context={context} />}
    </Ctx.Provider>
  );
}

export function useChat(): ChatCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useChat must be used inside ChatProvider");
  return c;
}
