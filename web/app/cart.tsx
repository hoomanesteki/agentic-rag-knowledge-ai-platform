"use client";

import { createContext, useContext, useEffect, useState, ReactNode } from "react";

export type CartLine = {
  id: string;
  name: string;
  price: number;
  color: string;
  size: string;
  qty: number;
};

type CartCtx = {
  lines: CartLine[];
  count: number;
  subtotal: number;
  add: (line: Omit<CartLine, "qty">, qty?: number) => void;
  setQty: (id: string, size: string, color: string, qty: number) => void;
  remove: (id: string, size: string, color: string) => void;
  clear: () => void;
};

// a cart line is unique by product + size + color, so two colorways stay separate
const sameLine = (a: { id: string; size: string; color: string }, b: { id: string; size: string; color: string }) =>
  a.id === b.id && a.size === b.size && a.color === b.color;

const Ctx = createContext<CartCtx | null>(null);
const KEY = "aster_cart";

export function CartProvider({ children }: { children: ReactNode }) {
  const [lines, setLines] = useState<CartLine[]>([]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(KEY);
      if (raw) setLines(JSON.parse(raw));
    } catch {
      /* ignore a corrupt cart */
    }
  }, []);

  useEffect(() => {
    localStorage.setItem(KEY, JSON.stringify(lines));
  }, [lines]);

  const add: CartCtx["add"] = (line, qty = 1) =>
    setLines((all) => {
      const i = all.findIndex((l) => sameLine(l, line));
      if (i >= 0) {
        const copy = [...all];
        copy[i] = { ...copy[i], qty: copy[i].qty + qty };
        return copy;
      }
      return [...all, { ...line, qty }];
    });

  const setQty: CartCtx["setQty"] = (id, size, color, qty) =>
    setLines((all) =>
      all
        .map((l) => (sameLine(l, { id, size, color }) ? { ...l, qty: Math.max(0, qty) } : l))
        .filter((l) => l.qty > 0),
    );

  const remove: CartCtx["remove"] = (id, size, color) =>
    setLines((all) => all.filter((l) => !sameLine(l, { id, size, color })));

  const clear = () => setLines([]);

  const count = lines.reduce((n, l) => n + l.qty, 0);
  const subtotal = lines.reduce((s, l) => s + l.price * l.qty, 0);

  return (
    <Ctx.Provider value={{ lines, count, subtotal, add, setQty, remove, clear }}>
      {children}
    </Ctx.Provider>
  );
}

export function useCart(): CartCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useCart must be used inside CartProvider");
  return c;
}

export const FREE_SHIP = 150; // free standard shipping over this, matches the shipping knowledge
