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
  setQty: (id: string, size: string, qty: number) => void;
  remove: (id: string, size: string) => void;
  clear: () => void;
};

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
      const i = all.findIndex((l) => l.id === line.id && l.size === line.size);
      if (i >= 0) {
        const copy = [...all];
        copy[i] = { ...copy[i], qty: copy[i].qty + qty };
        return copy;
      }
      return [...all, { ...line, qty }];
    });

  const setQty: CartCtx["setQty"] = (id, size, qty) =>
    setLines((all) =>
      all
        .map((l) => (l.id === id && l.size === size ? { ...l, qty: Math.max(0, qty) } : l))
        .filter((l) => l.qty > 0),
    );

  const remove: CartCtx["remove"] = (id, size) =>
    setLines((all) => all.filter((l) => !(l.id === id && l.size === size)));

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
