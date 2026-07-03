"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

const KEY = "aster_gate";

export const GATE_EVENT = "aster-gatechange";

export function isGated(): boolean {
  if (typeof window === "undefined") return false;
  return !!localStorage.getItem(KEY);
}
export function setGate(token: string) {
  localStorage.setItem(KEY, token || "ok");
  // The gate token is a valid API token, so reuse it for the chat. In production, where the
  // frictionless demo-login is disabled, this is how a gated reviewer talks to the assistant
  // without a second sign-in.
  if (token) localStorage.setItem("skein_token", token);
  window.dispatchEvent(new Event(GATE_EVENT)); // let the chat provider mount the widget
}

// Demo pages call this: if the visitor has not passed the landing gate, send them back to it.
export function useRequireGate(): boolean {
  const router = useRouter();
  const [ok, setOk] = useState(false);
  useEffect(() => {
    if (isGated()) setOk(true);
    else router.replace("/");
  }, [router]);
  return ok;
}
