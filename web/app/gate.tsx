"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

const KEY = "aster_gate";

export function isGated(): boolean {
  if (typeof window === "undefined") return false;
  return !!localStorage.getItem(KEY);
}
export function setGate(token: string) {
  localStorage.setItem(KEY, token || "ok");
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
