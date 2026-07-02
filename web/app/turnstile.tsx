"use client";

import { useEffect, useRef, useState } from "react";

const SITE_KEY = process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY;

type Turnstile = {
  render: (el: HTMLElement, opts: Record<string, unknown>) => string;
  remove: (id: string) => void;
  reset: (id: string) => void;
};

function turnstileApi(): Turnstile | undefined {
  return (window as unknown as { turnstile?: Turnstile }).turnstile;
}

// Renders the Cloudflare Turnstile widget (only when NEXT_PUBLIC_TURNSTILE_SITE_KEY is set) and
// exposes the current token. Shared by the customer and admin logins so both send the captcha the
// server verifies on every /api/login; without it, admin login 403s once Turnstile is configured.
export function useTurnstile() {
  const [token, setToken] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement | null>(null);
  const widgetId = useRef<string | undefined>(undefined);

  useEffect(() => {
    if (!SITE_KEY) return;
    function render() {
      const ts = turnstileApi();
      if (!ts || !ref.current || widgetId.current !== undefined) return;
      widgetId.current = ts.render(ref.current, {
        sitekey: SITE_KEY,
        callback: (t: string) => setToken(t),
        "expired-callback": () => setToken(null),
      });
    }
    if (turnstileApi()) {
      render(); // script already loaded (e.g. a remount after sign-out)
    } else {
      let script = document.querySelector("script[data-turnstile]") as HTMLScriptElement | null;
      if (!script) {
        script = document.createElement("script");
        script.src = "https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit";
        script.async = true;
        script.defer = true;
        script.setAttribute("data-turnstile", "1");
        document.head.appendChild(script);
      }
      script.addEventListener("load", render, { once: true });
    }
    return () => {
      const ts = turnstileApi();
      if (ts && widgetId.current !== undefined) {
        try {
          ts.remove(widgetId.current);
        } catch {
          /* widget already gone */
        }
        widgetId.current = undefined;
      }
    };
  }, []);

  // Turnstile tokens are single-use; after a failed login the consumed token would make every
  // retry fail the captcha, so the caller resets the widget (and clears the token) on failure.
  function reset() {
    const ts = turnstileApi();
    if (ts && widgetId.current !== undefined) {
      try {
        ts.reset(widgetId.current);
      } catch {
        /* widget already gone */
      }
    }
    setToken(null);
  }

  // null when Turnstile is not configured, so the caller can render {widget} unconditionally
  const widget = SITE_KEY ? <div ref={ref} /> : null;
  return { token, widget, reset };
}
