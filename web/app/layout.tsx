import "./globals.css";
import type { ReactNode } from "react";

import { CartProvider } from "./cart";
import { ChatProvider } from "./ChatProvider";

export const metadata = {
  title: "Technical apparel — RAG demo",
  description: "A domain-swappable agentic RAG platform, shown as an apparel storefront.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <CartProvider>
          <ChatProvider>{children}</ChatProvider>
        </CartProvider>
      </body>
    </html>
  );
}
