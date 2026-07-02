import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "Skein",
  description: "Grounded, cited answers over your data.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
