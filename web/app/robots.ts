import type { MetadataRoute } from "next";

// Private demo: ask all crawlers to stay out, so the shared link is not indexed or hammered.
export default function robots(): MetadataRoute.Robots {
  return { rules: { userAgent: "*", disallow: "/" } };
}
