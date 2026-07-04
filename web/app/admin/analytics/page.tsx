"use client";

import { useEffect, useState } from "react";

import { BarList, Donut, Funnel, Sparkbars } from "../charts";
import { Hint } from "../Hint";
import { AdminNav } from "../nav";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Analytics = {
  kpis: Record<string, number>;
  sessions_by_day: { day: string; count: number }[];
  top_searches: { term: string; count: number }[];
  top_questions: { question: string; count: number }[];
  top_products: { name: string; count: number }[];
  by_category: { label: string; pct: number }[];
  by_device: { label: string; pct: number }[];
  by_country: { label: string; pct: number }[];
  funnel: { step: string; count: number }[];
  unmet_demand?: { term: string; count: number; reason: string }[];
};
type Gap = { question: string; count: number };

const money = (n: number) => "$" + (n || 0).toLocaleString();
const num = (n: number) => (n || 0).toLocaleString();

export default function AnalyticsPage() {
  const [mounted, setMounted] = useState(false);
  const [a, setA] = useState<Analytics | null>(null);
  const [gaps, setGaps] = useState<Gap[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setMounted(true);
    const token = localStorage.getItem("skein_admin_token");
    if (!token) {
      setError("Sign in on the /admin page first.");
      return;
    }
    const headers = { Authorization: `Bearer ${token}` };
    fetch(`${API_BASE}/api/admin/analytics`, { headers })
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setA)
      .catch(() => setError("Could not load analytics (are you an admin?)."));
    fetch(`${API_BASE}/api/admin/gaps`, { headers })
      .then((r) => (r.ok ? r.json() : { gaps: [] }))
      .then((d) => setGaps(d.gaps || []))
      .catch(() => {});
  }, []);

  if (!mounted) return null;
  if (error)
    return (
      <main className="admin">
        <AdminNav />
        <p className="error">{error}</p>
      </main>
    );
  if (!a || !a.kpis)
    return (
      <main className="admin">
        <AdminNav />
        <p className="meta">
          {a ? "No analytics for this domain yet." : "Loading the dashboard..."}
        </p>
      </main>
    );

  const k = a.kpis;
  const kpis = [
    { label: "Sessions", value: num(k.sessions),
      hint: "Business: visits to the store in the window, the top of the funnel. Technical: distinct browser sessions counted from request traces, deduped by session id." },
    { label: "Visitors", value: num(k.visitors),
      hint: "Business: how many different people came, so repeat visits do not inflate reach. Technical: unique visitor ids across sessions; visitors under sessions means people return." },
    { label: "Orders", value: num(k.orders),
      hint: "Business: completed purchases, the outcome the store exists for. Technical: count of checkout-complete events in the window." },
    { label: "Revenue", value: money(k.revenue),
      hint: "Business: total sales value booked in the window. Technical: summed order totals in the store currency, gross of returns." },
    { label: "Conversion", value: `${k.conversion}%`,
      hint: "Business: the share of visits that buy, the single clearest health signal. Technical: orders divided by sessions; small denominators make it swing, so read it with the session count." },
    { label: "Avg order", value: money(k.aov),
      hint: "Business: how much a shopper spends per purchase; upsell and bundling move it. Technical: revenue divided by orders (AOV)." },
    { label: "Searches", value: num(k.searches),
      hint: "Business: how hard shoppers are hunting; high search with low conversion signals findability gaps. Technical: count of search and assistant retrieval queries in the window." },
    { label: "Added to bag", value: num(k.add_to_cart),
      hint: "Business: intent to buy, the step just before checkout; the gap to orders is where carts are abandoned. Technical: count of add-to-cart events." },
  ];

  return (
    <main className="admin">
      <AdminNav />
      <h1>Store overview</h1>
      <p className="ax-intro">
        The business view behind the assistant: traffic, what shoppers search and ask, and how the
        funnel converts. Last 30 days, {num(k.sessions)} sessions.
      </p>

      <div className="stats">
        {kpis.map((kp) => (
          <div key={kp.label} className="stat">
            <div className="stat-label">
              {kp.label}
              <Hint text={kp.hint} />
            </div>
            <div className="stat-value">{kp.value}</div>
          </div>
        ))}
      </div>

      <div className="ax-grid">
        <div className="ax-card ax-wide">
          <h3>Sessions per day
            <Hint text="Business: the traffic trend, so you can tie spikes to campaigns or outages. Technical: daily session counts from request traces over the window." />
          </h3>
          <Sparkbars data={a.sessions_by_day} />
        </div>
        <div className="ax-card">
          <h3>Conversion funnel
            <Hint text="Business: where shoppers drop off between arriving and buying; the biggest fall-off is the best place to invest. Technical: session to search to add-to-cart to order, each step a share of the one above." />
          </h3>
          <Funnel steps={a.funnel} />
        </div>
        <div className="ax-card">
          <h3>Top searches
            <Hint text="Business: what shoppers are actively hunting for, the demand signal for merchandising. Technical: most frequent search terms, ranked by count." />
          </h3>
          <BarList items={a.top_searches.map((s) => ({ label: s.term, count: s.count }))} />
        </div>
        <div className="ax-card">
          <h3>What shoppers ask the assistant
            <Hint text="Business: the questions people trust the assistant with, in their own words; a content and FAQ roadmap. Technical: most frequent assistant queries, clustered by count." />
          </h3>
          <BarList items={a.top_questions.map((s) => ({ label: s.question, count: s.count }))} />
        </div>
        <div className="ax-card">
          <h3>Most viewed products
            <Hint text="Business: what is pulling attention; pair with conversion to spot lookers versus buyers. Technical: product detail views ranked by count." />
          </h3>
          <BarList items={a.top_products.map((s) => ({ label: s.name, count: s.count }))} />
        </div>
        <div className="ax-card">
          <h3>Traffic by category
            <Hint text="Business: which departments draw the crowd, so stock and promotion follow demand. Technical: session share by product category, top six." />
          </h3>
          <Donut slices={a.by_category.slice(0, 6)} />
        </div>
        <div className="ax-card">
          <h3>By device
            <Hint text="Business: mobile versus desktop mix, which decides where UX effort pays off. Technical: session share by device class from the request user agent." />
          </h3>
          <Donut slices={a.by_device} />
        </div>
        <div className="ax-card">
          <h3>By country
            <Hint text="Business: where demand is, informing shipping, currency, and language priorities. Technical: session share by country inferred from the request." />
          </h3>
          <Donut slices={a.by_country} />
        </div>
        <div className="ax-card ax-wide">
          <h3>Questions we could not answer well
            <Hint text="Business: unmet questions, ranked; each one fixed lifts trust and conversion. Technical: turns where the assistant abstained or scored low grounding, grouped by question." />
          </h3>
          {gaps.length === 0 ? (
            <p className="meta">Nothing flagged. The assistant is answering confidently.</p>
          ) : (
            <BarList items={gaps.slice(0, 8).map((g) => ({ label: g.question, count: g.count }))} />
          )}
        </div>
        {a.unmet_demand && a.unmet_demand.length > 0 && (
          <div className="ax-card ax-wide">
            <h3>Unmet demand: what shoppers want that we don&apos;t offer
              <Hint text="Business: revenue we cannot capture yet, ranked by how often it comes up; the merchandising and roadmap worklist. Technical: searches and questions with no matching product or policy doc, grouped by term." />
            </h3>
            <p className="meta">
              Money left on the table: things people searched or asked for that we can&apos;t serve
              today. Each is a merchandising decision, stock it, expand sizes or regions, or
              add the feature.
            </p>
            <ul className="unmet-list">
              {a.unmet_demand.map((u) => (
                <li key={u.term}>
                  <span className="unmet-count">{u.count}</span>
                  <span className="unmet-term">{u.term}</span>
                  <span className="unmet-reason">{u.reason}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </main>
  );
}
