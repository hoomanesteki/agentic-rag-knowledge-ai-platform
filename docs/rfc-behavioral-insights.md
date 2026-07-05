# RFC: behavioral insights layer

## Status

Proposed, future work. Nothing here is built. This is the design for a focus feature that would sit
on top of the current system, written so the architecture is ready when it is time.

## The idea

The website generates a stream of interaction signals: what shoppers chat about, where they land,
what they browse, what they buy, and what they complain about. On its own that is noise. Aggregated
and read with care, it becomes insight for people who make decisions: marketing, merchandising and
buyers, and the business. This layer turns interaction data into stakeholder-facing insight, and it
flags change: drift, spikes, and emerging trends.

Two things must be true from the start. It has to respect privacy, because this is real people. And
it has to earn trust, because an insight that is wrong or invented is worse than no insight.

## What each stakeholder gets

- Marketing: what shoppers ask for that the catalog does not cover well, which occasions and themes
  are rising, and which messages land. Answer: unmet-demand and rising-theme reports from chat and
  search intent.
- Buyers and merchandisers: which products draw interest but convert poorly, which get consistent
  fit or quality complaints (this reuses the enrichment consensus features), and where demand is
  moving. Answer: a product-signal view that ranks attention against outcome.
- The business: week-over-week shifts in intent mix, a spike in a category, a drift in sentiment.
  Answer: a small set of tracked indicators with change detection, not a dashboard of everything.

## Data and privacy

- Event schema. A thin, append-only event log: a session id (not a person), an event type (chat,
  land, view, add-to-cart, order, complaint), a coarse timestamp, and a minimal payload (the intent
  lane, the product, the derived theme). No message text is stored raw; it is reduced to a derived
  signal at capture and the raw text is dropped.
- Consent and minimization. Analytics events are captured under consent, tied to a session, not to
  an identity. This follows the same posture the app already documents (PIPEDA and a GDPR-style
  stance): consent, data minimization, retention limits, and no server-side family or personal PII
  beyond what a shopper has explicitly shared. Insights are computed on aggregates, and a cohort is
  only reported when it is large enough to not single anyone out.
- Retention. Raw events age out on a short window; the derived aggregates and indicators are what is
  kept.

## Batch versus streaming

- Streaming, for the cheap real-time thing only: a live counter for a spike alert (a category
  suddenly hot), and the consent and moderation gate at capture.
- Batch, for everything that needs a window: the intent-mix indicators, the rising-theme
  clustering, the drift comparison against last period, and the stakeholder reports. Insight needs a
  window for the same reason enrichment does, a single session is not a trend.

## How change is detected

This reuses the drift machinery. The intent mix is a distribution; a shift is a PSI or a distance
on rolling windows. Emerging themes come from clustering the derived signals and watching a cluster
grow. A sentiment drift is the enrichment features moving over time. Detection is automatic;
acting on an insight (a merchandising decision, a campaign) is a human reading a report, never an
automated action on the business.

## Where it plugs in

- It consumes the same traces the app already writes and the enrichment features already computed,
  so it does not add a parallel data path.
- It is a batch worker and a small serving surface (the reports), which fits the services ADR: one
  more scheduled container, not a new platform.
- Its outputs are read by people. This layer never changes what the assistant says or what the
  store sells on its own; it informs the humans who decide.

## Why not now

The value of this layer scales with traffic and with a real stakeholder asking for it. Building it
before either exists would be speculative. The point of this RFC is that the pieces it needs (the
event log, the enrichment features, the drift monitors, the batch worker pattern) are already in the
architecture, so it is a focused addition when the time comes, not a rebuild.

## Roles demonstrated

Data Architect (the event schema, the privacy model, the batch and streaming split) and Data
Scientist (the indicators, the clustering, and the change detection), with a product sense for what
each stakeholder actually needs.
