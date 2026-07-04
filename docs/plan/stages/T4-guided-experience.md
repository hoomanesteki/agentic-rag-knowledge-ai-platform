# T4 Guided experience

**Theme:** T4 Guided experience. **Status:** done.

## What I set out to do

Make the product feel considered and never leave the user staring at a blank box. Two surfaces: the
customer chat should guide what to ask and set expectations, and the backoffice dashboards should
read at a glance.

## How I did it

**Guided chat.** Each domain pack declares a short set of starter prompts in its manifest, one per
capability: a product or account fact, a governed metric, a graph relationship, what reviews say,
and a policy. A new `/api/suggestions` endpoint serves them for the active domain (capped and
whitelisted to text/lang/kind so a pack cannot push arbitrary fields), so switching `DOMAIN`
switches the guidance and the engine hardcodes nothing. The chat shows them as clickable chips on
the empty screen, with a one-line hint on what to expect (grounded and cited, or an honest "I do
not know"), and as an "ask another" row after an answer. Clicking a chip submits it.

**Design polish.** A small design system on the existing tokens: dark mode via
`prefers-color-scheme`, soft shadows, a subtle rise animation (respecting reduced-motion), hover and
active states, a thinking indicator, and styled mic and speak controls. Calm and Apple-like without
a rebuild.

**Backoffice dashboard.** The admin console had no styling at all. Added a full admin stylesheet
(tables, cards, a nav across the admin pages) and metric cards with inline CSS bars on the answer-
quality view (grounding, abstain rate, escalation rate, helpful ratio), so the numbers are visible
at a glance, no chart library needed (CSP-safe). The insights page links straight to the Langfuse
traces and the MLflow runs.

## What I tested

- `make check` green (a new API test covers `/api/suggestions`: auth required, returns the active
  domain's capped, well-formed prompts; the domain serves six prompts across five capabilities).
- Web `tsc --noEmit` and `next lint` clean. The chat `send()` refactor lets a chip submit its text
  directly (React state is async, so the value is passed, not read back).
- Suggestion text is rendered through React escaping (no injection), and the endpoint sanitizes and
  caps what a pack can expose.

## Note

The chart visuals are intentionally CSS-only (inline bars, no chart dependency), so the web app
stays self-contained and there is no external asset to load. Richer historical charts (drift and
cost over time) can hang off the same pattern later.
