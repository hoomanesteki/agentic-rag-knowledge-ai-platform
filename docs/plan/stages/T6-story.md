# T6 The story

**Theme:** T6 The story. **Status:** done.

## What I set out to do

Make it easy for anyone to see how the system works and why, at a glance and in depth.

## How I did it

- **README with hand-drawn diagrams.** Rewrote the README around four ASCII views: the system at a
  glance (browser and voice into the API, the LangGraph brain, the three specialists, the stores,
  and the flywheel), the data architecture (the manifest driving the dbt medallion, the semantic
  layer, and the graph), how one turn works (understand, retrieve, ground, answer or abstain), and
  observability and CI. Plain text so it reads in any viewer, and it names the Phase 2 additions
  (the dbt semantic layer, Langfuse, the guided experience).
- **A runnable notebook.** `notebooks/01-data-architecture.ipynb` walks the data layer offline:
  build the medallion, see gold come out clean and typed, watch PII get masked between bronze and
  gold, and pull a governed number from the semantic layer with a chart. matplotlib and pandas are
  an optional `notebook` extra, so the base install stays lean.
- **A stage index.** `docs/plan/` now links every stage's result note, so the whole build reads as
  a sequence of small, checked steps.

## What I tested

- The notebook's code cells run clean end to end offline (built the medallion, read gold into a
  frame, showed the bronze-to-gold PII masking, resolved a governed metric, drew the chart).
- The `.ipynb` is valid JSON (nbformat 4). `make check` stays green. README links resolve.

## Where the phase lands

Phase 2 is done: a tested dbt semantic layer, fuller data for both domains, Langfuse tracing, a
guided and polished experience, new governance and reproducibility tests in CI, and the story to
tie it together. Every claim in the README has a test or a doc behind it.
