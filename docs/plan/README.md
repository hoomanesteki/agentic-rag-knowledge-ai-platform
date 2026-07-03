# Phase 2 plan: data architecture, observability, and experience

M0 to M9 built the platform: a domain-swappable agentic RAG engine with a medallion lakehouse, a
governed metric layer, a knowledge graph, an eval harness, and a web app. That is the working
skeleton. Phase 2 turns it into something a data team would actually trust and a user would enjoy.

I run this the way I run any build: one big picture, split into themes, each theme split into small
stages I can finish and check in a sitting. Every stage has a short spec, a "done when", and a
plain result note (what I did, what I tested, what passed or failed). Those notes live in
[stages/](stages/). I keep it honest: if a number is offline-zero or a piece is deferred, I say so.

## Why this shape

The north star for Phase 2 is the data-architecture discipline: a single source of truth, modeled
in tested and documented transformations, with a semantic layer on top that analysts, dashboards,
and the AI agent all read the same way. Around that I want the things that make it real: good
sample data for both domains, end-to-end observability I can watch on a dashboard, an experience
that guides the user instead of leaving them staring at a blank box, and reproducibility so anyone
can clone and run it.

## Themes

| Theme | What it delivers | Why it matters |
| --- | --- | --- |
| T1 Semantic layer in dbt | The medallion and metrics rebuilt as dbt models with tests, lineage, and docs | One tested, documented source of truth |
| T2 Sample data and knowledge | Fuller, realistic data and knowledge for both domains | The demo answers well the moment you pick a domain |
| T3 Observability with Langfuse | Traced LLM and graph calls, plus the two dashboards | I can see and debug every answer end to end |
| T4 Guided experience | Suggested prompts, guidance, and a clean design | The user always knows what to ask and what to expect |
| T5 Reproducibility and tests | New data-quality, governance, and reproducibility tests in CI | It runs the same everywhere, and regressions are caught |
| T6 The story | README with diagrams, notebooks, and stage reports | Anyone can follow how it works and why |

## T1. Semantic layer in dbt

The engine still ingests any domain from its manifest. On top of that, the analytics and semantic
layer is modeled in dbt (dbt-duckdb, local, no warehouse bill), so the medallion is tested,
documented, and lineage-traced like a production data stack.

- T1.1 Stand up dbt-duckdb; generate dbt sources and staging (bronze, silver) from the domain
  manifest so the models stay manifest-driven, not hand-copied per domain.
- T1.2 Gold marts as dbt models; the metric layer reads dbt's gold, one build path.
- T1.3 dbt tests: schema tests (not_null, unique, accepted_values, relationships) plus custom
  data-quality and governance tests (every declared PII column is masked in silver and gold).
- T1.4 The semantic layer: governed metrics defined once as semantic objects, the single source of
  truth the app, the eval, and the dashboards all read.
- T1.5 Lineage and docs: dbt docs (the DAG) and exposures that name the RAG app and the dashboards
  as downstream consumers, so I can trace what data feeds any output.

## T2. Sample data and knowledge for both domains

- T2.1 Apparel: fuller products, reviews, sales, suppliers, stores, plus a short company-knowledge
  set, sized so retrieval, the graph, and the metrics all return good answers.
- T2.2 SaaS support: the same treatment for plans, tickets, articles, and company knowledge.
- T2.3 Prove both: pick each domain, run the whole path, confirm real answers on common questions.

## T3. Observability with Langfuse

- T3.1 Add Langfuse and wire it through the LangGraph brain and the LLM and embedding adapters, so
  every turn is a trace with spans and scores (works natively with LangChain and LangGraph).
- T3.2 Backoffice dashboard: quality, drift, cost, latency, the dbt test status, and lineage in one
  place for me.
- T3.3 User dashboard: the chat itself, with the guidance from T4.

## T4. Guided experience

- T4.1 A small design system: type scale, spacing, color, and motion, applied to the chat and the
  admin, clean and calm.
- T4.2 Guidance: per-domain starter prompts, suggested follow-ups drawn from what was just asked,
  and short hints on what to expect (an answer, a number, a citation, or an honest "I do not know").
- T4.3 Polished admin charts so the dashboards read at a glance.

## T5. Reproducibility and tests

- T5.1 One-command reproducibility: setup, seed, build, run, all pinned and deterministic; verified
  end to end from a clean clone.
- T5.2 New tests: data quality, data governance, the semantic layer, and a reproducibility check,
  all offline and in CI alongside the dbt tests.
- T5.3 CI runs dbt build and tests, the eval gate, and the dependency audit on every change.

## T6. The story

- T6.1 README rewritten with ASCII architecture diagrams (a few views, not one), including the
  data-architecture and dbt thinking.
- T6.2 Notebooks that walk the pipeline step by step with charts, so the results are reproducible
  and readable.
- T6.3 The model-selection note ([../model-selection.md](../model-selection.md)) and the per-stage
  reports under [stages/](stages/).

## How I sequenced it

T1 first, because the semantic layer is the foundation everything else reads. T2 rode along inside
T1 (better data, modeled in dbt). Then T3 so I can watch it, T4 so it feels good to use, T5 so it
stays correct, and T6 so it reads well. Each stage is small, reviewed by an independent model before
it merges, and green on `make check`.

## Stage reports

Each stage has a short, plain result note: what I did, what I tested, what passed.

- [T1.1 The medallion in dbt](stages/T1.1-dbt-medallion.md)
- [T1.2 The semantic layer, parity, and lineage](stages/T1.2-semantic-layer.md)
- [T2 Sample data and company knowledge](stages/T2-sample-data.md)
- [T3 Observability with Langfuse](stages/T3-langfuse.md)
- [T4 Guided experience](stages/T4-guided-experience.md)
- [T5 Reproducibility and tests](stages/T5-reproducibility-tests.md)
- [T6 The story](stages/T6-story.md)
