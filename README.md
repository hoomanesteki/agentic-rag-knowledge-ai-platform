# Skein Lite

A local-first, domain-swappable **agentic RAG platform**. It answers questions over a mix of
structured and unstructured data, grounds every answer in citations (or abstains), routes hard
questions through a supervisor that coordinates specialist agents, and hands off to a human when it
is unsure, then learns from that answer. One engine serves any topic: a domain is a config folder,
not code.

Built step by step, MVP first, each step reviewed by an independent model before it merged.
Everything runs offline on fakes for tests and CI; the hosted models (Groq, Cohere) and the stores
(Qdrant, DuckDB, Neo4j, Postgres) are config swaps.

## What it does

- **Grounded or honest.** Hybrid retrieval (dense + sparse, RRF) with a reranker, sentence-level
  citation checks, and an abstain gate. Retrieved text is sanitized against prompt injection.
- **A brain, not a prompt.** A LangGraph state machine (understand, dispatch, reconcile) with a
  **supervisor** that dispatches to three **specialists** (a Retriever, a governed **Metrics**
  agent, and a knowledge-**Graph** agent) and reconciles their findings (a governed number beats a
  contradicting review), wrapped by a gate and a bounded agent loop that retries before it escalates.
- **A real data stack.** The medallion (bronze, silver, gold) is modeled in **dbt** with schema,
  relationship, and PII-masking tests, generated from each domain's manifest. A **semantic layer**
  (`metrics.yaml`) is the single source of truth the agent, the eval, and the dashboards all read.
- **Governed numbers.** A read-only DuckDB metric layer answers "what is the return rate for size M"
  from a validated query, never free-form SQL, and the number is cited as its own evidence.
- **A knowledge graph.** Neo4j nodes and typed edges built from gold; relational questions ("which
  supplier makes X") answer from the graph.
- **Human-in-the-loop flywheel.** Escalations land in a review queue; an operator answers; the answer
  becomes a retrievable verified chunk and grows the eval set.
- **Observability.** Langfuse traces every turn and LLM call (model, tokens, latency, cost); MLflow,
  a faithful RAGAS eval, four drift monitors, and an offline CI eval gate cover the rest.
- **Guided and voiced.** Per-domain starter prompts, spoken input (Groq Whisper) and output, a clean
  chat, and a backoffice dashboard.
- **One engine, two domains.** `apparel_ecommerce` and `saas_support`, proven in CI on every commit.

## The system at a glance

```text
     Browser (chat + /admin)          Voice
          |  text                       |  audio
          v                             v
   +----------------------------------------------------+
   |                     FastAPI                         |  JWT auth, rate limit,
   |    /chat (SSE)    /transcribe    /api/admin/*       |  Turnstile, degraded mode
   +---------------------------+------------------------+
                               |  CHAT_BRAIN=agent
                               v
   +----------------------------------------------------+
   |                 The brain (LangGraph)              |
   |    understand  ->  dispatch  ->  reconcile  -> gate |
   |    (rewrite,       (supervisor   (merge,      (auto |
   |     route)          fans out)     rank,        /    |
   |                         |         resolve)   escalate)
   |            +------------+------------+               |
   |            v            v            v               |
   |       Retriever      Metrics       Graph             |  three specialists,
   |       specialist     specialist    specialist        |  one Finding contract
   +------------|------------|------------|--------------+
                v            v            v
          +----------+ +-----------+ +-----------+
          |  Qdrant  | |  DuckDB   | |   Neo4j   |
          |  hybrid  | | governed  | | knowledge |
          |  vectors | | metrics   | |   graph   |
          +----------+ +-----------+ +-----------+
                ^
                |  verified answers grow the index (the flywheel)
          +---------------------------------+
          |  review queue  ->  /admin        |  human in the loop
          +---------------------------------+
```

## The data architecture

The engine reads only a domain's manifest, so the same code builds any domain. On top of that, the
analytics and semantic layer is real dbt: tested, documented, and lineage-traced.

```text
   domains/<name>/          the pack: data + a manifest, no engine code
   +----------------------------------------------------------------+
   |  seed/structured/*.csv        seed/unstructured/*.jsonl        |
   |  domain.yaml  (schema, PII, graph edges, metrics, suggestions) |
   +---------------------------+------------------------------------+
                               |  the manifest drives everything
         structured           |            unstructured
              v                |                  v
   +--------------------------+|      +-------------------------+
   |  dbt medallion (DuckDB)  ||      |  chunk + context prefix |
   |                          ||      |  Cohere embeddings      |
   |   bronze  raw text       ||      |          v              |
   |     v     (lineage)      ||      |  Qdrant hybrid index    |
   |   silver  typed + PII    ||      +-------------------------+
   |     v     masked         ||
   |   gold    curated        ||   dbt tests on every build:
   +-----------+--------------+|     not_null, unique, relationships,
               |               |     is_masked (PII never reaches gold raw)
        +------+------+        |
        v             v        |   semantic layer: metrics.yaml is the single
   +---------+  +-----------+  |   source of truth, read by the agent, the
   | metric  |  | knowledge |  |   eval, and the dashboards. dbt exposures
   | layer   |  |  graph    |  |   name those consumers, so lineage answers
   | (read-  |  | (gold ->  |  |   "what does this table feed".
   | only)   |  |  Neo4j)   |  |
   +---------+  +-----------+  |
                               v
     the same transform runs two ways (an in-app Python builder and dbt);
     a parity test proves the gold is byte-identical.
```

Nothing under the engine names a product, metric, or brand; those live only in `domains/<name>/`,
enforced by a leak linter in `make check`. See [docs/semantic-layer.md](docs/semantic-layer.md).

## How one turn works

```text
   question
      |
      v
   understand ----- rewrite a follow-up into a standalone question, pick a route
      |
      v
   retrieve -------- dense + sparse, fused with RRF, then reranked
      |
      v
   ground ---------- sentence-level citation check; retrieved text is sandboxed
      |
      +-- confident? -- yes --> answer with [1][2] citations
      |                  no
      +-- in scope? ---- no --> "I do not have enough information" (abstain)
                         yes
                          +--> agent loop retries the hard tail, then escalates
```

## Observability and CI

```text
   every turn  --> Langfuse trace: model, tokens, latency, cost
               \-> request trace --> /admin dashboard: quality, health, gaps
                                \--> MLflow runs
                                \--> four drift monitors
                                \--> RAGAS answer-quality eval

   every commit --> CI: make check (lint, tests, validation, leak, eval gate)
                        + dbt build and tests (both domains) + dependency audit
```

## Quick start

Needs [uv](https://docs.astral.sh/uv/) (it manages Python 3.12) and Docker.

```bash
make setup                 # venv + locked dependencies
cp .env.example .env       # fill in GROQ_API_KEY and VOYAGE_API_KEY for real runs
make check                 # lint, tests, dbt-ready checks, eval gate (fully offline)
make doctor                # if a step hangs or fails, this says why (Docker, .env, keys)
make up                    # Qdrant, Postgres, Neo4j, MLflow in Docker (preflighted)
make dbt-build             # build + test the semantic layer (medallion + governance tests)
make ingest && make graph-load                     # build the vector index and the graph
make serve                 # API on :8000     (set CHAT_BRAIN=agent for the full brain)
cd web && npm install && npm run dev             # web chat on :3000
```

Demo login: `demo` / `Canada54321`. Admin console at `/admin` (`admin` / `skein-admin-2026`).
Switch topic with `DOMAIN=saas_support` (the starter prompts switch with it). Voice needs
`TRANSCRIBE_PROVIDER=groq`; tracing needs the `LANGFUSE_*` keys. `make reproduce` runs the whole
offline verification in one command.

## Evaluate

```bash
make eval        # hit@k, MRR, entity recall, abstain recall, false-abstain rate on the golden set
make ablation    # dense vs hybrid vs +rerank, per language -> docs/eval-report.md
make ragas       # faithfulness, answer relevance, context precision/recall (LLM judge)
make gate        # the offline CI eval gate (also runs in CI)
make drift       # drift across the four monitors from recent traffic
make dbt-docs    # the dbt lineage graph and column docs
```

Only real runs (keys + `make up` + `make ingest`) produce real numbers; offline they are zero by
design. The ablation lands in [docs/eval-report.md](docs/eval-report.md) (currently the offline
placeholder until a keyed run fills it).

## The thinking

- **The plan, theme by theme:** [docs/plan/](docs/plan/) (the big picture, split into stages, each
  with a short result note).
- **Architecture end to end:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (data flow, the agentic
  loop, and the fallback chain).
- **Decisions and tradeoffs:** [docs/BUILD-PLAN.md](docs/BUILD-PLAN.md) Part A, the
  [semantic layer](docs/semantic-layer.md), and [model selection](docs/model-selection.md) (why
  Groq + Cohere, evidenced against the live Health view).
- **Showcase roadmap:** [docs/SHOWCASE-ROADMAP.md](docs/SHOWCASE-ROADMAP.md) (the staged plan and
  progress log).
- **Build log and deliberate deferrals:** [docs/DEV-NOTES.md](docs/DEV-NOTES.md).
- **Deploy:** [docs/DEPLOY.md](docs/DEPLOY.md) (Vercel, Cloud Run at min-instances 0, the hosted
  stores, the cost cap, and the keepalive job).
- **Notebooks:** [notebooks/](notebooks/) walk the data architecture and the eval step by step.

## Development

Short-lived `build/<step>` branches, one stage each, merged to `main` only when `make check` is
green and an independent review has passed. CI (`.github/workflows/ci.yml`) runs the same checks
plus the eval gate, the dbt build and tests, and a dependency audit on every change.
