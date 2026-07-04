# Skein Lite

[![CI](https://github.com/hoomanesteki/agentic-rag-knowledge-ai-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/hoomanesteki/agentic-rag-knowledge-ai-platform/actions/workflows/ci.yml)

**[Read the showcase →](https://hoomanesteki.github.io/agentic-rag-knowledge-ai-platform/)**:
architecture, evaluation, the decisions behind it, and an honest look at the failure modes. The
site's source is in [`showcase/`](showcase/) (Quarto, rendered to GitHub Pages by CI).

A local-first, domain-swappable **agentic RAG platform**. It answers questions over a mix of
structured and unstructured data, grounds every answer in citations (or abstains), routes hard
questions through a supervisor that coordinates specialist agents, and hands off to a human when
it is unsure, then learns from that answer. One engine serves any topic: a domain is a config
folder, not code.

Built step by step, MVP first, each step reviewed by an independent model before it merged. Tests
and CI run entirely offline on fakes; the hosted models (Groq, Cohere) and the stores (Qdrant,
DuckDB, Neo4j) are config swaps.

## What it does

- **Grounded or honest.** Hybrid retrieval (dense + sparse, RRF) with a reranker, sentence-level
  citation checks, and an abstain gate. Retrieved text is sanitized against prompt injection.
- **A brain, not a prompt.** A LangGraph state machine (understand, dispatch, reconcile) with a
  **supervisor** that dispatches to three **specialists** (a Retriever, a governed **Metrics**
  agent, and a knowledge-**Graph** agent) and reconciles their findings (a governed number beats a
  contradicting review), wrapped by a gate and a bounded agent loop that retries before it
  escalates.
- **Guards that do not trust the model.** Order PII, prompt injection, customer enumeration, and
  gender-correct recommendations are enforced deterministically in code, before the prompt. See
  [the guardrails](#guardrails-enforced-in-code-not-in-the-prompt).
- **A real data stack.** The medallion (bronze, silver, gold) is modeled in **dbt** with schema,
  relationship, and PII-masking tests, generated from each domain's manifest. A **semantic layer**
  (`metrics.yaml`) is the single source of truth the agent, the eval, and the dashboards all read.
- **Governed numbers.** A read-only DuckDB metric layer answers "what is the return rate for size
  M" from a validated single-SELECT query, never free-form SQL, and the number is cited as its own
  evidence.
- **A knowledge graph.** Neo4j nodes and typed edges built from gold; relational questions ("which
  supplier makes X") answer from the graph via allowlisted traversals, not free Cypher.
- **Human-in-the-loop flywheel.** Escalations land in a review queue; an operator answers; the
  answer becomes a retrievable verified chunk and grows the eval set.
- **Observability.** Langfuse traces every turn and LLM call (model, tokens, latency, cost);
  MLflow, a faithful RAGAS eval, four drift monitors, and an offline CI eval gate cover the rest.
- **Guided and voiced.** Per-domain starter prompts, spoken input (Groq Whisper), spoken replies
  (browser voice by default, ElevenLabs when keyed), a storefront-style demo UI with the chat
  widget, and a backoffice dashboard at `/admin`.
- **One engine, two domains.** `apparel_ecommerce` and `saas_support`, proven in CI on every
  commit.

## The stack

| Layer | Choice | Where |
| --- | --- | --- |
| API | FastAPI: SSE chat, JWT auth, rate limiting, Turnstile, degraded mode | `api/` |
| Brain | LangGraph supervisor + three specialists, gate + bounded retry loop | `rag/` |
| Retrieval | Qdrant hybrid (dense + sparse, server-side RRF), Cohere `embed-v4.0` + `rerank-v3.5` | `adapters/`, `retrieval/` |
| Generation | Groq Llama 3.3 70B (large) and Llama 3.1 8B (small) | `adapters/groq.py` |
| Voice | Groq Whisper in; browser voice or ElevenLabs Flash v2.5 out (key stays server-side) | `adapters/groq_whisper.py`, `adapters/elevenlabs.py` |
| Analytics | DuckDB + dbt medallion, `metrics.yaml` semantic layer | `dbt/`, `data/` |
| Graph | Neo4j, loaded from gold | `knowledge/`, `adapters/neo4j_store.py` |
| MLOps | Langfuse tracing, MLflow (compose server backed by Postgres, or `./mlruns`), RAGAS eval, drift monitors, CI gate | `mlops/`, `evaluation/` |
| Web | Next.js 14 storefront demo with the chat widget and `/admin` | `web/` |

Every provider sits behind an adapter interface, and the defaults are offline fakes
(`adapters/fakes.py`), so a fresh clone verifies end to end with no keys.

## The system at a glance

```text
     Browser (storefront chat + /admin)        Voice
          |  text                                |  audio
          v                                      v
   +-----------------------------------------------------+
   |                      FastAPI                        |  JWT auth, rate limit,
   |  /api/chat (SSE)   /api/transcribe   /api/tts       |  Turnstile, degraded mode
   |  /api/admin/*                                       |
   +--------------------------+--------------------------+
                              |  CHAT_BRAIN=agent
                              v
   +-----------------------------------------------------+
   |           The brain (LangGraph + the gate)          |
   |   understand  ->  dispatch  ->  reconcile  -> gate  |
   |   (rewrite,       (supervisor    (merge,     (auto /|
   |    route)          fans out)      rank,     retry / |
   |                        |          resolve) escalate)|
   |           +------------+------------+               |
   |           v            v            v               |
   |      Retriever      Metrics       Graph             |  three specialists,
   |      specialist     specialist    specialist        |  one Finding contract
   +-----------|------------|------------|---------------+
               v            v            v
         +----------+ +-----------+ +-----------+
         |  Qdrant  | |  DuckDB   | |   Neo4j   |
         |  hybrid  | | governed  | | knowledge |
         |  vectors | | metrics   | |   graph   |
         +----------+ +-----------+ +-----------+
               ^
               |  verified answers grow the index (the flywheel)
         +---------------------------------+
         |  review queue  ->  /admin       |  human in the loop
         +---------------------------------+
```

The LangGraph graph itself is understand -> dispatch -> reconcile (`rag/supervisor.py`); the gate
and the bounded loop wrap it (`rag/agent.py`): answer when confident and conflict-free, retry with
a reformulated query up to a step cap when the question looked answerable, escalate to the review
queue otherwise.

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

A leak linter in `make check` greps every engine folder for each pack's brand, product, metric,
and glossary vocabulary and fails the build on a hit, which is what keeps the engine reusable
across domains. The apparel pack's content is written for realism: 304 human-voice reviews (one
positive and one honest-critical per product), 152 product descriptions grounded in fabric, fit,
and care detail, and synthetic orders whose fake PII exercises the order gate below. See
[docs/semantic-layer.md](docs/semantic-layer.md).

## How one turn works

```text
   question
      |
      v
   understand ------ expand a short follow-up with the prior turns, repair catalog
      |              typos, pick a route (factual / relational / qualitative / metric)
      v
   retrieve -------- dense + sparse, fused with RRF, then reranked
      |
      v
   ground ---------- sentence-level citation check; retrieved text is sanitized
      |              and framed as data, never as instructions
      +-- confident? -- yes --> answer with [1][2] citations
      |                  no
      +-- in scope? ---- no --> "I do not have enough information" (abstain)
                         yes
                          +--> the agent loop retries the hard tail, then escalates
```

## Guardrails enforced in code, not in the prompt

The riskiest behaviours are deterministic, applied before the model sees anything, so they hold no
matter what the model would say (all in `pipeline/answer.py`, exercised by tests):

- **Order PII needs name + email.** Order documents only enter retrieval for a first-person
  account question, and each one must then pass an identity check against the shopper's own words:
  both the account email and the holder's name, where a name token derivable from the email does
  not count as a second factor. An unverified order document is dropped before it reaches the
  prompt, so an email-only turn cannot leak a name, an order number, or a tracking link.
  Third-party lookups ("orders placed by x@y") never qualify.
- **Prompt injection is refused, not resisted.** Requests to reveal or override the system prompt
  get a deterministic refusal before retrieval; retrieved text is sanitized and every prompt frames
  context as untrusted data.
- **Customer enumeration is refused.** "Who bought X" and "list your customers" are declined
  before retrieval, so no reviewer or account-holder name can surface.
- **A stated gender is a hard constraint.** Opposite-gender SKUs are filtered out of the retrieval
  hits, and opposite-gender picks are redacted clause by clause from guides and reviews before the
  model sees them. Product gender is read from the domain manifest, never hardcoded in the engine.
- **Harmful requests are declined**, with the pattern scoped so ordinary shopping phrasing (an
  "explosive sprint") never trips it.

## Quick start

Needs [uv](https://docs.astral.sh/uv/) (it manages Python 3.12) and Docker.

```bash
make setup                 # venv + locked dependencies
cp .env.example .env       # fill in GROQ_API_KEY and COHERE_API_KEY for real runs
make check                 # lint, tests, domain validation, leak check, eval gate (fully offline)
make doctor                # if a step hangs or fails, this says why (Docker, .env, keys)
make up                    # Qdrant, Postgres, Neo4j, MLflow in Docker (preflighted)
make dbt-build             # build + test the semantic layer (medallion + governance tests)
make ingest && make graph-load                   # build the vector index and the graph
make serve                 # API on :8000     (set CHAT_BRAIN=agent for the full brain)
cd web && npm install && npm run dev             # web chat on :3000
```

Demo login: `demo` / `Canada54321`. Admin console at `/admin` (`admin` / `skein-admin-2026`).
Switch topic with `DOMAIN=saas_support` (the starter prompts switch with it). Voice input needs
`TRANSCRIBE_PROVIDER=groq`; spoken replies use the browser voice by default, or ElevenLabs with
`TTS_PROVIDER=elevenlabs` and a key. Tracing needs the `LANGFUSE_*` keys. `make reproduce` runs
the whole offline verification in one command.

## Evaluate

```bash
make eval        # hit@k, MRR, entity recall, abstain recall, false-abstain rate on the golden set
make ablation    # dense vs hybrid vs +rerank, per language -> docs/eval-report.md
make ragas       # faithfulness, answer relevance, context precision/recall (LLM judge)
make gate        # the offline CI eval gate (also runs in CI)
make drift       # drift across the four monitors from recent traffic, per language
make promote     # gate the config through MLflow stages (dev -> staging -> prod) by eval score
make dbt-docs    # the dbt lineage graph and column docs
```

Only real runs (keys + `make up` + `make ingest`) produce real numbers; offline they are zero by
design. The ablation lands in [docs/eval-report.md](docs/eval-report.md) (currently the offline
placeholder until a keyed run fills it).

Two of these run fully offline on recorded fixtures and produce real signal: the CI gate blocks a
regression, and the drift monitor flags a distribution shift. The walkthrough is in
[notebooks/02-evaluation.ipynb](notebooks/02-evaluation.ipynb):

| The gate blocks a regression | Drift catches a shift |
| --- | --- |
| ![eval gate blocks a regression](docs/img/eval-gate.png) | ![drift PSI](docs/img/drift-psi.png) |

## Observability and CI

```text
   every turn  --> Langfuse trace: model, tokens, latency, cost
               \-> request trace --> /admin dashboard: quality, health, gaps
                                \--> MLflow runs
                                \--> four drift monitors (per language)
                                \--> RAGAS answer-quality eval

   every commit --> CI: make check (lint, tests, domain validation, leak check, eval gate)
                        + dbt build and tests for both domains + a gold parity test
                        + web lint and build + dependency audits
```

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
plus the eval gate, the dbt build and tests for both domains, the web build, and dependency audits
on every change.
