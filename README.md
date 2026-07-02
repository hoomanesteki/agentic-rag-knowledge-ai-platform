# Skein Lite

A local-first, domain-swappable **agentic RAG platform**. It answers questions over a mix of
structured and unstructured data, grounds every answer in citations (or abstains), routes hard
questions through a supervisor that coordinates specialist agents, and hands off to a human when
it is unsure, then learns from that answer. One engine serves any topic: a domain is a config
folder, not code.

Built step by step, MVP first, each step reviewed by an independent model before it merged.
Everything runs offline on fakes for tests and CI; the hosted models (Groq, Voyage) and the
stores (Qdrant, DuckDB, Neo4j, Postgres) are config swaps.

## What it does

- **Grounded or honest.** Hybrid retrieval (dense + sparse, RRF) with a reranker, sentence-level
  citation checks, and an abstain gate. Retrieved text is sanitized against prompt injection.
- **A brain, not a prompt.** A LangGraph state machine (understand, dispatch, reconcile) with a
  **supervisor** that dispatches to three **specialists** (a Retriever, a governed **Metrics**
  agent, and a knowledge-**Graph** agent) and reconciles their findings (a governed number beats a
  contradicting review), wrapped by a gate and a bounded agent loop that retries the hard tail
  before it escalates.
- **Governed numbers.** A read-only DuckDB metric layer answers "what is the return rate for size
  M" from a validated query, never free-form SQL, and the number is cited as its own evidence.
- **A knowledge graph.** Neo4j nodes and typed edges built from the gold lakehouse; relational
  questions ("which supplier makes X") answer from the graph.
- **Human-in-the-loop flywheel.** Escalations land in a review queue; an operator answers; the
  answer becomes a retrievable verified chunk and grows the eval set.
- **MLOps.** MLflow run tracking, a faithful RAGAS answer-quality eval, four drift monitors, and
  an offline CI eval gate that blocks a regression.
- **Voice.** Groq hosted Whisper speech-to-text with a browser Web Speech fallback, and spoken
  answers.
- **One engine, two domains.** `apparel_ecommerce` and `saas_support`, proven in CI on every
  commit so reproducibility is not a claim made at the end.

## Architecture

```mermaid
flowchart TD
  user([User: text or voice]) --> api[FastAPI: auth, rate limit, SSE]
  api -->|CHAT_BRAIN=agent| brain

  subgraph brain [LangGraph brain]
    understand[understand: rewrite + route] --> dispatch[supervisor: dispatch]
    dispatch --> reconcile[reconcile: merge, resolve conflicts] --> gate{gate}
    gate -->|auto| answer[grounded, cited answer]
    gate -->|agent| dispatch
    gate -->|escalate| queue[(review queue)]
  end

  dispatch --> retr[Retriever specialist] --> qdrant[(Qdrant: hybrid vectors)]
  dispatch --> metrics[Metrics specialist] --> duckdb[(DuckDB: governed metrics)]
  dispatch --> graph[Graph specialist] --> neo4j[(Neo4j: knowledge graph)]

  queue --> admin[/admin: claim + answer/]
  admin -->|flywheel| qdrant
  api --> traces[(traces)] --> mlflow[MLflow / RAGAS / drift / CI gate]
```

The medallion lakehouse (bronze to gold in DuckDB) and the graph are built from each domain's
manifest. Nothing under the engine names a product, metric, or brand; those live only in
`domains/<name>/`, enforced by a leak linter in `make check`.

## Quick start

Needs [uv](https://docs.astral.sh/uv/) (it manages Python 3.12) and Docker.

```bash
make setup                 # venv + locked dependencies
cp .env.example .env       # fill in GROQ_API_KEY and VOYAGE_API_KEY for real runs
make check                 # lint, tests, domain validation, leak check (fully offline)
make up                    # Qdrant, Postgres, Neo4j, MLflow in Docker
make lakehouse && make ingest && make graph-load   # build the stores for the active DOMAIN
make serve                 # API on :8000     (set CHAT_BRAIN=agent for the full brain)
cd web && npm install && npm run dev             # web chat on :3000
```

Demo login: `demo` / `skein-demo-2026`. Admin console at `/admin` (`admin` / `skein-admin-2026`).
Switch topic with `DOMAIN=saas_support`. Voice needs `TRANSCRIBE_PROVIDER=groq`.

## Evaluate

```bash
make eval        # hit@k, MRR, entity recall, abstain recall, false-abstain rate on the golden set
make ablation    # dense vs hybrid vs +rerank, per language -> docs/eval-report.md
make ragas       # faithfulness, answer relevance, context precision/recall (LLM judge)
make gate        # the offline CI eval gate (also runs in CI)
make drift       # drift across the four monitors from recent traffic
```

Only real runs (keys + `make up` + `make ingest`) produce real numbers; offline they are zero by
design. The ablation lands in [docs/eval-report.md](docs/eval-report.md) (currently the offline
placeholder until a keyed run fills it).

## The thinking

- **Decisions and tradeoffs:** [docs/BUILD-PLAN.md](docs/BUILD-PLAN.md) Part A (why DuckDB over
  Spark, Groq + Voyage, the supervisor over specialists, and the risks with their fixes).
- **Build log and deliberate deferrals:** [docs/DEV-NOTES.md](docs/DEV-NOTES.md).
- **Deploy:** [docs/DEPLOY.md](docs/DEPLOY.md) (Vercel, Cloud Run at min-instances 0, the hosted
  stores, the cost cap, and the keepalive job that stops free tiers idling out).
- **Demo:** a 3-minute recorded walkthrough (add the link once recorded).

## Development

Short-lived `build/<step>` branches, one milestone step each, merged to `main` only when
`make check` is green and an independent review has passed. CI (`.github/workflows/ci.yml`) runs
the same gate plus the eval gate and a dependency audit on every PR. The `ship`, `review`, and
`preflight` skills in `.claude/skills/` encode the loop.
