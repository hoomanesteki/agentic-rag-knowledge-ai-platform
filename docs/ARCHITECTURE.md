# Architecture

How the platform works end to end, why it is shaped this way, and how it stays up when a hosted
dependency does not. Read this alongside `model-selection.md` (why each model) and
`semantic-layer.md` (governed metrics).

## The one-paragraph version

Raw structured and unstructured data is loaded through a **medallion lakehouse** (bronze → silver →
gold) with data contracts and PII masking. Gold tables and the unstructured corpus are **ingested**
into a hybrid vector store and a knowledge graph. At query time a **LangGraph agentic loop**
understands the question, retrieves and reconciles evidence from three specialists (vector,
governed metrics, graph), passes a **confidence gate**, and either answers with citations or
abstains and escalates to a human. Every turn is **traced**; traces feed **quality/health/drift**
dashboards and **MLflow**; human answers feed a **flywheel** that re-indexes them so the assistant
learns. Every hosted call has a **fallback** so the system degrades instead of failing.

## Data flow (raw → answer)

```
  Sources                Lakehouse (DuckDB + dbt)         Serving indexes
  ────────               ────────────────────────         ───────────────
  products.csv  ─┐       bronze  (typed, 1:1 raw)          Qdrant  (dense + sparse hybrid)
  sales.csv     ─┼─ETL─▶ silver  (clean, PII-masked)  ─┬─▶ Neo4j   (Product/Supplier/Store graph)
  stores.csv    ─┘       gold    (marts + contracts)   │
                                                        │   (gold feeds governed metrics;
  *.jsonl (reviews,      ingest: chunk → contextualize  │    unstructured feeds vector + graph)
  guides, orders,   ────────────  → embed → index  ─────┘
  expertise)             (leak linter keeps engine domain-agnostic)
```

- **Bronze/silver/gold** live in `dbt/models/{staging,silver,marts}`, generated from the domain
  manifest (`scripts/dbt_codegen.py`) so a new domain is config, not SQL. Tests (not-null, unique,
  relationships) and a **PII-masking test** (`dbt/tests/generic/is_masked.sql`) run on build.
- **Ingestion** (`ingest/`, `make ingest`) chunks each document, prefixes it with context (M2.4),
  embeds with the configured provider, and writes dense + sparse vectors to Qdrant plus graph nodes
  to Neo4j. Order/PII docs are tagged so retrieval can gate them.
- **Governance:** the leak linter (`scripts/check_domain_leak.py`) fails CI if brand/product
  vocabulary appears in engine folders, so the engine stays domain-agnostic and reproducible.

## The agentic loop (per turn)

Built with LangGraph (`rag/graph.py`); the supervisor and specialists live in `rag/supervisor.py`
and `rag/specialists.py`.

```
        ┌─────────────┐
  query │  understand │  route + rewrite a follow-up using history
        └──────┬──────┘
               ▼
        ┌─────────────┐   supervisor dispatches specialists in parallel:
        │   retrieve  │──▶  • Retriever  (hybrid dense+sparse + rerank)
        │  + evidence │     • Metrics    (governed SQL over gold, validated params, read-only)
        └──────┬──────┘     • Graph      (allowlisted traversals, e.g. supplier of a product)
               ▼            reconcile: a governed number beats a contradicting review
        ┌─────────────┐
        │    gate     │  confidence / grounding check (+ PII gate on order docs)
        └───┬─────┬───┘
       pass │     │ thin
            ▼     ▼
   ┌──────────┐ ┌───────────────────────────┐
   │ generate │ │ abstain → offer a human    │
   │ + cite   │ │ (escalate to review queue) │
   └────┬─────┘ └──────────┬────────────────┘
        ▼                  ▼
      answer          human specialist answers  ──▶  flywheel: verified answer
     (traced)              (rag/hitl.py)              re-indexed (rag/flywheel.py)
```

- **Bounded, not open-ended.** The loop retries a weak retrieval a fixed number of times, then
  escalates rather than spinning. Small/frequent jobs (query rewrite, metric slot-fill) use the
  cheap model; only the final synthesis uses the large model.
- **Human in the loop.** An abstain or escalation writes to the review queue (`rag/hitl.py`); an
  operator answers it in the back office; the flywheel re-embeds that answer and grows the eval set,
  so the same question is handled automatically next time. That is the learning loop.

## Resilience and fallbacks (why it does not just 500)

Every hosted dependency is wrapped so a failure degrades gracefully (`api/resilience.py`,
`pipeline/answer.py`, the web widget):

| Dependency | Primary | On failure / limit |
| --- | --- | --- |
| Embeddings (Cohere) | `ResilientEmbedder` + `CachingEmbedder` (retry + LRU) | dedup cache serves repeats; retry with backoff on 429/5xx |
| Reranker (Cohere) | `ResilientReranker` | on exhaustion, skip rerank and use the hybrid order |
| App LLM (Groq) | stream from the 70B | transient error → degraded answer event, never a dead stream |
| Metrics (DuckDB) | governed query over gold | missing lakehouse → metrics disabled, vector-only answers |
| Voice TTS (ElevenLabs) | premium voice via `/api/tts` | 204 / error → browser `speechSynthesis` (free) |
| Speech-to-text (Groq Whisper) | hosted transcription | browser Web Speech API |
| Everything, in tests/CI | deterministic fakes | no keys or network needed to develop or gate |

`is_transient()` decides retry-vs-fail; retries are capped (default 3) with backoff to bound cost.

## Evaluation, drift, and MLOps

- **Eval:** a golden fixture set + RAGAS (faithfulness, answer-relevancy, context precision/recall).
  `make gate` blocks a merge if quality drops; `make ablation` proves the reranker earns its place.
- **Drift:** `mlops/drift.py` compares recent traffic against a baseline (input, embedding, and
  answer-quality proxies, per language) so a distribution shift is visible before users feel it.
- **Tracking:** every request writes a trace; `mlops/mlflow_sink.py` logs eval runs to MLflow, and
  the back office renders quality/health/gaps for a human, so the whole thing is observable, not a
  black box.

## Scaling and domain-swap

- **Swap the domain:** drop a new `domains/<name>/` manifest and re-run ingest; no engine change.
- **Swap a model or store:** change the provider env var (adapter seam). The same code runs on
  fakes, local Docker stores, or hosted stores.
- **Scale out:** the API is stateless behind a token; the vector store, graph, and lakehouse scale
  independently; caching and the cheap/large model split keep cost flat as traffic grows.
