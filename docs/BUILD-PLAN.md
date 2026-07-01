# Skein Lite build plan

This is the working plan for building Skein Lite. It rewrites the phase order in the
original spec so you get a working, demoable thing early, then thicken each layer. It is
built for someone on Claude Pro, so every step is sized to fit one focused session and to
keep token use low.

Read this file at the start of each session, do one step, commit, then stop. You do not
need to hold the whole project in your head or in the chat.

---

## Part A. What I checked in the spec, and what to change first

The spec is strong. The design principles (local first, config driven domains, grounded
or abstain, everything traced) are the right ones. Below are the things that will bite
during the build if we do not settle them now. Each has a one line fix.

### Decisions to lock before writing code

| Topic | Problem in the spec | Decision for this build |
|---|---|---|
| Data engine | The spec lists PySpark local mode AND DuckDB+Parquet, and puts both `pyspark_jobs/` and `dbt-duckdb` doing bronze to gold. That is two engines doing the same job, and PySpark needs a JVM (not installed here). | Use the light path: DuckDB + dbt-duckdb owns bronze to gold. No Spark, no JVM, no Delta for the MVP. Keep a PySpark/Delta adapter as a later "enterprise parity" swap, not part of the core. |
| Table format | Delta via delta-rs vs Delta via Spark are different tools and get mixed. | Parquet files in DuckDB for the MVP. Revisit Delta only if we do the Databricks demo. |
| First demoable output | Spec builds the full lakehouse (P1) and knowledge graph (P2) before any answer comes out (P3). That is a lot of infra before the first "wow". | Ship a thin vertical slice first: ingest a few docs, answer one grounded cited question. Add the lakehouse, graph, and metric layer after the thread works end to end. |
| App LLM | Needs to be picked. Claude Pro does not give API access for the app itself. | App uses Groq (hosted, OpenAI compatible, fast). Small tasks use `llama-3.1-8b-instant`, synthesis uses `llama-3.3-70b-versatile`. Claude Code (this tool) is separate and only helps you build. |
| Embeddings and rerank | Groq does not serve embeddings or rerankers, and the laptop (MacBook Air, limited disk) should not run local models. | Use hosted Voyage for both: `voyage-3-large` embeddings and `rerank-2` reranker. One account, strong multilingual quality, free tier. Swap to Cohere or OpenAI later via the adapter. Quality first, cost second. |

### Risks to keep in view (with the fix)

Blocking:
- Domain agnostic contract leaks in the data layer. Bronze to gold transforms, dbt models,
  metrics, and the golden set question templates are all schema specific. They cannot live
  in "the engine that never changes". Fix: add a `sources` manifest to `domain.yaml` that
  declares each seed file, its role, primary key, columns, and PII columns. The engine reads
  that manifest and builds bronze generically. Metrics and graph load reference roles from
  the manifest, not hardcoded table names. This is what the `domain-pack` skill enforces.

High:
- Laptop memory and disk. All models are hosted (Groq for the LLM, Voyage for embeddings and
  rerank), so nothing heavy runs on the MacBook Air. Locally you only run Qdrant and Postgres
  in Docker plus the dev servers. Still do not run Spark as a service, and start Neo4j only
  from milestone M5, not before.
- Confidence gate depends on a weak local judge. A small local model doing faithfulness
  checks gives noisy confidence, which makes the gate misfire. Fix: keep the gate simple at
  first (retrieval score plus a citation coverage check), add the model judge later, and
  tune thresholds against the golden set.
- CI eval gate cannot run local 14B models on free GitHub runners (no GPU, low RAM, time
  caps). Fix: the CI gate runs on recorded fixtures and a tiny judge, or nightly on your own
  machine. Do not assume the full stack runs in CI.
- Free tier idle windows are shorter than the spec says. Neo4j Aura Free pauses after a few
  days idle and is deleted around 30 days. Supabase free pauses after about a week idle.
  Fix: one keepalive job that pings both, and a one command reseed if either is wiped.
  Verify current terms when you set up hosting, since these change.

Medium:
- Fusing metric results with text chunks by RRF does not really make sense (one number vs
  fifty passages). Fix: metric answers skip fusion and go straight into the prompt as a
  separate, labeled evidence block. RRF is only for vector plus graph text hits.
- Text to Cypher over user content is an injection risk. Fix: templated Cypher first, read
  only transactions, and an allowlist of clauses. Free text to Cypher only behind that guard.
- Voice via the browser Web Speech API is mostly Chrome and Edge. Fix: fine for the demo,
  just show a clear note and a text fallback.
- "Hard billing cap" on GCP is not truly hard (budgets are alerts). Fix: cap with Cloud Run
  max-instances and request quotas, rely on the provider free tier 429s, add a kill switch.

Keep as is: the medallion idea, dbt for tests and lineage, the KG for relations and for
generating the eval set, hybrid vector plus rerank, the LangGraph brain, MLflow plus RAGAS,
HITL feeding a flywheel, and the adapters boundary. Those are the senior parts. We are only
reordering when they get built and tightening the domain contract.

---

## Part B. The build, step by step

Milestones go M0 to M9. Each milestone is demoable on its own. Each step is one session.
Check the box when the "Done when" line is true and you have committed.

Legend: size S = short session, M = medium, L = long or split into two.

### M0. Foundations (no answer yet, just the skeleton)

- [ ] M0.1 Repo layout and env. Create the lean folder tree (see Part C), `.env.example`,
  `Makefile` with `seed`, `run`, `test` targets, and a Python project (`pyproject.toml` or
  `requirements.txt`) with pinned versions. Done when: `make` targets exist and a fresh
  `pip install` works. Size S.
- [ ] M0.2 Domain pack contract and tiny seed. Use the `/domain-pack` skill to scaffold
  `domains/lululemon/` with a small hand written seed (about 8 products, 20 reviews, a few
  stores and suppliers). Done when: `python .claude/skills/domain-pack/scripts/validate_domain_pack.py domains/lululemon` passes. Size M.
- [ ] M0.3 Compose for what M1 needs only. Docker Compose with just `qdrant` and `postgres`.
  Add `neo4j`, `mlflow` later when their milestone arrives. Done when: `docker compose up`
  is healthy for those two. Size S.

### M1. First answer (the MVP walking skeleton)

This is the milestone that proves the idea. Keep it linear and simple. No LangGraph yet.

- [ ] M1.1 Adapters seams. Define thin interfaces in `adapters/` for embeddings, vector
  store, and LLM, with one local impl each. Business code imports only these. Done when: a
  fake impl can be swapped in a test. Size M.
- [ ] M1.2 Ingest unstructured. Chunk the reviews/docs from the domain pack, embed with the
  hosted embedder (Voyage), index in Qdrant with metadata (source, doc_type, lang, date).
  Done when: a script indexes the seed and a manual query returns relevant chunks. Size M.
- [ ] M1.3 Retrieve and answer. One function: embed the query, search Qdrant, build a
  grounded prompt, call the app LLM (Groq), return an answer with inline
  citations. If nothing relevant is found, abstain. Done when: asking a seed question in the
  CLI returns a cited answer, and a nonsense question abstains. Size M. This is the demo.

### M2. Make the answer trustworthy

- [ ] M2.1 Reranker. Add the hosted reranker (Voyage rerank-2) over the top ~50 hits down to
  ~8. Done when: reranked order is visibly better on 3 sample questions. Size S.
- [ ] M2.2 Grounding check and confidence. After generate, check each sentence has a
  citation and the cited chunk supports it (start with overlap, not a model). Produce a
  confidence score and abstain below a threshold. Done when: an unsupported claim is caught
  and the answer abstains. Size M.
- [ ] M2.3 Chunking upgrade. Token and structure aware chunks with a short doc summary
  prefix. Done when: retrieval quality on samples improves and chunks carry clean metadata.
  Size S.

### M3. Customer chat UI

- [ ] M3.1 API. FastAPI `POST /api/chat` that streams tokens (SSE) and returns citations,
  tier, confidence. Auth stubbed for now. Done when: curl streams a cited answer. Size M.
- [ ] M3.2 Web chat. Minimal Next.js page: input, streaming answer, citation chips, thumbs.
  Calm layout, one accent color. Done when: you can ask and read a streamed cited answer in
  the browser. Size L.
- [ ] M3.3 Real auth. JWT users in Postgres, seeded demo login, Turnstile on the login form.
  Done when: login gates the chat and demo creds are in the README. Size M.

### M4. Structured side, the light way

- [ ] M4.1 Bronze to gold in DuckDB. Generic bronze load driven by the `sources` manifest,
  then dbt-duckdb models for silver (clean, mask PII) and gold (curated), with dbt tests.
  Done when: gold tables build and `dbt test` passes. Size L.
- [ ] M4.2 One or two metrics. Define `metrics.yaml` metrics (for example return rate by
  size), a slot filling resolver that validates params and runs read only, never free SQL.
  Done when: a metric call returns a correct governed number. Size M.
- [ ] M4.3 Metric retriever. Route obvious number questions to the metric resolver and put
  the result in the prompt as a labeled evidence block (not fused with text). Done when:
  "what is the return rate for size M" answers from the metric layer with the number cited.
  Size M.

### M5. Knowledge graph

- [ ] M5.1 Add Neo4j and load gold. Start the `neo4j` service. Load nodes and typed edges
  from gold, driven by the ontology and the manifest. Done when: a templated Cypher query
  returns expected relations. Size M.
- [ ] M5.2 Entity linking. LLM pass at build time links review mentions to canonical
  entity_ids, above a confidence threshold, low confidence goes to a review list. Done when:
  reviews carry `mentions` edges to products. Size M.
- [ ] M5.3 Graph retriever and expand. Vector first then hop from mentioned entity_ids,
  plus a graph first path for relational questions. Done when: "which suppliers feed the
  most returned products" answers using graph plus text. Size M.

### M6. The brain (LangGraph)

Now fold M1 to M5 into the real state machine. Before this it was a linear pipeline. This
milestone makes it agentic.

- [ ] M6.1 State and linear graph. Typed state, nodes understand, route, retrieve, fuse,
  rerank, generate, verify, gate. Wire the existing pieces in. Done when: all four query
  types answer through the graph. Size L.
- [ ] M6.2 Agent loop and gate. Bounded ReAct over the same retriever tools, step and
  budget caps, confidence gate decides auto vs agent vs escalate. Done when: a hard question
  triggers the loop and a still unanswerable one escalates. Size M.
- [ ] M6.3 HITL. Postgres review queue, LangGraph checkpointer so interrupts survive, an
  honest "escalated" reply to the user, admin can answer and that answer is stored as gold.
  Done when: a low confidence question lands in the queue and a human answer closes it. Size L.

### M7. Back-office

- [ ] M7.1 Queue UI and answering. `/admin` queue list, row lock on claim, answer form.
  Done when: an operator can claim and answer a queued item. Size M.
- [ ] M7.2 Quality dashboard. Thumbs, faithfulness, escalation rate, grouped by language.
  Done when: a thumbs down from M3 shows up here per language. Size M.
- [ ] M7.3 Read only views. Knowledge gap list, ontology, metric, and dbt lineage viewers,
  and a link to MLflow. Done when: each view renders from real data. Size M.

### M8. MLOps

- [ ] M8.1 MLflow tracing. Trace every request (nodes, latency, retrieval hits, confidence).
  Thread this into the graph nodes now that they are stable. Done when: a run shows a full
  trace. Size M.
- [ ] M8.2 KG golden set and RAGAS. Generate Q and A by walking graph paths, with the
  question templates living in the domain pack (not the engine). Run RAGAS plus retrieval
  metrics. Done when: an eval report is produced in at least two languages. Size L.
- [ ] M8.3 Drift and CI gate. The four drift monitors stratified by language, and a CI eval
  gate on fixtures that blocks a deliberate regression. Done when: the gate fails on a seeded
  regression and passes otherwise. Size L.

### M9. Voice, polish, deploy, second domain

- [ ] M9.1 Voice. Browser Web Speech in and out, with a text fallback and a clear support
  note. Done when: a full voice round trip works in Chrome. Size M.
- [ ] M9.2 Polish. Motion, loading and cold start states, empty states, error and fallback
  paths from the edge case list. Done when: the demo feels smooth. Size M.
- [ ] M9.3 Deploy. Web to Vercel, API to Cloud Run (min-instances 0), Neo4j Aura, Qdrant
  Cloud, Supabase, Gemini, plus the keepalive job. Done when: the public demo login works
  inside the cost cap. Size L.
- [ ] M9.4 Second domain. Add `domains/saas_support/` with the skill, reseed, switch the
  `DOMAIN` env var. Done when: the same engine answers support questions with no engine code
  change. This proves reproducibility. Size L.

---

## Part C. Lean file structure

Keep the tree flat and obvious. Add folders only when a milestone needs them.

```
skein-lite/
  README.md  docker-compose.yml  .env.example  Makefile  pyproject.toml
  domains/
    lululemon/            # see the domain-pack skill for the contract
    saas_support/         # added at M9.4
  adapters/               # embeddings, vectorstore, llm, (later) graph, storage
  ingest/                 # chunk, embed, index  (M1, M2)
  retrieval/              # vector, metric, graph, fuse, rerank
  pipeline/               # linear pipeline first (M1-M5), becomes the graph at M6
  rag/                    # LangGraph app, state, nodes  (M6+)
  data/                   # duckdb + dbt project  (M4)
  api/                    # FastAPI  (M3+)
  web/                    # Next.js  (M3+)
  mlops/                  # mlflow config, eval, drift  (M8)
  scripts/                # seed.py, reset.py, keepalive
  docs/                   # this plan and notes
```

Rule that keeps it domain agnostic: nothing under `adapters/`, `retrieval/`, `pipeline/`,
`rag/`, `ingest/` may name a product, a metric, or an ontology label. Those names only exist
in `domains/<name>/`. If you catch a domain word in engine code, move it to the domain pack.

---

## Part D. Working on Claude Pro without burning tokens

- One step per session. Start by reading this file plus only the files that step touches.
  Do not ask Claude to re-read the whole repo.
- Keep state in the repo, not in chat. After each step, commit. When context gets long,
  you can safely reset the conversation because the plan and code hold the state.
- Use the `/domain-pack` skill to generate and check packs instead of reasoning it out each
  time. Deterministic scaffolding is cheaper than fresh generation.
- Let the app do its own model calls (Groq). That work does not touch your Claude quota.
  Claude is for building, not for serving answers.
- Write each "Done when" as a command you can run. A runnable check ends the debate about
  whether a step is finished, which saves back and forth.
- Prefer small diffs. If a step feels like an L, split it into two S sessions.
```
