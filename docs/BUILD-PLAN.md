# Skein Lite build plan

This is the working plan. It rewrites the phase order in the original spec so you get a
working, cited answer early, then thicken each layer. The guiding rule after this revision
is simple: **measure everything**. From M2 on, no step ships on "looks better", it ships on a
number moving on a golden set. That single discipline is what separates a portfolio project
from a demo.

Read this file at the start of each session, do one step, run `make check`, commit, then
stop. Pacing and cost notes live in [DEV-NOTES.md](DEV-NOTES.md), out of the way.

---

## Part A. What I checked in the spec, and what to change first

The spec is strong. The design principles (local first, config driven domains, grounded or
abstain, everything traced) are the right ones. The decisions below settle the things that
would otherwise bite during the build. Several come from a second senior review that pushed
the plan from "good intentions" to "measured".

### Decisions locked before writing code

| Topic | Problem in the spec | Decision |
|---|---|---|
| Data engine | It lists PySpark AND DuckDB both doing bronze to gold, and PySpark needs a JVM (not installed). | DuckDB + dbt-duckdb owns bronze to gold. No Spark, no JVM. PySpark/Delta is a later enterprise-parity adapter, not core. |
| App LLM | Needs picking. Claude Pro gives no API access for the app itself. | Groq (hosted, OpenAI compatible, fast). Model IDs are config in `.env`, not hardcoded, and get benchmarked once on the golden set. Start point: a fast small model for understand/route/verify, a strong large model for synthesis. Pick the best current Groq models at M1, do not assume 2024 ones. |
| Embeddings and rerank | Groq serves neither, and the laptop (limited disk) should not run local models. | Hosted Voyage: `voyage-3-large` embeddings, current rerank model (check the latest, it moves). One account, strong multilingual. Verify free-tier terms at M1 (Voyage is now MongoDB owned). Swappable to Cohere via the adapter. |
| Toolchain | A reproducibility project cannot ship a 3.9 venv against a 3.11 CI with unpinned deps. | uv, Python 3.12 (pinned in `.python-version`), dependencies locked in `uv.lock`. Done at M0. |
| First output | The spec builds the lakehouse and graph before any answer. | Thin vertical slice first: ingest a few docs, answer one grounded cited question (M1), then add structured, graph, and the agentic brain. |
| Measurement | Every quality claim in the spec is unmeasured. | A hand-written golden set lives in the domain pack from M0.2. `make eval` exists by M2. Every step after that reports a metric delta. |
| Reproducibility | It is the top priority but the spec only proves it at the very end. | Prove it continuously: a domain-leak linter in `make check` from M0, a second-domain stub wired into CI from M4. |
| Local-first honesty | README says local-first, but all models are hosted. | Keep infra local (Qdrant, Postgres, Neo4j in Docker). Add one optional local model adapter (Ollama / ONNX) behind the M1 seam so the claim is real and the adapter boundary is demonstrated. Hosted stays the default. |

### Risks to keep in view (with the fix)

Blocking:
- Domain-agnostic contract leaks in the data layer. Transforms, metrics, and eval templates
  are schema specific. Fix: the `sources` manifest in `domain.yaml` declares each seed file,
  its role, keys, columns, and PII columns. The engine reads the manifest, never hardcodes
  names. The leak linter (`make check`) enforces it on every commit.
- Hybrid search is required but easy to skip. Fix: M1.2 builds dense AND sparse in Qdrant
  with RRF, not dense-only. The ablation at M2 proves what hybrid and rerank each buy.

High:
- Confidence gate on lexical overlap misfires on paraphrase. Fix: ship the simple gate first,
  but tune its threshold against the golden set (M2) and the thumbs data (M7), and add a
  model judge later. Nothing to tune against until the golden set exists, another reason it
  comes at M0.2.
- CI cannot run hosted models or 14B locals. Fix: the eval gate runs on a small recorded
  fixture set with a cheap judge, checked into `rag/eval/fixtures/`. Design it at M8, do not
  assume the full stack runs in CI.
- Free-tier decay. Neo4j Aura Free pauses after a few days idle and is deleted around 30 days;
  Supabase free pauses after about a week. Fix: one keepalive job pings both, plus a one
  command reseed. Provision hosted services late (M9) so nothing rots between milestones.
- Public demo abuse. Real Groq/Voyage keys behind a public URL drain in an afternoon. Fix:
  read-only + rate limit + cached demo answers + a hard request cap, wired at M3 and enforced
  at deploy.

Medium:
- Indirect prompt injection. Retrieved reviews are user content; one saying "ignore previous
  instructions" flows into the prompt. Fix: delimit and sandbox retrieved text, strip
  instruction-like spans, and keep one adversarial seed doc to prove the defense (M2).
- Metric layer must be genuinely read-only. Fix: run metric SQL on a DuckDB read-only
  connection and reject any non-SELECT statement (M4), not just trust the template.
- Fusing a metric number with text by RRF is nonsense. Fix: metric results are a separate
  labeled evidence block in the prompt. RRF is only for vector plus graph text hits.
- Voice via Web Speech is Chrome-centric. Fix: use Groq hosted Whisper for speech-to-text
  with Web Speech as fallback (M9).
- Multi-agent orchestration can be pure cost and latency theater. Fanning out to specialists
  and a consensus round multiplies LLM calls, and "agents that talk" can loop. Fix: the
  supervisor fans out only when a query needs more than one evidence type (a single-source
  question stays single-pass), every specialist and the consensus round is bounded by step and
  token caps, and M6.3 must show consensus beats single-pass on the golden set plus a planted
  set of conflict cases. If it does not beat single-pass, it does not ship: it stays a demoable
  path behind a flag, not the default. Measured, not assumed.

Keep as is: the medallion idea, dbt for tests and lineage, the KG for relations and for
generating eval questions, hybrid vector plus rerank, the LangGraph brain (now a supervisor
over specialist agents, see M6), MLflow plus RAGAS, HITL feeding a flywheel, and the adapters
boundary. Those are the senior parts.

### The agent architecture: a supervisor over specialists that must agree (M6)

The brain is not one prompt with tools. It is a **supervisor agent** (the orchestrator) that
decomposes a question, dispatches it to the **specialist agents** that each own one evidence
source (a Retriever over hybrid text + graph, a Metrics agent over the governed numbers, a
Graph agent over relations), and then runs a **consensus step**: the specialists' structured
findings are checked for agreement, conflicts are reconciled (a metric number that contradicts
a text claim is caught, not averaged), and only agreed, grounded evidence is synthesized into
the answer. If they cannot agree, the system abstains or escalates rather than papering over
the disagreement. This is the multi-agent layer, built on the same tools from M1 to M5, so no
earlier code is rewritten: the M1 to M5 pipeline functions become the specialists' tools.

This earns its cost only if it is bounded and measured (see the risk note below).

---

## Part B. The build, step by step

Milestones M0 to M9. Each is demoable. Each step is one session with a runnable "Done when".
Size S = short, M = medium, L = long or split in two.

### M0. Foundations

- [x] M0.1 Toolchain and scaffold. uv + Python 3.12 + `uv.lock`, Makefile, `.env.example`,
  smoke tests. Done when: `make check` is green in a fresh clone.
- [ ] M0.2 Domain pack and golden set. Use `/domain-pack` to scaffold
  `domains/apparel_ecommerce/` (a fictional brand, synthetic data) plus a hand-written
  `eval/golden.jsonl` (about 20 questions: answerable, unanswerable, out-of-domain, in two
  languages). Done when: `make validate DOMAIN=apparel_ecommerce` passes and the golden set
  is present and well formed.
- [ ] M0.3 Compose for M1 only. Docker Compose with `qdrant` and `postgres`. Done when: both
  are healthy.

### M1. First answer (the MVP walking skeleton)

Keep it linear, no LangGraph yet.

- [ ] M1.1 Adapter seams. Thin interfaces for LLM, embeddings, vector store, with the hosted
  impls (Groq, Voyage) as default and one optional local impl behind the same interface.
  Done when: a fake impl swaps in a test, and the local impl answers one query offline.
  Size M.
- [ ] M1.2 Ingest, hybrid. Chunk the pack's text, embed dense AND sparse (Voyage dense +
  a sparse/lexical vector) into Qdrant, collection named with the embedding model and
  version, content-hash so re-ingest is idempotent. Done when: a hybrid query returns
  relevant chunks with metadata, and re-running ingest changes nothing. Size M.
- [ ] M1.3 Answer, with tracing from day one. Retrieve (dense + sparse, RRF), build a
  grounded prompt, call Groq, return an answer with inline citations, abstain when nothing is
  relevant. Emit a per-request trace as JSON (query, retrieved ids and scores, prompt hash,
  model, tokens in and out, latency, estimated cost, confidence). Done when: a seed question
  returns a cited answer, a nonsense one abstains, and a trace file is written per request.
  Size M. This is the demo.

### M2. Make it trustworthy, and measured

- [ ] M2.1 Eval harness. `make eval` runs the golden set and reports recall@k, MRR, and
  abstain precision (did it correctly abstain on the unanswerable and out-of-domain ones).
  Done when: `make eval` prints a scorecard for the current retriever. Size M.
- [ ] M2.2 Rerank, measured. Add the Voyage reranker (top ~50 to top ~8). Done when:
  recall@8 on the golden set improves versus M2.1 and the delta is recorded. Size S.
- [ ] M2.3 Grounding, confidence, injection defense. Sentence-level citation check and a
  confidence score with an abstain threshold; sandbox retrieved text and strip instruction
  spans. Done when: abstain precision holds or improves on the golden set, and the adversarial
  seed doc does not change the answer. Size M.
- [ ] M2.4 Chunking, measured. Token and structure aware chunks with a short doc-summary
  prefix. Done when: recall on the golden set improves and the delta is recorded. Size S.
- [ ] M2.5 Ablation. Record dense vs hybrid vs +rerank vs +chunking, per language, into a
  table. Done when: `docs/eval-report.md` holds the ablation with real numbers. Size S.

### M3. Customer chat UI

- [ ] M3.1 API, with resilience. FastAPI `POST /api/chat` streaming tokens (SSE), returning
  citations, tier, confidence. Retry/backoff and timeouts on the hosted calls, a degraded mode
  on a Groq 429 (smaller model or honest abstain), and per-IP rate limiting. Done when: curl
  streams a cited answer and a forced 429 degrades instead of erroring. Size M.
- [ ] M3.2 Web chat. Minimal Next.js page: input, streaming answer, citation chips, thumbs.
  Calm layout, one accent color. Done when: you can ask and read a streamed cited answer in
  the browser. Size L.
- [x] M3.3 Real auth. JWT sessions, hashed passwords, seeded demo login, Turnstile on login,
  protected endpoints. Users live in SQLite for now; swap to Postgres at M9.3 (same schema).
  Done: login gates chat and feedback, demo creds in the README.

### M4. Structured side, and continuous reproducibility

- [x] M4.1 Bronze to gold in DuckDB. A manifest-driven medallion (bronze raw, silver typed +
  PII-masked, gold curated) built with plain DuckDB, plus manifest-driven data contracts
  (primary key non-null/unique, declared columns present). dbt-duckdb was dropped on purpose:
  dbt models are per-domain SQL, which fights the one-engine-many-domains thesis; the contracts
  cover the dbt generic tests that matter here. dbt can overlay as a lineage/docs artifact
  later. Done: gold builds for two domains and contracts pass. Size L.
- [ ] M4.2 Metrics, read-only. Metrics from `metrics.yaml`, a slot-filling resolver that
  validates params and runs on a DuckDB read-only connection rejecting any non-SELECT. Done
  when: a metric call returns a correct governed number and a write attempt is refused. Size M.
- [ ] M4.3 Metric retriever. Route number questions to the resolver; put the result in the
  prompt as a labeled evidence block, not fused with text. Done when: a return-rate question
  answers from the metric layer with the number cited. Size M.
- [ ] M4.4 Second-domain stub in CI. Add a skeleton `domains/saas_support/` and a CI fixture
  that ingests both domains and asks one question of each. Done when: CI proves the same engine
  serves two domains, on every commit from here on. Size M.

### M5. Knowledge graph

- [ ] M5.1 Neo4j load. Start `neo4j`, load nodes and typed edges from gold, driven by the
  ontology and manifest. Done when: a templated Cypher query returns expected relations. Size M.
- [ ] M5.2 Entity linking. LLM pass at build time links review mentions to canonical
  entity_ids above a threshold; low confidence goes to a review list. Done when: reviews carry
  `mentions` edges to products. Size M.
- [ ] M5.3 Graph retriever, measured. Vector-first then hop from mentioned entity_ids, plus a
  graph-first path for relational questions, templated read-only Cypher with a clause
  allowlist. Done when: a relational golden question answers using graph plus text and the
  score is recorded. Size M.

### M6. The brain: a supervisor over specialist agents (LangGraph)

Fold M1 to M5 into one state machine, then put a supervisor agent on top that coordinates
specialist agents which each own one evidence source and must agree before an answer ships.
The M1 to M5 pipeline functions become the specialists' tools, so nothing earlier is rewritten.

- [ ] M6.1 State, routing, multi-turn. Typed state; nodes understand (with conversation
  history and follow-up rewriting, so "what about size S?" works), route, retrieve, fuse,
  rerank, generate, verify, gate. Done when: all four query types answer, a follow-up resolves
  from context, and routing accuracy is measured on the golden set. Size L.
- [ ] M6.2 Specialist agents. Wrap the existing tools as three specialists behind one common
  finding contract (answer text, evidence contexts, self-scored confidence, citations): a
  Retriever (hybrid text + graph), a Metrics agent (governed numbers from M4), and a Graph
  agent (relations from M5). Done when: each specialist answers its slice of a golden question
  and returns a structured, self-scored finding. Size M.
- [ ] M6.3 Supervisor and agent-to-agent consensus. The supervisor (the orchestrator, the
  "omniagent") decomposes a query, dispatches to the relevant specialists in parallel, then a
  reconciliation step makes them agree: a reconciler node (an LLM judge given each specialist's
  claim plus its evidence) flags contradictions (a metric number contradicting a text claim, two
  sources disagreeing), resolves by evidence rank (governed metric over text over model prior)
  or by asking a specialist to revise, and commits only agreed, grounded evidence, otherwise
  abstains or escalates. All
  bounded by step and token caps; the fan-out fires only for multi-source questions. Done when:
  a mixed number-and-text question is answered by two specialists agreeing, a planted conflict
  is caught and reconciled or escalated, and consensus is shown to beat single-pass on the
  golden set (else it ships behind a flag, not as default). Size L.
- [ ] M6.4 Agent loop and gate. Bounded ReAct over the same specialists for the hard tail, step
  and budget caps, the gate decides auto vs agent vs escalate, calibrated against the golden
  set. Done when: a hard question triggers the loop and an unanswerable one escalates. Size M.
- [ ] M6.5 HITL. Postgres review queue, LangGraph checkpointer so interrupts survive, an honest
  "escalated" reply, admin answers and the answer is stored as gold. Done when: a low-confidence
  question lands in the queue and a human answer closes it. Size L.

### M7. Back-office and the flywheel

- [ ] M7.1 Queue UI. `/admin` queue with row lock on claim and an answer form. Done when: an
  operator can claim and answer a queued item. Size M.
- [ ] M7.2 Quality dashboard. Thumbs, faithfulness, escalation rate, grouped by language. Done
  when: a thumbs-down from M3 shows up per language. Size M.
- [ ] M7.3 Close the flywheel. HITL-approved answers are re-indexed as verified knowledge with
  provenance and appended to the golden set; thumbs data tunes the gate threshold. Done when: a
  human answer becomes retrievable and grows the golden set. Size M.
- [ ] M7.4 Read-only views. Knowledge-gap list, ontology, metric, and dbt lineage viewers, plus
  a link to MLflow. Done when: each renders from real data. Size M.
- [ ] M7.5 Operational monitoring view. An admin page that reads the M1.3 per-request traces
  and shows live platform health: p95 latency, throughput, error rate, cost per request, and
  retrieval-quality trend, stratified by language. This is the continuous-monitoring surface
  that pairs with MLflow (M8.1) and the drift monitors (M8.3). Done when: the page shows real
  numbers from recent traffic. Size M.

### M8. MLOps

- [ ] M8.1 MLflow as viewer. Point the existing M1.3 trace schema at MLflow (tracing already
  exists, this just routes it). Done when: a run shows a full trace with retrieval, tokens, and
  cost. Size M.
- [ ] M8.2 RAGAS on the golden set. Faithfulness, context precision and recall, answer
  relevance, in both languages. Done when: an eval report is produced and matches the M2.5
  scorecard shape. Size M.
- [ ] M8.3 Drift and CI gate. Four named drift monitors (query-embedding drift, retrieval-score
  distribution, confidence distribution, feedback rate), stratified by language, and a CI eval
  gate on recorded fixtures with a cheap judge that blocks a deliberate regression. Done when:
  the gate fails on a seeded regression and passes otherwise. Size L.

### M9. Voice, story, deploy, second domain

- [ ] M9.1 Voice. Groq hosted Whisper for speech-to-text with Web Speech as fallback, browser
  speech for output. Done when: a full voice round trip works cross-browser. Size M.
- [ ] M9.2 Ship the story. Architecture diagram, a 3-minute recorded demo, the published
  bilingual eval report and ablation table, and a one-page decisions-and-tradeoffs doc (Part A
  is most of it). Done when: the README links all four and reads well in 90 seconds. Size M.
- [ ] M9.3 Deploy. Web to Vercel, API to Cloud Run (min-instances 0), Neo4j Aura, Qdrant Cloud,
  Supabase, plus the keepalive job and enforced read-only + rate limit. Done when: the public
  demo login works inside the cost cap and idles to zero. Size L.
- [ ] M9.4 Second domain, full. Flesh out `domains/saas_support/` and switch `DOMAIN`.
  Reproducibility is already proven in CI (M4.4); this makes the second case study real. Done
  when: the same engine answers support questions with no engine code change. Size L.

---

## Part C. Lean file structure

Add folders only when a milestone needs them.

```
skein-lite/
  README.md  docker-compose.yml  .env.example  Makefile  pyproject.toml  uv.lock  .python-version
  domains/
    apparel_ecommerce/    # first pack (fictional brand, synthetic data) + eval/golden.jsonl
    saas_support/         # stub at M4.4, full at M9.4
  adapters/               # llm, embeddings, vectorstore, (later) graph, storage
  ingest/                 # chunk, embed (dense + sparse), index  (M1, M2)
  retrieval/              # vector, metric, graph, fuse, rerank
  pipeline/               # linear pipeline first (M1-M5), becomes the specialists' tools at M6
  rag/                    # LangGraph app, state, nodes, agents/ (supervisor + specialists), eval/  (M6+)
  data/                   # duckdb + dbt project  (M4)
  api/                    # FastAPI  (M3+)
  web/                    # Next.js  (M3+)
  mlops/                  # mlflow config, drift  (M8)
  scripts/                # seed.py, reset.py, keepalive, check_domain_leak.py
  docs/                   # this plan, DEV-NOTES, eval-report
```

Domain-agnostic rule: nothing under `adapters/`, `retrieval/`, `pipeline/`, `rag/`,
`ingest/`, `api/`, `data/` may name a product, metric, brand, or ontology label. Those live
only in `domains/<name>/`. `make check` runs the leak linter to enforce this on every commit.

---

## Part D. What "done" looks like for the whole project

The system is finished when a stranger can judge it in 90 seconds and a skeptic can dig for an
hour and stay convinced:

- One engine, two domains, reproducibility proven in CI (not by hand at the end).
- Retrieval quality shown as an ablation table with real bilingual numbers.
- Every answer grounded, cited, or an honest abstain, with the injection defense demonstrated.
- Every request traced (retrieval, tokens, cost, latency); drift monitored and named.
- A supervisor agent that coordinates specialist agents which agree (or escalate) before an
  answer ships, with consensus measured to beat single-pass, not assumed.
- A human hand-off that visibly makes the system better (the flywheel), not just a queue.
- Runs free locally, a ~$0 hosted demo that idles to zero, portable to Databricks/AWS by
  adapter swap only.
- A short story around it: diagram, recorded demo, eval report, decisions doc.

Pacing, token discipline, and other dev-process notes live in [DEV-NOTES.md](DEV-NOTES.md).
