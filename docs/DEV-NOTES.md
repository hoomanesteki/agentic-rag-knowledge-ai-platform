# Dev notes

Process and pacing notes, kept out of the build plan so that file reads clean for anyone
evaluating the project.

## Toolchain

- Package manager: uv. It manages the Python version too (pinned to 3.12 in `.python-version`).
- Dependencies: ranges in `pyproject.toml`, exact versions pinned in `uv.lock` (committed).
  That is what makes installs reproducible. Add a dep with `uv add <pkg>`, then commit the
  updated lock.
- One gate: `make check` runs ruff lint, pytest, domain-pack validation, and the domain-leak
  linter. CI runs the same target. The `/preflight` skill runs it and reports go or no-go.

## Working on Claude Pro without burning tokens

Claude Pro has usage limits, so keep each building session small and let the repo hold the
state instead of the chat.

- One milestone step per session. Start by reading BUILD-PLAN.md plus only the files that
  step touches. Do not ask Claude to re-read the whole repo.
- Keep state in the repo, not in chat. After each step, run `make check` and commit. When the
  conversation gets long, reset it safely: the plan and code hold the state.
- Use the `/domain-pack` skill to generate and check packs instead of reasoning them out each
  time. Deterministic scaffolding is cheaper than fresh generation.
- Let the app make its own model calls (Groq, Voyage). That work does not touch your Claude
  quota. Claude is for building, not for serving answers.
- Write each "Done when" as a command you can run. A runnable check ends the debate about
  whether a step is finished.
- Prefer small diffs. If a step feels like an L, split it into two S sessions.

## Evaluation and the rerank delta

`make eval` scores the golden set. To record the reranker's effect (M2.2 done-when), run it
both ways against a real Voyage + Qdrant index and compare `hit@k` / `mrr` / `entity_recall@k`:

```bash
make up && make ingest
PYTHONPATH=. uv run python scripts/run_eval.py --no-rerank   # hybrid baseline
make eval                                                    # hybrid + rerank
```

Only the real Voyage run counts. The offline `RERANK_PROVIDER=fake` reranker scores by word
overlap, the same signal as the sparse leg and the abstain gate, so its "delta" is circular
and must not be recorded as the M2.2 number. Paste the two scorecards and the delta here once
you have run them (this environment has no network, so the numbers come from your machine).

## M2.3 grounding and injection (known limits)

The abstain gate holds unchanged, so abstain precision holds by construction; still run
`make eval` on a real index and record the number here to close the done-when. The injection
sanitizer is defense in depth next to the system prompt: it redacts to sentence end (a payload
split into a separate sentence from its trigger survives), and covers English and French plus
zero-width bypasses, not every phrasing. The grounding score measures citation discipline, not
faithfulness (RAGAS at M8). Two follow-ups for M8: compute the gate on sanitized text so an
injection cannot inflate confidence, and add an adversarial question to the golden set.

## M2.4 chunking (scope and follow-ups)

The seed reviews are single sentences, so the sentence packer produces one chunk each and is
a retrieval no-op on this corpus; it is exercised by unit tests, and the real recall delta is
recorded via the M2.5 ablation on a real index. The contextual prefix is opt-in per pack
(`context_fields` in the manifest) and is embedded with the text but kept out of the stored
and displayed text, so it cannot pollute citations or the abstain gate. Whether the prefix
helps is measured, not assumed. Tracked follow-ups (M4, when data grows): delete points by
`record_id` before re-upsert so a record that shrinks from N to fewer chunks does not orphan
old points; an intra-sentence window fallback for a single sentence over the token budget;
and preserving paragraph structure.

## M4 review follow-ups (deliberate deferrals)

An end-to-end review after M4 surfaced items fixed in place (broadened the SSE error catch so a
non-RuntimeError cannot kill the stream silently, attributed feedback to the logged-in user,
warned when the lakehouse is missing, dropped bare "how" from the metric pre-gate, and
strengthened the dual-domain test to assert real retrieval plus the metric layer per domain).
Three were deferred on purpose:

- Pack `defaults` (confidence_high/low, top_k_in/top_n_out) are not read yet. This is not dead
  config: the confidence bands feed the M6 tiered gate (auto/agent/escalate), which does not
  exist yet, and the M1 gate is a single global threshold tuned against the golden set. Wiring
  per-domain thresholds now would put unjustified numbers into the gate, against the rule that
  no threshold ships without a golden-set number behind it. Wire them at M6, tuned per domain.
- Metric SQL guard is regex plus a read-only, external-access-off connection. A pack author
  could still reach a raw layer via a table function like `query_table('bronze_x')`, which the
  regex misses. Out of the current threat model (the model fills params only; templates are
  pack-authored and reviewed), so left as is. Harden at M6 by building gold into a separate
  schema the resolver attaches, instead of denylisting.
- Raw user queries persist unmasked in traces and feedback (needed for eval and debugging),
  while structured PII is masked in the lakehouse. Revisit with a retention/repro policy at M8
  (trace to MLflow) so query logging is a governance decision, not an accident.

## M5 knowledge graph (local run and follow-ups)

The graph layer is fully tested on the in-memory fake, so `make check` proves the loader,
linking, and retriever logic with no database. The real path is a local step:

```bash
make up                 # starts Neo4j (neo4j:5-community) alongside qdrant/postgres
make lakehouse          # builds gold (the graph loads from it)
make graph-load         # loads nodes + typed edges into Neo4j, then links mentions
```

Then open http://localhost:7474 (neo4j / skein_password) and confirm a traversal, for example
`MATCH (s:Supplier)-[:SUPPLIES]->(p:Product) RETURN s.name, p.name LIMIT 5`. `make graph-load`
prints node and edge counts; paste them back so the real load can be checked against the fake.

To record the M5.3 done-when number (does the graph help), run `make eval` twice on a real
index, once with `GRAPH_PROVIDER=neo4j` and once with `GRAPH_PROVIDER=memory` (graph off), and
compare the relational-question scores. Only the real run counts; paste the delta here.

Two review findings were deferred to their natural milestone on purpose:
- A query that merely names an entity currently suppresses abstain even when the question is not
  relational (asking an attribute the graph does not hold). A lexical guard would regress common
  synonyms ("how much does X cost" vs a `price` property, "makes" vs `SUPPLIES`), so the honest
  fix is intent routing at M6, where a node classifies relational vs qualitative before deciding.
  Until then the grounded prompt still makes the model say it lacks the fact; only the tier label
  is optimistic.
- The metric router's per-query LLM pre-call is not counted in the trace, so an abstain trace can
  report cost 0.0 while a small routing call happened. Fold full token accounting in at M8 when
  the trace feeds MLflow and cost is the point.

Known follow-ups (not blocking M5):
- The retriever builds its entity name index once at construction and `get_components` caches it
  for the process, so a graph reloaded while the API runs serves stale names until restart. Add
  a rebuild hook when the flywheel (M7.3) starts writing to the graph.
- `find_nodes(label)` scans a label; the retriever does it per label at startup and `neighbors`
  runs per resolved entity per query. Fine at demo scale; add a Neo4j full-text index on names
  and cap fan-out if a domain's catalog grows large.
- Entity linking calls the LLM per doc at build time. Fine for these packs; batch and cache by
  text if a corpus grows.

## M6 the brain (supervisor and consensus)

The chat graph (M6.1, `rag/graph.py`, `run_chat`) is the single-pass brain and stays the default
answer path. The supervisor (M6.3, `rag/supervisor.py`, `run_supervised`) dispatches to the
specialists and reconciles their findings; it is available and tested but is NOT wired as the
default yet, on purpose. The rule is that consensus ships as default only if it beats single-pass.

Deciding that is an answer-quality comparison, not a retrieval one: `make eval` scores recall and
the abstain gate, which the supervisor does not change. The real measurement is RAGAS answer
faithfulness and correctness (M8) run once through `run_chat` and once through `run_supervised` on
the golden set, plus the planted-conflict set where a governed metric and a review disagree; the
supervised path should win or tie while resolving the conflict to the governed number. The
mechanism is already proven by the M6.3 tests (two specialists agree, a planted conflict is
flagged and resolved, a wrong-but-cited synthesis falls back to the governed value). Until the M8
comparison, the supervisor ships behind the call site, not as the default answer path.

Wiring: the chat API serves `CHAT_BRAIN=linear` (streams tokens via the M1 path, the default) or
`CHAT_BRAIN=agent` (the full M6 brain: supervisor, gate, and escalation to the review queue, as a
buffered SSE response). The agent path is wired and tested end to end (an escalated question
enqueues to the review queue), so the brain is switchable, not shelf-ware; it stays off by default
until the M8 consensus comparison and because it does not stream token by token yet.

Three end-to-end findings were deferred to their consuming milestone: `confidence` means lexical
overlap in the linear/graph paths and evidence strength in the supervisor/agent paths (reconcile
at M7.5 when the monitoring view reads it, likely as `overlap_confidence` vs `evidence_confidence`);
the LangGraph checkpointer persists state in-process (MemorySaver) but cross-restart HITL resume
needs SqliteSaver and a thread_id config, wired at M9; and per-call token accounting still omits
the reformulation and metric slot-fill calls, folded in at M8 when the trace feeds MLflow.

`detect_conflict` is a numeric heuristic: it only flags a governed number disagreeing with a
number in a review chunk that shares a content word with the metric subject, so incidental numbers
(sizes, ids, shipping windows) do not fire. The flag never changes the answer by itself. Evidence
rank plus a post-synthesis check do: if the model answers with a non-governed number under a
conflict, the reconciler ships the governed evidence and marks the conflict unresolved (which M6.4
turns into an escalation). A real semantic judge is an M6.4 concern.

## M7 back-office (admin hardening deferred to deploy)

The admin queue (M7.1) works in dev. Two hardening items are deferred to M9.3 (deploy), when the
real secrets are set: the admin login page needs the Turnstile widget the customer page already
has (otherwise, once TURNSTILE_SECRET_KEY is set, admin login 403s), and running with the default
JWT_SECRET must be refused in production (today it logs an error, since admin tokens are forgeable
with the default). A stale claim (older than 15 minutes) is auto-reclaimable so an abandoned item
never sticks.

## Git and attribution

Commits use your own git identity. No assistant attribution goes into commit messages or PR
bodies. Work on short-lived `build/<step>` branches, run `make check`, then open a PR (CI runs
on it) and merge when green.
