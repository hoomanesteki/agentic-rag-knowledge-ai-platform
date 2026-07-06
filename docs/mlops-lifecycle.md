# The MLOps lifecycle in this repo

The ML surface here is not a trained model. It is a serving configuration: the retrieval stack, the
prompts, the router, and the model tiers, selected by config. "Improving the model" means changing
one of those and proving it is at least as good as what runs today. This is how that proof is
organized, from a change on a laptop to something safe to serve, and how it feeds back.

## The loop

1. Offline evaluation. Every quality claim is a number from a committed set. Retrieval and the
   abstain gate are scored by `evaluation/harness.py` on the golden set. Routing is scored by
   `evaluation/agent_eval.py` on `domains/apparel_ecommerce/eval/routing.jsonl`. Answer quality
   uses RAGAS in `evaluation/ragas_eval.py`. Each writes a scorecard that regenerates from one
   command, so the numbers in the docs are never hand-typed.

2. Experiment tracking. Runs land in MLflow. In dev that is a local file store, so a fresh checkout
   tracks experiments with no server. In prod it is the tracking server on Postgres behind
   `MLFLOW_TRACKING_URI`. The per-request traces the app already writes are routed into MLflow by
   `mlops/mlflow_sink.py` (route, model, tier as params; latency, tokens, cost, grounding as
   metrics), and the routing experiment logs its scorecards to the `skein-omni-eval` experiment.
   Logging degrades to the JSON artifact when no server is reachable, so the eval never fails
   because tracking is down.

3. The gate. `evaluation/ci_gate.py` runs the eval on recorded fixtures in CI and fails the build
   if a core number regresses. A change does not merge on a claim; it merges on a passing gate.

4. Promotion. A candidate (a prompt, a tier, a retrieval setting) is promoted only when it beats
   the baseline on the metric and does not regress safety. Promotion stays human gated. The
   prompt-optimization loop can propose candidates automatically, but it never self-deploys, which
   keeps prompt drift and reward hacking out of production.

5. Production monitoring. The same traces feed the four drift monitors in `mlops/drift.py`. Drift
   detection is automatic and scheduled; acting on it is tiered. Safe, reversible steps run on
   their own (open a review-queue item, re-run the eval, mark a derived feature low confidence,
   refresh embeddings for changed content). Anything that changes served behavior (promote a
   prompt, update the knowledge graph) is human gated, with an agent allowed to draft the
   diagnosis but not to ship it.

6. Feedback. Two loops close the system. The human-in-the-loop review queue turns escalated turns
   into verified answers that the flywheel re-indexes as gold. The batch enrichment pipeline turns
   new reviews into product features. Both feed the next round of evaluation.

## Dev versus prod, concretely

| Concern | Dev | Prod |
| --- | --- | --- |
| Experiment tracking | MLflow local file store | tracking server on Postgres |
| Providers | offline fakes, no keys | Groq, Cohere, Qdrant, Neo4j |
| Eval | run locally, JSON scorecard | same eval as the CI gate |
| Promotion | inspect the diff and the numbers | human approves, gate must be green |
| Monitoring | traces on disk | drift monitors on the trace stream |

## What each role in this touches

- Data Scientist: the eval design, the metrics that actually separate good from bad routing, and
  the honest reading of the numbers (the tie-break finding).
- ML software engineer: the router, the lanes, the tools, and the gated serving path.
- MLOps: the tracking, the gate, the promotion discipline, the drift monitors, and the fact that
  every number is reproducible.
- Data Architect: the medallion split for enrichment, the feature store on DuckDB, the service
  and container topology, and where a queue and k8s enter at scale.

## Why the numbers drove decisions

The routing eval (`docs/eval-routing-findings.md`) is the worked example. It showed the cheap
deterministic router beats a naive LLM tie-break, and that a 70B tie-break barely beats an 8B while
both trail the free layer. That is why the system stays Groq-only and leans on deterministic
routing with a cheap fallback, rather than reaching for a bigger model. The decision is in the
numbers, not in a preference.
