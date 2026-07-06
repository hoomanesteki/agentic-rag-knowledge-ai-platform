# Model lifecycle and operations

How a change to the serving configuration goes from a laptop to production, how the system improves
itself over time, how it holds up under load and failure, and what it costs to run. This is the
production story built on the pieces the repo already has: the eval gate, the drift monitors, the
review-queue flywheel, the prompt-optimization loop, the resilient adapter layer, and the offline
fakes.

## The rollout ladder

A serving change (a prompt, a model tier, a retrieval setting) climbs a ladder, and each rung has an
objective gate, not a vibe.

1. Dev. Change it, run the eval locally, read the scorecard. Nothing ships on a claim.
2. Offline eval and CI gate. The change runs the routing eval, RAGAS, and the retrieval scorecard;
   `evaluation/ci_gate.py` fails the build if a core number regresses. This is the first gate.
3. Staging. The change runs against the real stores and models with staging config, so integration
   problems surface before any user sees them.
4. Shadow. The new config runs on a mirror of live traffic and serves nothing. Its answers and its
   cost and latency are logged next to production's, so the two are compared on the same real
   inputs. A change that looks good offline but drifts on live phrasing is caught here.
5. Canary. The change serves a small percentage of real traffic. The drift monitors and the eval
   gate watch it; a regression rolls it back automatically.
6. Soft launch. It serves a defined slice (a region, a cohort, signed-in shoppers) while the
   business metrics (containment, assisted conversion, CSAT) are watched against a holdout.
7. Production. Full traffic, with the same monitoring the canary had, because a model in production
   is never done being evaluated.

The promotion criteria are the eval gate and the drift monitors, and promotion that changes served
behavior is human-approved. Nothing self-ships.

## The self-improvement loop

The system gets better from its own traffic, with a human always on the promotion step.

- A turn that is confident and grounded needs nothing. It is served and logged.
- A turn that abstains, is corrected by the shopper, or is flagged low-confidence is written to the
  durable review queue (`rag/hitl.py`). This is the signal worth learning from.
- A reviewer resolves it: a human, or the largest Groq-served model used as a one-off judge on the
  hard cases (the one place a bigger model earns its cost, offline and rarely, not in the serving
  path, and still on Groq so no second vendor is added). The verified answer is the truth.
- That verified answer becomes gold. The flywheel (`rag/flywheel.py`) re-indexes it, and the eval
  set grows from it, so the next evaluation is measured against real failures the system has already
  seen.
- Two more loops feed the same discipline: the prompt-optimization loop proposes better prompts
  against the grown eval set, and the batch enrichment pipeline turns new reviews into governed
  features. All three propose automatically and promote only through a human.

The rule that keeps this safe: automatic proposal, human-gated promotion. A system that rewrites its
own served behavior invites drift and reward hacking, so an agentic coding flow may draft a fix and
open it for review, but it never merges itself. When a fix is small, safe, and the eval confirms no
regression, a human approves it in seconds; when it is not, the log is there for a careful look.

## Continuous training (CT)

The loops above are the discipline; CT is the concrete pipeline that runs them on a cadence. It is a
third loop, distinct from the other two:

- **CI** runs on every push and asks *is the code correct?* (lint, tests, the eval gate).
- **CD** ships once CI is green.
- **CT** runs on a schedule, on measured drift, or on enough new human-verified data, and asks *is a
  retrained candidate better, and safe to promote?*

"Training" here is not gradient descent on model weights, since the LLM is a hosted Groq model. It is
the data-and-prompt layer this system owns, retrained on a cadence and versioned like a model: the
retrieval index (re-embed new reviews and descriptions, incrementally via `run_ingest.py --only`),
the governed enrichment features (recompute consensus on new reviews), and the router and answer
prompts (the OPRO loop). Each is gated the way a model would be.

A CT cycle (`make ct`; policy in `mlops/ct.py`, wiring in `scripts/run_ct.py`, scheduled by
`.github/workflows/ct.yml`) is deterministic control flow:

1. **Trigger.** Fire on the weekly schedule, when the drift monitors cross threshold, or when the
   review-queue flywheel has accumulated enough new verified answers to be worth retraining on. If
   nothing changed, the cycle does nothing and says so.
2. **Retrain.** Re-index the new data and re-optimize the target prompt against the ground-truth
   eval (the safety-gated prompt-optimization loop).
3. **Gate.** Score the candidate on a held-out split and run the same regression gate CI runs.
4. **Propose.** Promote only when the candidate beats the baseline by a margin with the gate and the
   safety check green, and even then only *propose* it: the cycle writes
   `evaluation/reports/ct_report.json`, logs the run to MLflow, and the workflow uploads that report
   as the artifact a human approves. Nothing retrains and ships itself.

This is the self-improvement rule made operational: automatic proposal, human-gated promotion, so a
candidate that games the metric is caught at the gate and never reaches production. It is resilient
too: with no model key the training step is skipped and CT runs the gate as a health check, so a run
always produces an auditable report.

## Drift: detect, decide, act

Four monitors (`mlops/drift.py` and the routing re-eval) watch four kinds of drift: input (the
intent mix shifts), semantic (new meaning or vocabulary), prompt and quality (eval scores slip, or a
provider silently changes a model), and model (the pinned version changes). Detection is automatic
and scheduled. Acting is tiered: safe, reversible steps run on their own (open a review item, re-run
the eval, mark a feature low-confidence, refresh embeddings for changed content); anything that
changes served behavior is human-gated. The knowledge graph is updated the same way: the enrichment
job proposes new entities and relations by consensus, and a human approves a schema-level change.

## Scale and resilience

- Shape. The request path is one image; the batch enrichment job is one worker (see the services
  ADR). Under load, the API scales horizontally behind a load balancer, and an autoscaler adds
  instances on CPU or p95 latency. The stateful stores (Qdrant, Neo4j, Postgres) scale as managed
  services; only their URLs change, because the adapter layer abstracts them.
- Triggers. A queue enters when enrichment needs retries or fan-out; Kubernetes enters when the app
  needs autoscaling or zero-downtime rollout. Both are documented with their triggers, not built
  before they are needed.
- Resilience already in place. `api/resilience.py` retries transient model failures and falls back
  to the cheap 8B model when the primary fails, so a bad minute degrades quality instead of losing
  the turn. The whole stack runs on offline fakes with no keys, which is both a dev convenience and
  a proof that every external dependency is swappable. The posture throughout is degrade, do not
  die.

## Fallback mechanisms

Each external dependency has a planned failure mode and a graceful degrade:

- LLM 429 or outage: retry with backoff, then fall back to the 8B model; if that fails too, return
  an honest "I'm having trouble right now" rather than a broken stream. A 429 is labeled as the
  metered free tier, not an outage, so the demo reads as a rate limit to wait out.
- Vector store down: the retrieval layer surfaces the failure as a degraded event, and the
  confidence gate abstains rather than answering ungrounded.
- Voice service (STT or TTS) fails: text keeps working, and the browser's built-in speech synthesis
  is the fallback voice, so voice degrades to a lower-fidelity voice rather than going silent.
- Rate limit on the app: the limiter returns a clear message; the client backs off.
- Knowledge graph or metric layer unavailable: those specialists simply report nothing found, and
  the turn is answered from retrieval alone.

## How it modernizes over time

The serving configuration is data, not code, so moving to a newer or cheaper model is a config
change and an eval run, not a rewrite. New models are added to the cost model and the tier eval, and
promoted only if the numbers justify them, which is how the Groq-only decision was made and how it
would be revisited. The eval set grows from real failures, so the bar rises as the system runs. The
cost model (`docs/cost-and-business-metrics.md`) is rerun whenever a price or a model changes, so the
economics stay current. Nothing here assumes the model is finished, because it never is.
