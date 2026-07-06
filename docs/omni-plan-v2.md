# Skein Lite: Omni-Agent + MLOps Evaluation, unified build plan (v2)

This supersedes the frontier-model parts of `docs/omni-agent-plan.md`. The architecture
design in that document (master orchestrator, lanes as data rows, Send map-reduce, clarify
fork, deterministic guards) still stands. What changed here is the model-tier decision, the
addition of an MLOps evaluation spine, and a prompt-optimization loop. Everything is on branch
`feat/omni-agent`. Never push or merge to main; the owner merges manually after review.

## 0. Decision log (why this plan looks the way it does)

These are the decisions the owner asked to see reasoned and measured, not assumed. They are
themselves a deliverable: the point is to show judgment, cheaply.

1. **Frontier vendor: dropped, pending Experiment 2.** The escalation persona (Tiffany) does a
   structured job: gather the shopper's issue, confirm a numbered list of stated facts, file a
   case, hand back. That is reliability, format-following, and tool use, not hard reasoning,
   which is where a frontier model earns its premium. Adding a separate frontier vendor means a
   new dependency, a new key, per-token frontier cost, and fragile plumbing (cache padding,
   thinking config, refusal handling) for a marginal gain. Decision: stay Groq-only, and prove
   the tier is enough with a real A/B (Experiment 2, section 3). Expected finding: 70B saturates
   the task; 8B handles routing; no frontier justified.

2. **Eval set size: about 500, tiered, not 1000.** Routing accuracy needs volume and is almost
   free to measure (router only, no generation). Answer quality needs depth and costs tokens
   (full agent turns, sometimes on several models). So spend volume where it is free and depth
   where it counts: a routing tier of about 350 cheap cases and a quality tier of about 150
   fuller cases. This is a demo, not a shipped product, so the set is sized to make the point
   without burning tokens.

3. **Eval data honesty: curated synthetic, labeled and spot-checked.** Described exactly that way
   in the docs. A production note states the honest scale-up: human annotation with two
   annotators, adjudication, and an inter-annotator agreement (kappa) report. We do not claim the
   demo set is human-annotated, because the portfolio's value is honest engineering.

4. **Prompt optimization: lightweight OPRO/APE loop, not DSPy.** DSPy is strong but heavy: its
   own LM abstraction, a compiler that wants to own the prompting layer, invasive for prompts
   that are hand-tuned and safety-critical. Instead, a small loop on the ground-truth set: score
   a prompt, let Fable propose variants from the failing cases, re-score on a held-out slice,
   promote only a winner that beats baseline and never regresses safety. Human-gated. Details in
   section 4.

## 1. Three pillars, one shared backbone

- **Pillar A, the product:** the master-orchestrator multi-agent brain (Sara lanes + Tiffany
  escalation), per the v1 design doc, minus the frontier vendor.
- **Pillar B, the proof:** an MLOps evaluation that turns "is multi-agent better" and "do we need
  a bigger model" into numbers logged to MLflow.
- **Pillar C, the maintenance loop:** prompt evaluation and optimization against ground truth,
  safety-gated and human-promoted.

All three sit on one **ground-truth eval set**. Build the eval spine and the single-agent
baseline first, before the new brain exists, because a baseline measured after the change is not
a baseline.

## 2. Ground-truth eval set (the backbone)

Files under `domains/apparel_ecommerce/eval/`:
- `routing.jsonl` (about 350): each case has the user turn, session context (signed-in or
  anonymous), and the **intended lane** label. Cheap: run the router only.
- `agent_quality.jsonl` (about 150): fuller labels (intended lane, is-multi-task, should-clarify,
  is-PII-probe, expected key facts, a short rubric). Expensive: run full agent turns.

Stratification (both tiers, roughly): stylist/gift 15%, care/order-status 15%, complaint 10%,
escalation 8%, product-facts/policy 12%, multi-task 10%, ambiguous/should-clarify 10%,
safety/PII-probe 10%, smalltalk 5%, multilingual 5%.

Generation: curated synthetic via Fable, one batch per stratum, then a deterministic validator
(schema, label sanity, dedupe) and an author spot-check pass. Honest framing in the docs, with
the production human-annotation note.

## 3. Pillar B: metrics and MLflow experiments

Metrics (same set for both brains, logged per run):
- **Routing accuracy** (routing tier): intended-lane match rate, plus a confusion matrix.
- **Task success** (quality tier): rubric score, did it do what was asked.
- **Multi-task completion:** on 2-in-1 turns, were both parts handled. Expected single-agent
  weak spot.
- **Clarify precision and hallucination rate:** on ambiguous turns, ask vs guess/invent.
- **PII-leak rate:** must be 0 for both brains. The refactor must not regress the order-PII gate.
- **Cost per turn by tier**, and **latency**.

Experiments (MLflow experiment `skein-omni-eval`, reusing `mlops/mlflow_sink.py` patterns):
- **Experiment 1, single vs multi:** run the current single-agent brain and the new omni brain
  on the eval set; the delta is the headline result.
- **Experiment 2, model-tier A/B (the frontier decision):** run the escalation and hard slice on
  `llama-3.1-8b-instant` vs `llama-3.3-70b-versatile` vs one larger Groq model if configured
  (probe at runtime, skip gracefully if unavailable). Decide the tier from quality-per-dollar.

## 4. Pillar C: prompt optimization and regression

Components:
- **Prompt registry:** the quality-bearing prompts (Sara system, clarifier, metric slot-fill,
  follow-up rewrite, escalation) as named, versioned entries. Deterministic guards stay code,
  not optimizable prompts.
- **Prompt eval:** score a named prompt (or the whole system) against the ground-truth set into a
  scorecard and an MLflow run. This is the manual "test against ground truth."
- **Optimizer (opt-in, offline):** Fable reads the current prompt and its failing cases, proposes
  N variants; each is scored on a held-out slice; the best that beats baseline by a margin and
  passes safety is surfaced as a **candidate** (a diff), not deployed.
- **Fallback and safety (the backup plan):** baseline is always retained; a candidate must beat
  it on the metric and never regress PII-leak (stays 0) or hallucination; train/test split guards
  against overfitting the optimizer to the eval set; promotion is human-gated. An automatic mode
  may run on a schedule to **propose** candidates, but never self-deploys, because autonomous
  prompt rewriting risks drift and reward hacking. That gate is the scalable, safe choice, and it
  is documented as such.
- **Demo scope:** prove the whole loop end to end on one representative prompt with real
  before/after numbers; document how it generalizes to the rest. Do not run it on every prompt
  (token cost).

## 5. Build phases (each `make check` green, committed on feat/omni-agent)

- **omni.0, cost observability (finishing):** frontier prices trimmed to Groq plus the
  cache-aware cost helper (done, b0efacf); thread `make_small_llm()` into `route_metric` and
  `rewrite_followup` so classification stops paying 70B; add `tier` and `cost_by_tier` to the
  trace dict (the MLflow sink already accepts `tier`).
- **omni.1, eval spine + baseline:** the eval set files, `evaluation/agent_eval.py` (mirrors
  `evaluation/harness.py`), `scripts/run_agent_eval.py`, MLflow logging, and the single-agent
  baseline numbers.
- **omni.2, persona:** Aria retired to Sara; Tiffany added as a Groq-tier escalation persona
  (different voice + avatar shape). Manifest, prompts, frontend, avatar.
- **omni.3, guards:** extract the intercepts to `rag/guards.py`, no logic change; PII stays in
  `retrieve()`.
- **omni.4, router + tools:** `rag/router.py` (3-layer `route()`), `rag/tools.py`. Routing tier
  now scores the real router.
- **omni.5, fast-path lanes + PII parity:** `rag/roles.py` lane rows, `CHAT_BRAIN=omni`
  single-task path.
- **omni.6, heavy path:** `rag/omni_graph.py` (plan, Send fan-out capped at 3, stitch, reroute
  budget 1, output_guard tripwire); Tiffany escalation on the Groq tier with a CaseFile and the
  switch-back to Sara.
- **omni.7, prompt-optimization loop:** the registry, prompt eval, and the Fable optimizer with
  the safety-gated promotion, demonstrated on one prompt.
- **omni.8, run experiments, decide, document:** run Experiments 1 and 2, log to MLflow, write
  the decision report, README section, and Quarto page(s).

## 6. Documentation deliverables

- A README section on the evaluation and the decisions.
- A Quarto page (in `showcase/`) on the multi-agent architecture, the single-vs-multi and
  model-tier results, and the prompt-optimization loop.
- A decision report capturing the frontier call, the sizing call, the DSPy-vs-lightweight call,
  and the annotation stance, each with its number or its reasoning.

## 7. Constraints

Branch `feat/omni-agent`, never push or merge to main. `make check` green each commit. Groq-only.
Order-PII gate strict for anonymous and third-party. No em dashes. Owner identity, no attribution.
Cost-efficient: this is a demo, so size everything to make the point without burning tokens.

## 8. Final verification

A lean 10 Opus + 10 Fable pass on routing correctness, each lane's mastery, the Tiffany contract,
PII parity, and multi-task turns. Report honestly, including anything that did not improve.

## 9. Full scope and the 5-branch delivery stack

The epic grew past the agent brain into a portfolio that shows range across Data Scientist, ML
software engineer, MLOps, and Data Architect. It ships as five well-named branches, none pushed or
merged to main. The owner reviews and merges them manually, in order. Each branch is stacked on
the previous so the PRs stay incremental and reviewable, and each gets an adversarial self-review
pass before it is marked ready.

1. **feat/omni-agent**: the master-orchestrator multi-agent brain (Sara lanes + Tiffany
   escalation, Groq-only). Phases omni.0 (done) through the heavy path.
2. **feat/mlops-eval**: the ground-truth eval set, the agent-eval harness, MLflow experiments
   (single vs multi, model-tier A/B for the frontier decision), the MLOps dev-to-prod lifecycle
   doc, and the drift design and monitors (section 11).
3. **feat/prompt-optimization**: the lightweight OPRO/APE prompt-optimization loop (section 4).
4. **feat/data-enrichment**: the batch AI-annotation and feature-engineering slice with a
   containerized worker and its ADR (section 10).
5. **docs/insights-architecture**: the future behavioral-insights RFC and the system-design and
   services ADR, design only (sections 10 and 12).

## 10. Data enrichment and the services and containers architecture

Enrichment (build a lightweight slice, plus an ADR): descriptions are authored, stable, trusted,
low-churn, so enrich once at ingest and re-run only on a content-hash change; their features can
feed retrieval filters directly. Reviews and comments are user-generated, high-churn, and
untrusted, so never trust one alone. Compute their features in a periodic batch with an
AI voting and consensus rule (a signal is promoted only when several sources agree, with a
confidence score), a live-at-submit moderation, PII, and injection gate, and embedding plus
clustering to surface themes and defect spikes. Derived features land in the DuckDB lakehouse as a
feature table with provenance (source, model, date, confidence). Why batch, not live: a single new
review must not flip a product feature, consensus needs a window, drift is a windowed comparison,
and amortized batch is cheaper and more governable. It gets faster and cheaper at scale by
content-hash skipping and serving precomputed features.

Services and containers: the app stays a modular monolith. The one piece that becomes its own
containerized, schedulable worker is the batch enrichment job. Services talk over HTTP now; the
ADR marks where a queue or event bus goes at scale, and gives docker-compose dev and prod profiles
and the k8s scale-up path. No shattering into many microservices for a demo.

## 11. Drift monitoring and the action policy

Four drift types, one monitor to detect to decide to act policy. Input and data drift (the intent
mix shifts): PSI plus embedding-space distance on rolling windows. Semantic and embedding drift
(new meaning or vocabulary): surfaced by the enrichment clustering. Prompt and quality drift (eval
scores regress, or a provider silently changes a model): the CI eval gate plus a scheduled
re-eval on the ground-truth set. Model drift: pin the Groq model version, detect with a canary
eval. Detection is automated and scheduled. Acting is tiered: safe reversible steps run
automatically (open a review-queue item, re-run eval, mark features low-confidence, refresh
embeddings for changed content); anything that changes served behavior (promote a prompt, update
the Neo4j graph, act on a merchandising signal) is human-gated, and an agent may draft the
diagnosis and proposed action but never self-deploys. This extends the existing mlops/drift.py
monitors rather than replacing them.

## 12. Documentation, reproducibility, and the writing bar

Deliverables: Quarto pages with real visualizations, README sections, and reports or notebooks
where they fit. Every figure and number is reproducible: pinned seeds, versioned fixtures and eval
sets, and a make target or script that regenerates them, with no hand-typed metrics. All prose
reads as human-written, with no AI tells and no em dashes (see the docs-writing-standard memory).

## 13. Final integration and CI/CD gate

Beyond per-commit `make check`, the epic ends with a full integration pass: the whole test suite,
the leak linter, the eval gate, and the CI/CD checks all green across the stacked branches, with
the services wired together and exercised in the common, uncommon, and edge cases. If anything
fails, fix and re-run until it passes. Nothing is called done until this is green.
