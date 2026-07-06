# Production-readiness epic: plan and branches

This is the third epic on top of the omni-agent brain and the MLOps stack. It makes the whole thing
read as a finished, production-minded system: the showcase reflects what was actually built, the
numbers are reproducible not hand-typed, and the production story (lifecycle, self-improvement,
cost, scale, resilience, fallbacks) is designed and documented. Owner merges branches manually, in
order; nothing is pushed to main.

## Branches (merge order)

1. `docs/quarto-showcase`: the Quarto site and the notebooks reflect the omni-agent brain, the
   routing eval, the prompt-optimization loop, and the MLOps experiments. Aria is retired to Sara
   and Tiffany. Metrics are pulled from the committed artifacts by code chunks, not hand-typed, so
   the site regenerates. Storytelling and visual, precise text, clean structure.
2. `docs/production-lifecycle`: the design docs that were requested. Model lifecycle from dev to
   staging to shadow to canary to soft launch to production; the self-improvement loop; drift
   detection and resolution; scaling, resilience, and auto-scale; fallback mechanisms; a cost and
   business-metrics model; and a short competitor best-practice benchmark.
3. `feat/self-improve-loop` (optional build): a lightweight slice that wires the pieces that already
   exist (the review-queue flywheel, the prompt-optimization loop, the drift monitors) into one
   documented feedback loop, with a small demo, rather than new heavy machinery.

## What each must satisfy

- Reproducible: every number in a doc, notebook, or Quarto page comes from a committed artifact or a
  code chunk that regenerates it. No hand-typed metrics.
- Honest and human-written: no AI tells, no em dashes, storytelling with real visualizations.
- Consistent: the numbers match the code word for word (462 tests, deterministic routing 80.9%, 8B
  tie-break 84.1%, 70B 83.1%, escalation 100% precision and 90% recall, enrichment 486 reviews to 11
  governed features, prompt-opt held-out 73.9% to 79.5%).
- Everything integrates: `make check` green, no domain leaks, the branch stack merges cleanly.

## The design topics to cover (docs/production-lifecycle)

- Lifecycle and rollout: offline eval, then staging, then shadow (mirror live traffic to the new
  config, compare, serve nothing), then canary, then soft launch to a slice of users, then full. The
  eval gate and the drift monitors are the promotion criteria.
- Self-improvement loop: a turn that is confident and grounded needs nothing; a turn that abstains,
  is corrected, or is flagged low-confidence is logged to the review queue; a human or a frontier
  model reviews it; the verified answer becomes gold that the flywheel re-indexes and the eval set
  grows from; the prompt-optimization loop and the enrichment pipeline are the other two loops.
  Automatic proposal, human-gated promotion, always.
- Drift: the four monitors (input, semantic, prompt/quality, model) detect; a tiered action policy
  resolves (auto for safe reversible steps, human-gated for anything that changes served behavior);
  an agentic coding flow may draft a fix but never self-ships.
- Scale and resilience: how the monolith plus one batch worker scales, where a queue and k8s enter,
  auto-scaling triggers, and the resilience already in place (retries, small-model fallback, the
  offline fakes, the degrade-not-die posture).
- Fallbacks: the LLM 429 or outage, the vector store down, the voice service failing, the rate
  limit, and the graceful degrade for each.
- Cost and business metrics: the cost per served user with a data-driven estimate and its sources; a
  comparison of the tiers (deterministic, 8B, 70B, a frontier model, a human agent, premium voice)
  in dollars; and how the assistant plausibly affects customer lifecycle and purchase behavior, with
  the metrics to measure it and honest caveats about what a demo can and cannot claim.

## Deferred: the deep Fable + Opus verification

The 12-dimension comprehensive verification (agents, edge cases, PII/safety, enrichment, prompt-opt,
MLOps reproducibility, docs, integration) hit the session limit and must be re-run after it resets.
It triages findings into urgent, semi-urgent, and not-urgent, and everything real gets fixed before
the final go.
