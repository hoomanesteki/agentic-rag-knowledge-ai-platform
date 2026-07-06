# Architecture and MLOps docs

A map to how this system is built and how it is kept honest. Start here.

## The multi-agent brain

- [omni-plan-v2.md](omni-plan-v2.md) is the unified build plan and the decision log: why Groq-only,
  why the eval set is sized the way it is, why a lightweight prompt loop instead of DSPy, and how
  the work is split across branches.
- The brain itself is a master orchestrator that routes every turn to a lane (stylist, care,
  complaint, answers, escalation) and answers through one gated pipeline, so specialization never
  creates a second, weaker safety surface. Multi-task turns fan out and stitch; a genuine ambiguity
  is asked about, not guessed; escalation files a ready case for a human.

## The evaluation that proves it

- [eval-routing-findings.md](eval-routing-findings.md) is the worked measurement: the deterministic
  router hits 84% at zero marginal cost, a naive LLM tie-break hurt, and a bigger model was not
  worth it. Every number regenerates from `scripts/run_agent_eval.py`.
- [mlops-lifecycle.md](mlops-lifecycle.md) ties the pieces together: offline eval, MLflow tracking,
  the CI gate, human-gated promotion, drift monitoring, and the feedback loops, with a dev-versus-
  prod table and a role-to-surface map.

## Improving prompts against the ground truth

- [prompt-optimization.md](prompt-optimization.md) is the OPRO loop that took the tie-break prompt
  the eval flagged and lifted it from 73.9% to 79.5% on a held-out split, then had a human promote
  it. On the full set the tie-break went from hurting (81.2%) to helping (85.6%).

## Turning reviews into governed features

- [adr-enrichment.md](adr-enrichment.md) is the batch pipeline that turns untrusted reviews into
  confidence-scored product features by consensus, with provenance, served from a feature table.

## Where it runs, and what is next

- [adr-services.md](adr-services.md) is the service and container topology: a modular monolith with
  one batch worker carved out, and the triggers that would introduce a queue and Kubernetes.
- [rfc-behavioral-insights.md](rfc-behavioral-insights.md) is the design-only RFC for a future
  insights layer that turns interaction data into stakeholder reports, built on the pieces already
  in the architecture.

## Running the numbers yourself

```
PYTHONPATH=. uv run python scripts/run_agent_eval.py --with-llm   # routing eval + model-tier A/B
PYTHONPATH=. uv run python scripts/run_prompt_opt.py              # optimize the tie-break prompt
PYTHONPATH=. uv run python scripts/run_enrichment.py             # reviews -> governed features
```
