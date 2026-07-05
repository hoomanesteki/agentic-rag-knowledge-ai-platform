# Final verification and hardening

Before calling the stack done, an adversarial verification ran over it: 21 agents (10 breadth
probes and 10 depth audits across ten dimensions, then a synthesis) trying to break the routing,
the multi-task handling, the escalation contract, the PII gate, the enrichment, and the prompt loop.
This is what it found and what changed.

## The headline

- PII parity is solid. Two independent auditors traced every omni path (single-task, multi-task
  stitch, escalation confirm) back through the same gated retrieve() and confirmed no cross-shopper
  disclosure. The escalation confirm echoes only the signed-in shopper's own proven identity.
- The router lane cues had overfit. The cues that lifted routing on the synthetic set (63% to 88%)
  also misfired on real apparel vocabulary: escalation fired on "gift for a person" and "staff
  picks", complaint fired on "ripped jeans" and "stain-resistant". The synthetic set did not
  contain those phrasings, so the tuning never saw them.

That second point is the honest lesson. I tuned the cues against a synthetic eval and overfit to it.
The adversarial verification, which read real vocabulary the tuning had not seen, caught it. That is
the reason to verify against something the optimization did not touch.

## What was fixed

- Escalation false positives: the intercept was rewritten to high precision (a support verb landing
  on a human noun, or an unambiguous phrase), with the broad "to a <person>" and "want a <person>"
  branches removed, a refusal guard added ("I don't want a human" no longer escalates), and the
  ambiguous nouns ("staff", "reps", "operator") dropped. A false positive here files a real case, so
  precision matters more than recall, and the frontend plus the 8B tie-break cover the rest.
- Complaint and care cues tightened: garment-damage words match only in a problem frame ("arrived
  torn"), so catalog styles do not read as complaints; the order-id cue is case-sensitive so a
  lowercase product code is not an order; return and money-back requests now route to care.
- Enrichment: a tie no longer promotes a feature (strict majority), a duplicate review id no longer
  double-votes, a negated fit phrase is not annotated as its opposite, and an allowlist stops an
  injected annotator value from ever reaching a served feature.
- Prompt optimization: the safety gate now checks the candidate's behavior, not just its text, so a
  degenerate prompt that collapses every turn to one lane is rejected, and promotion requires a real
  held-out margin.
- Multi-task: a late complaint clause is no longer dropped by the clause cap (the cap is applied
  after complaint-first ordering), an answers sub-question is answered and stitched in rather than
  filtered out, and the escalation reply no longer claims a case was filed when none was.

Eleven regression tests were added so none of these can silently return.

## Numbers after hardening

Regenerated from `scripts/run_agent_eval.py` (in `evaluation/reports/routing_eval.json`):

| Mode | Accuracy | Note |
| --- | --- | --- |
| Deterministic | 81.9% | down from 84.1%: the honest cost of removing the overfit cues |
| 8B tie-break | 84.7% | the production number, still the best, still beats 70B |
| 70B tie-break | 83.4% | a bigger model still is not worth it for routing |

The deterministic number dropped because precision went up: the routes it now declines to guess are
handed to the cheap tie-break, which is why the 8B mode is the strongest. The escalation intercept
keeps 100% precision on the eval with recall back at 90%.

## Known residuals, documented not hidden

- The shared `problem_intent` guard in the linear pipeline still fires on bare damage words, so
  "ripped jeans" can read as a complaint on both brains. That is long-standing linear-brain behavior
  and out of scope for this hardening; the fix is to frame-anchor it in the shared guard.
- The clarify question is emitted in English; a non-English shopper on an ambiguous turn should get
  it in their language.
- The shipped UI intercepts an escalation phrase client-side and shows the specialist intro without
  calling the backend, so the backend case-file path is exercised by the API and the eval; wiring
  the UI to it is a small follow-up.

## Verdict

The integration gate is green on the full stack: 461 tests pass, no domain leaks, the eval gate is
1.0. The stack is ready for a manual merge in order. The residuals above are documented follow-ups,
not regressions.
