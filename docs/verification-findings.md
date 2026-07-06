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
  on a human noun, or an unambiguous phrase), with the broad "to a person" and "want a person"
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
| Deterministic (layers 0 and 1) | 81.6% | free; the safe floor, ambiguity deferred to the tie-break |
| 8B tie-break | 85.9% | the production path; escalation recall reaches 100% here |
| 70B tie-break | 85.6% | a bigger model still is not worth it for routing |

The deterministic number is lower because precision went up: the routes it now declines to guess are
handed to the cheap tie-break, which is why the 8B mode is the strongest. The escalation intercept
keeps 100% precision, and with the tie-break's escalation option added in the final round, escalation
recall reaches 100% on the 8B path.

## The confirmation round

Because the first hardening added new patterns, a second focused pass (six auditors) re-checked the
fixed areas and asked the right question: did the fix introduce a new defect? It did. The looser
escalation recovery patterns had a branch with no clause-boundary guard, so a gift phrase like "a
scarf to hand to a person as a present" escalated again, and the complaint frame was defeated by the
bare copula "is", so "is this stain-resistant" read as a complaint. Both were fixed by giving every
loose branch a clause-boundary guard, dropping the weak "is" frame word plus a negative lookahead
for "-resistant", and, the deeper fix, routing complaint from the frame-anchored cue ONLY, never the
shared bare-damage-word guard, so a genuinely ambiguous "ripped jeans" is deferred to the tie-break
instead of decided wrongly at layer 1. The lesson repeats: each round of heuristics needs its own
adversarial check, and the durable answer is to defer ambiguity to the model rather than add another
rule. Sixteen adversarial checks and the regression tests now cover both rounds.

## The comprehensive round

A final wide pass (a Fable breadth probe and an Opus depth audit per dimension) covered routing,
edge cases and non-English, each lane's mastery, the escalation contract, multi-task, clarify, PII
and safety, enrichment, the prompt loop, MLOps reproducibility, and the docs. PII parity held again,
and a real set of issues was fixed: the negation guard was suppressing genuine human requests ("No,
I want a human"), so it now only catches real refusals; the small-model tie-break gained an
escalation option, so a human request the English-only Layer 0 misses (a non-English phrasing, a
typo) can still reach a person, taking escalation recall on the 8B path to 100%; the complaint cue
gained high-precision failure-verb patterns ("the zipper broke"); enrichment drops a malformed
annotator return instead of crashing; a multi-task turn carrying a human request now escalates the
whole turn and files a case, and it answers each distinct lane once so a comma-split complaint does
not double-apologize; the care lane now surfaces a signed-in shopper's order docs; and the routing
numbers in every doc were corrected to the committed values (the docs had gone stale across the
rounds, which is why their numbers now regenerate from the artifact).

## Known residuals, documented not hidden

- The clarify question is emitted in English; a non-English shopper on an ambiguous turn should get
  it in their language. The rest of the pipeline already replies in the shopper's language.
- The shipped web widget intercepts an escalation phrase client-side and shows the specialist intro
  without posting the turn, so the backend case-file path is exercised by the API and the eval;
  wiring the widget to post the turn (so a real case is filed from the browser) is a contained
  frontend follow-up.
- The order-PII gate uses name-plus-email as its two factors; a regulated deployment would add a
  stronger factor and an audit log, as the decisions page already notes.

## Verdict

The integration gate is green on the full stack: 469 tests pass, no domain leaks, the eval gate is
1.0. Three adversarial rounds and their regression tests now cover the routing, escalation,
multi-task, enrichment, and prompt surfaces, and PII parity held every time. The stack is ready to
merge in order.
