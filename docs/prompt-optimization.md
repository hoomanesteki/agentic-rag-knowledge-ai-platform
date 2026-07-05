# Prompt optimization

The app has many hand-written prompts, and some of them can be tuned against the ground-truth eval
instead of by hand. This is a small loop that does that: an LLM proposes better prompts, the eval
scores them, and the best one that beats the baseline and passes a safety gate is surfaced for a
human to promote. It is the standard idea behind OPRO and automatic prompt engineering, kept
deliberately small.

## Why not DSPy

DSPy is the well-known framework here, and it is powerful, but it is heavy for this codebase. It
brings its own LM abstraction and a compiler that wants to own the prompting layer, and our prompts
are hand tuned and safety critical (the PII gate, the harm intercepts). Adopting a framework to
optimize a handful of prompts is a large dependency for a small gain. The loop here is about eighty
lines, uses the models and the eval we already have, and keeps the control flow obvious.

## The loop

`mlops/prompt_opt.py` is generic. The caller supplies three functions:

- `evaluate(prompt)` returns a score and the cases the prompt still gets wrong.
- `propose(prompt, failures, n)` returns n rewrites, an LLM reading the failures.
- `safety(prompt)` is a hard gate a candidate must pass.

The loop proposes from the current best, scores each candidate, and keeps one only if it beats the
best by a margin and passes safety. It runs for a few rounds and returns the winner. It never edits
code. Promotion is a person editing the source, because a system that rewrites its own production
prompts invites drift and reward hacking. The winner is written to `mlops/prompt_registry` as a
candidate with the numbers that justify it.

## The worked example: the router tie-break prompt

The routing eval (`docs/eval-routing-findings.md`) found the router's small-model tie-break guesses
a specialist lane on turns it should defer on, which is why enabling it lowered accuracy on the
ambiguous and general-question strata. That prompt is the target.

```
PYTHONPATH=. uv run python scripts/run_prompt_opt.py
```

The run splits the affected strata into a 60/40 train and test, deterministically. A Groq model
proposes more conservative rewrites from the failing cases, each is scored on train, and the winner
is reported on the held-out test so the gain has to generalize rather than memorize.

Result, from `evaluation/reports/prompt_opt.json`:

| Prompt | Train | Held-out test |
| --- | --- | --- |
| Baseline | 75.0% | 73.9% |
| Optimized candidate | 81.8% | 79.5% |

The candidate lifts held-out accuracy by 5.6 points, and the change it made is exactly the one a
person would make: it tells the router to prefer answers for general questions and to reserve
unclear for genuinely two-intent messages, instead of guessing a lane. Because it beat the baseline
on the held-out split and passed the safety gate (it keeps the JSON contract and the lane
vocabulary), it was written as a candidate and then promoted by hand into `rag/router.py`, with the
candidate file and this report as the audit trail.

## Guardrails

- Train and test are split, so a candidate that only fits the training cases does not win.
- The safety gate rejects any prompt that drops the output contract or a lane, no matter its score.
- Promotion is human. An automatic mode may run the loop on a schedule to propose candidates, but
  it never self-deploys. That gate is the scalable, safe choice, and it is the same posture the
  drift action policy takes: propose automatically, change served behavior only with a human.

## Scaling up

The same loop optimizes any prompt with a ground-truth metric: the clarifier, a lane's focus, the
metric slot-fill prompt. The demo runs on one prompt to prove the mechanism cheaply. In a real
system this would run per prompt on a schedule, write candidates, and open a review for a human,
with the eval gate as the backstop before anything merges.
