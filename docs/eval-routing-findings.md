# Routing evaluation: what the numbers say

The omni brain routes every shopper turn to one lane before it answers. This is the measurement of
how well that router works, and two decisions fall out of it. Every number here regenerates from
one command, so nothing is hand-typed:

```
PYTHONPATH=. uv run python scripts/run_agent_eval.py --with-llm
```

It reads the labeled set at `domains/apparel_ecommerce/eval/routing.jsonl` (350 cases across 10
strata) and writes `evaluation/reports/routing_eval.json`.

## The set

350 curated synthetic cases, each a realistic shopper message labeled with the lane a thoughtful
person would intend: stylist, care, complaint, escalation, answers, plus multi-task, ambiguous,
safety probe, smalltalk, and multilingual strata. It is labeled and spot-checked, not human
annotated. In production this tier would be human annotated with two annotators and adjudication,
and an inter-annotator agreement score reported. For a portfolio demo, a curated synthetic set of
this size makes the point without the cost, and it is described honestly as such.

## Results

These are the current committed numbers from `evaluation/reports/routing_eval.json`, after the
prompt-optimization and the two verification-driven hardening rounds. The journey that produced
them is Finding 1.

| Mode | Accuracy | Marginal cost | Escalation |
| --- | --- | --- | --- |
| Deterministic (layers 0 and 1) | 81.6% | none | 100% precision, 84% recall |
| 8B small-model tie-break | 85.9% | ~$0.0005 | 100% precision, 100% recall |
| 70B large-model tie-break | 85.6% | ~$0.0006 | 100% precision, 100% recall |

Per stratum with the 8B tie-break (the production path): care 95%, stylist 98%, smalltalk 100%,
escalation 100%, complaint 82%, multilingual 87%. The deterministic layer alone is weaker on
complaint (60%) and multilingual (27%), because a bare damage word ("are these scuffed") is
genuinely ambiguous with a product question and the cues are English; those are exactly the turns
handed to the cheap tie-break.

## Finding 1: cheap-first routing wins, and the tie-break had to be taught to defer

The story matters more than any single number. When the tie-break was first added with a naive
prompt, it LOWERED accuracy: it guessed a confident specialist lane on genuinely ambiguous turns
instead of asking, and pulled general questions into specialist lanes. That is the finding that
motivated the prompt-optimization loop, which rewrote the tie-break prompt to defer, taking it from
73.9% to 79.5% on a held-out split. With the deferring prompt (and, after the last verification
round, an escalation option so a missed human request can still reach a person), the 8B tie-break
now HELPS: 81.6% deterministic to 85.9%. The lesson holds either way: an eager classifier is the
wrong tool for the turns a router is least sure about, and the fix was to make it defer, not to
make it bigger.

## Finding 2: a bigger model is not worth it for routing

The 70B tie-break scores 85.6% against the 8B's 85.9%, so a model roughly 20x the price per token
does not even edge out the cheap one on this task. Model tier barely moves the result, and the
deterministic layer plus a cheap 8B is the right architecture. This is the direct, measured evidence
behind staying Groq-only and not adding a frontier vendor for classification. The same restraint is
applied to the escalation persona, whose task (gather, confirm, file) is structured rather than
reasoning heavy.

## Honest limitations

- Multilingual routing leans on the LLM tie-break, since the deterministic cues are English. That
  is a known trade-off, not a bug: the cheap layer covers the common case and the model covers the
  long tail.
- The set is synthetic. The router cues were written from phrasings a person would recognize, not
  by fitting the specific cases, and the ambiguous and answers strata are graded on whether the
  brain defers rather than on a single lane, so a router cannot score well here by memorizing.
- These numbers measure routing only. Full-turn answer quality (retrieval, grounding, the PII gate)
  is measured by the existing retrieval scorecard and the safety suite.

## Where this plugs into the lifecycle

The scorecard is logged to MLflow (experiment `skein-omni-eval`) when a tracking server is
configured, and written to a JSON artifact either way. The naive-tie-break weakness in Finding 1
was the input to the prompt-optimization loop (`docs/prompt-optimization.md`), which proposed a more
conservative tie-break prompt, scored it on this same set, and promoted it only after a human
reviewed the held-out gain. That is the measure, find, improve, re-measure loop the MLOps work is
built around, and it is why the numbers in the table are the current post-optimization state and
regenerate from `scripts/run_agent_eval.py`.
