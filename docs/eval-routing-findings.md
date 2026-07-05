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

| Mode | Accuracy | Marginal cost | Escalation precision |
| --- | --- | --- | --- |
| Deterministic (layers 0 and 1) | 84.1% | none | 100% |
| 8B small-model tie-break | 81.2% | ~$0.0005 | 100% |
| 70B large-model tie-break | 83.8% | ~$0.0006 | 100% |

Per stratum, the deterministic router already handles the clear lanes well: care 98%, stylist 96%,
smalltalk 100%, escalation recall 90% at 100% precision. It is weaker on complaint (78%, the
phrasings are open ended) and, as expected, on multilingual (27%, the deterministic cues are
English).

## Finding 1: cheap-first routing wins, and a naive LLM tie-break hurts

The deterministic layers, which cost nothing, are the strongest single mode at 84.1%. Adding a
small-model tie-break on the turns they do not decide actually lowers overall accuracy to 81.2%.
It helps where you would expect (complaint 78 to 90, stylist 96 to 100, multilingual 27 to 67) but
it hurts the two strata where the right move is restraint: on genuinely ambiguous turns it guesses
a confident lane instead of asking (80 to 30), and it pulls general or policy questions into
specialist lanes (answers 73 to 56). The lesson is that an eager classifier is the wrong tool for
the turns a router is least sure about. The fix is a tie-break that defers, which is the concrete
target taken up in the prompt-optimization work.

## Finding 2: a bigger model is not worth it for routing

The 70B tie-break scores 83.8% against the 8B's 81.2%, a 2.6 point gain for roughly the same tiny
per-call cost but about 20x the price per token, and both still trail the free deterministic
router. For the routing task, model tier barely moves the result, and the deterministic layer plus
a cheap 8B is the right architecture. This is the direct, measured evidence behind the project's
decision to stay Groq-only and not add a frontier vendor for classification. The same restraint is
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
configured, and written to a JSON artifact either way. The ambiguous-turn weakness in Finding 1 is
the input to the prompt-optimization loop, which proposes a more conservative tie-break prompt,
scores it on this same set, and only promotes it if it beats the baseline without regressing
safety. That is the measure, find, improve, re-measure loop the MLOps work is built around.
