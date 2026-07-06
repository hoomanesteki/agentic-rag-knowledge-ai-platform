# Cost and business metrics

What does it cost to serve one shopper, and what could that be worth. Every dollar figure here comes
from `mlops/cost_model.py`, a transparent bottom-up model, so it regenerates and nothing is
hand-typed:

```
PYTHONPATH=. uv run python -m mlops.cost_model   # writes evaluation/reports/cost_model.json
```

The model sums the stages a turn actually runs, with explicit token and usage assumptions you can
change and rerun. It is an estimate for a demo, not a metered production bill, and the assumptions
are stated so the estimate is honest.

## Cost to serve

Per turn, at list prices:

| Tier | Per text turn | Note |
| --- | --- | --- |
| Deterministic route + 8B answer | $0.0021 | the cheapest working path |
| Deterministic route + 70B answer | $0.0031 | the default |
| Frontier (Claude Sonnet 5) | $0.0103 | 3.3x the 70B, for the same turn |
| Frontier (Claude Opus 4.8) | $0.0158 | 5.1x the 70B |
| Human agent | $1.33 | 430x the 70B turn |

Where the money goes in a 70B text turn: retrieval $0.0020 (the Cohere rerank call dominates),
generation $0.0011, routing $0.00001 (routing is essentially free, which is the whole point of the
cheap-first router). A voice turn costs $0.035, and it is not the model that makes it expensive: the
premium text-to-speech is $0.032 of it, the speech-to-text is $0.0001.

Per eight-turn session: text $0.025, voice $0.28, a human agent $10.67.

## The reading

A shopper served entirely by text costs about two and a half cents a session. The same session with
a human agent costs about ten dollars, roughly four hundred times more. A frontier model would
triple to quintuple the text cost for no measured quality gain on this workload (the routing
evaluation showed the cheap tier matching the expensive one), which is the cost case behind the
Groq-only decision. Voice is the one place cost climbs, and it climbs on the premium voice, not the
model, so voice is where a tiered policy matters: use the browser's built-in speech for most turns
and the premium voice only where it earns its keep.

At scale the shape holds. A store handling 10,000 assistant sessions a day spends about $250 a day
on text, roughly $90,000 a year, against a human-staffed equivalent in the millions. The assistant
does not replace the humans; it handles the routine so the humans handle the hard cases, which is
exactly the escalation design: the AI care specialist gathers and files, a person resolves.

## What it could be worth

A demo cannot measure real customer impact, so this is the instrument panel, not a results claim.
The metrics that would tell whether the assistant pays for itself:

- Containment rate: the share of sessions resolved without a human. At $0.025 versus $10.67, even a
  modest containment rate pays for the whole system many times over.
- Assisted conversion and average order value: whether shoppers who use the assistant convert more
  often or add more, measured against a holdout that does not see it.
- Return rate: whether the governed fit features from the enrichment pipeline ("runs small") reduce
  size-driven returns, the most expensive kind.
- CSAT and escalation quality: whether the gather-and-file handoff shortens human handle time.
- Retention and repeat purchase: whether a shopper who had a good assisted experience comes back.

The honest caveat: these are the right instruments and the system is built to emit the events that
feed them (the trace stream, the review queue, the drift monitors), but the numbers themselves need
real traffic and a proper experiment. The behavioral-insights RFC is where that measurement lives.

## Why this is data-driven, not a guess

The prices are vendor list prices, the token and usage assumptions are explicit constants in
`mlops/cost_model.py` with their reasoning, and the output is a committed artifact. Change an
assumption, rerun the command, and every figure in this doc updates. That is the difference between
a cost model and a number in a slide.
