# MLOps: experiments and drift

A snapshot of what the pipeline tracks, so the loop is visible without spinning up the stack. All of
it is generated from the request traces and evals the app already writes, with `mlflow-skinny` for
tracking and PSI-style monitors for drift. Regenerate any of it with the `make` targets below.

## Experiment tracking (MLflow)

A "model" here is the whole serving config: LLM, embeddings, reranker, and brain. Promotion is a
gate, not a rubber stamp. Every candidate is logged to MLflow as a run, and its stage
(`dev -> Staging -> Production`) is decided in this order:

1. **Pipeline smoke gate** (precondition). The offline eval gate must pass on recorded fixtures, so
   retrieval, grounding, and the abstain path are wired. It is not a quality score, so on its own it
   can never promote anything.
2. **Measured answer quality** (the gate). The decision uses the RAGAS aggregate from the real
   providers, written by `make ragas`. Staging needs at least 0.65, Production at least 0.78.
3. **Same-config check.** That score must have been measured on the exact config being promoted. A
   changed reranker (or LLM, or brain) makes the old number stale, and a stale number cannot promote.
4. **Champion check.** A Production candidate must also beat the frozen champion
   (`ragas_baseline.json`). With no champion committed yet, the first candidate is capped at Staging,
   so the first Production push is always a deliberate, compared-against-a-baseline decision.

The gate in action, straight from `make promote` on the committed code:

| candidate | change | measured RAGAS | gate decision |
|-----------|--------|----------------|---------------|
| current config | cohere embed-v4.0, rerank-v3.5, linear brain | 0.82 | **Staging** (clears the 0.78 bar, capped until a champion is frozen) |
| reranker swap | same number, reranker changed | stale | **REJECTED**, measured on a different config |
| fresh checkout | nothing measured yet | none | **REJECTED**, needs a real `make ragas` first |

The swap rejection is the whole point, a config change cannot inherit the old model's blessing:

```text
REJECTED: the RAGAS score was measured on a different config than the one being
promoted, so it is stale. Mismatches (measured -> current):
{'rerank_model': ('rerank-v3.5-swap', 'rerank-v3.5')}. Re-run `make ragas` first.
```

The RAGAS number in row one is a recorded score fed to the gate to show the decision; a live
Production promotion needs real providers (a `make ragas` run) plus a frozen champion, which are
deliberately not shipped on a hobby budget. What ships is the gate itself, and its refusals are
reproducible offline.

Reproduce or browse locally:

```bash
make promote                              # gate the current config, log an MLflow run
mlflow ui --backend-store-uri ./mlruns    # open http://localhost:5000
```

Point `MLFLOW_TRACKING_URI` at a remote server to log there instead; the code falls back to the
local `./mlruns` file store when no server is set, so it always records something.

## Drift detection

Four monitors compare a reference window of traffic to the current one, all from the trace store.
[drift-report.json](drift-report.json) is a real run over the demo's traffic:

| monitor | signal | value | drift |
|---------|--------|-------|-------|
| retrieval score | PSI of the top hit score | 0.49 | yes |
| confidence | PSI of the lexical-overlap gate | 2.29 | yes |
| query embedding | cosine distance between query centroids | 0.11 | yes |
| feedback rate | change in thumbs-down rate | n/a | no |

The retrieval-score and confidence monitors are also stratified by language. A monitor that trips
exits non-zero, so a scheduled job can alert. The point is not to page anyone, it is to catch the
questions people ask drifting away from what the corpus can answer before it shows up as bad answers.

```bash
make drift                   # prints the report, non-zero if anything drifted
```

That run also exercised the provider fallback: the primary Cohere key failed auth and the embedder
rolled to the backup key with no interruption (see [../fallbacks.md](../fallbacks.md)).

## The rest of the loop

- `make eval` scores retrieval and the abstain gate against the golden set.
- `make gate` runs the offline CI eval gate on recorded fixtures; it fails the build on a regression
  and needs no external services, so it runs on every push.
- `make ablation` writes [../eval-report.md](../eval-report.md), the dense vs hybrid vs
  hybrid+rerank comparison behind the retrieval design choice.
