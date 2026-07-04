# MLOps: experiments and drift

A snapshot of what the pipeline tracks, so the loop is visible without spinning up the stack. All of
it is generated from the request traces and evals the app already writes, with `mlflow-skinny` for
tracking and PSI-style monitors for drift. Regenerate any of it with the `make` targets below.

## Experiment tracking (MLflow)

Every candidate config is scored and promoted through stages (`dev -> Staging -> Production`) by its
eval numbers, and each decision is logged as an MLflow run. Two runs from the local store:

| run | embed | rerank | brain | gate | ragas | smoke | stage |
|-----|-------|--------|-------|------|-------|-------|-------|
| big-deer-206 | cohere embed-v4.0 | cohere rerank-v3.5 | linear | 0.95 | 0.95 | 1.0 | Production |
| skittish-whale-911 | cohere embed-v4.0 | rerank-vDIFFERENT | linear | 0.95 | 0.95 | 1.0 | Production |

The second run changes only the reranker model. The promotion guard treats a config change as a new
candidate that must re-earn its stage, so a swap cannot inherit the old model's blessing.

View the runs locally:

```bash
make promote                 # scores the current config and logs a run
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
