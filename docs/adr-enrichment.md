# ADR: batch enrichment of reviews into governed product features

## Status

Accepted, with a working slice built. The full LLM-annotator path and the description path are
described here as the scale-up.

## Context

The catalog has two kinds of unstructured text with very different lifecycles, and treating them
the same would be a mistake.

Product descriptions are authored, stable, trusted, and low-churn. Reviews are user generated,
high-churn, and untrusted: a person can write anything in a review, including spam, off-topic text,
and injection attempts. A single review is not a fact. The aggregate of many reviews can be.

We want the signal reviews carry ("this runs small", "the fabric is thin") available to the
assistant as structured product features, without ever trusting one review and without paying to
re-read raw reviews at query time.

## Decision

Compute review-derived features in a batch job, by consensus, and serve them from a feature table.

- Two paths by lifecycle. Descriptions are enriched once at ingest and re-run only when the text
  changes, keyed on a content hash, so unchanged descriptions cost nothing on a re-run. Reviews are
  enriched in the periodic batch job described here. The slice built now is the reviews path.
- Consensus, not trust. `data/enrichment.py` annotates each review for an aspect (fit: runs_small,
  runs_large, true_to_size) and votes per product. A value becomes a feature only when at least
  N reviews agree and they are a majority of the reviews that spoke to that aspect. Each feature
  carries a confidence, its support count, and the source review ids, so it is auditable and can be
  recomputed. On the seed reviews this yields 486 reviews to 11 governed features (for example
  product P041 gets fit=runs_small at confidence 1.0 from three agreeing reviews). Most reviews
  contribute nothing, which is correct: only agreed signal is served.
- A pluggable annotator. The default is a deterministic keyword pass, which is cheap and
  reproducible and good enough to prove the pipeline. An LLM annotator (or several LLM annotations
  voting per review) plugs into the same `annotate` slot with no change to the consensus logic.
- A feature store on the lakehouse. Features land in the DuckDB `product_features` table with
  provenance. The write is idempotent (this annotator's rows are replaced), so re-running converges
  instead of duplicating. The app reads a precomputed feature, not raw reviews.
- Live is only the cheap safe thing. A moderation, PII, and injection gate at review-submit time is
  a separate, small check. It is not this job. This job is batch.

## Alternatives considered

- Trust and serve each review directly. Rejected: one review is noise, and serving user text
  directly is a safety and quality risk.
- Annotate live on every review write. Rejected: bursty embedding and annotation work would steal
  from request latency, and a single new review must not flip a product's feature. The aggregate
  signal changes slowly even though reviews arrive continuously.
- Recompute features at query time. Rejected: it repeats the heavy work on every request. Precompute
  once in batch, serve many times.

## Consequences

- The assistant can cite a governed, confidence-scored feature ("reviewers say this runs small")
  instead of guessing or quoting one review.
- It gets cheaper and faster at scale, not slower: content-hash skipping avoids re-enriching
  unchanged descriptions, only new reviews are annotated, and serving reads a precomputed table.
- Trade-off: a feature lags the newest reviews by one batch interval. That is the point; consensus
  needs a window.

## Why batch, and how often

Reviews arrive continuously but their aggregate signal is stable, so a daily or weekly batch is
enough. Batch also makes the job idempotent, backfillable, and easy to test and govern, and drift
detection is naturally a this-window-versus-last comparison. The cadence is a config choice, not a
code change.

## Deployment

The batch job runs as its own container, the one service carved out of the modular monolith. It
reuses the app image with a different command and runs on demand (`docker compose --profile batch
run --rm worker`), never as a long-lived service, so it cannot compete with request latency. At
scale it becomes a Kubernetes CronJob, and if throughput outgrows one shot, a queue feeds N worker
replicas. See the services ADR for the topology.

## Roles demonstrated

Data Architect (the lifecycle split, the feature store, the provenance), Data Scientist (the
consensus rule and the confidence), and MLOps (the idempotent, reproducible, containerized batch
job).
