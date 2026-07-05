# ADR: service and container topology

## Status

Accepted. The request path stays one image, with one batch worker carved out. Full decomposition is
the documented scale-up, not current work.

## Context

The app is a modular monolith today: FastAPI in `api/`, an adapters layer that swaps real providers
for offline fakes, the agent brain in `rag/`, the linear pipeline in `pipeline/`, plus `evaluation/`,
`mlops/`, and `data/` in the same process. The stateful backing services already run as containers
in `docker-compose.yml`: Qdrant, Neo4j, Postgres, and the MLflow tracking server. The web app is a
separate Node runtime.

The open question is what stays in-process versus what becomes its own container. This is a
portfolio demo, so the goal is the smallest honest topology, not a microservice per module.

## Decision

- Keep the request path as one image. The retrieval, routing, and generation path is a tightly
  coupled decision loop; splitting it across the network would add latency and failure modes and
  break tracing into pieces, for no benefit at demo scale. In-process calls keep p95 low and
  produce one trace per turn.
- Keep the web app as its own image. It is a separate runtime and already deploys independently. It
  talks to the API over HTTP only.
- Carve out exactly one worker. The batch enrichment job (and the other one-shot scripts: ingest,
  graph build, lakehouse build) runs as its own container, reusing the app image with a different
  command. It is isolated because its lifecycle (scheduled or one-shot) and resource profile
  (embedding and annotation bursts) differ from serving, and it must never compete with request
  latency. It shares the codebase, so there is no adapter or schema drift.
- Communication is HTTP and in-process only. Inside the monolith, direct calls. Outward, HTTPS to
  the hosted model APIs and the stores. No internal message broker: there is one scheduled producer
  and no fan-out yet, so a broker would be infrastructure to run and secure for zero benefit.
- Compose profiles. The batch worker sits behind a `batch` profile, so `make up` brings up the
  stores and the tracking server but not the worker, which runs on demand:
  `docker compose --profile batch run --rm worker`.
- Health and config. The API keeps a liveness endpoint and a readiness endpoint that checks store
  reachability. The worker has no HTTP surface; its exit code is its health signal, backed by an
  MLflow run per job. Both load config from the environment; no secrets are baked into images.

## Alternatives considered

- Full microservices (separate retrieval, agent, embedding, eval services). Rejected: every hop
  adds latency, a failure mode, and versioning cost, and the brain is one loop. For one engineer at
  demo scale, in-process wins on speed and debuggability.
- Enrichment inside the API as a background task. Rejected: bursty jobs would steal CPU and memory
  from the request path and cannot be scheduled or scaled on their own.
- A message bus now. Rejected: premature. It is the first thing to add when enrichment needs
  retries, backpressure, or fan-out.

## Consequences

- One image to build, test, and deploy for the request path; the worker reuses it, so the surface
  stays tiny.
- The monolith scales as a unit, so a hot retrieval path scales the whole API. Acceptable at demo
  volume, and the first thing to split when it is not.

## Scale-up path, with triggers

- Enrichment needs retries or fan-out: put a queue (Redis Streams or a cloud pub/sub) between an
  enqueue endpoint and N worker replicas. This is where the deferred broker lands.
- Retrieval and generation need independent scaling, or the team passes about two engineers: split
  the retrieval and rerank hot path into its own service behind the API, keeping the brain in the
  API.
- Need autoscaling or zero-downtime rollout: move compose to Kubernetes. Deployments for the API
  and web, a CronJob for the enrichment worker, managed stores for Qdrant, Neo4j, and Postgres,
  secrets from the platform secret manager, and an HPA on the API. Only the store URLs in config
  change, because the adapters already abstract every store.

## Roles demonstrated

ML software engineer (service boundaries and the API surface), MLOps (containerization, health
probes, the scheduled worker, secrets), and Data Architect (the topology, store isolation, and the
queue-and-k8s boundary at scale).
