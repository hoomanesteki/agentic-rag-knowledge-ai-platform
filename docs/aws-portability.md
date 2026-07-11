# AWS portability

Where this runs in AWS, mapped component by component behind the seams the app already has. Nothing
here is built or coupled in the code: the point is that local-first is a deliberate design with a
priced exit, not a limitation. Every hosted dependency is already reached through an adapter
(`adapters/factory.py`) or a provider env var, so moving to AWS is configuration and infrastructure,
not a rewrite. Groq (LLM) and Cohere (embeddings, rerank) stay as plain HTTPS calls from any VPC.

Each row is badged:

- **portable today**: runs on AWS with only config or a container, no code change.
- **needs a seam**: one adapter or a small change, sized below.
- **not built**: deliberately out of scope for the demo; noted so the gap is honest.

## Component map

| Component | Local today | On AWS | Badge |
| --- | --- | --- | --- |
| Streaming chat API (FastAPI, SSE) | uvicorn | **ECS Fargate** behind an ALB (SSE needs a long idle timeout; Lambda response-streaming is the caveated serverless alternative) | portable today |
| LLM (Groq) | HTTPS to Groq | same HTTPS call from the VPC (or swap the adapter for Bedrock if a customer requires in-account inference) | portable today |
| Embeddings + rerank (Cohere) | HTTPS to Cohere | same HTTPS call (or Bedrock/SageMaker embeddings behind the embedder seam) | portable today |
| Vector store (Qdrant) | Docker Qdrant | **Qdrant on ECS/EKS** zero code; **Aurora PostgreSQL + pgvector** as the managed option | needs a seam |
| Knowledge graph (Neo4j) | Docker Neo4j | **Neo4j on ECS** as-is; **Neptune** is NOT drop-in (no APOC / openCypher parity), so Neo4j-on-ECS or run graphless | needs a seam |
| Lakehouse (DuckDB + dbt) | local files | build in CI, publish gold **Parquet to S3**, query with Athena or keep DuckDB on the task | portable today |
| MLflow (tracking, registry) | `./mlruns` or a container | **RDS Postgres** backend + **S3** artifact store; the four experiments (`skein-requests/drift/ct/shadow`) log unchanged | needs a seam |
| Model/prompt registry + candidates | JSON under `evaluation/reports/` | **S3 versioned objects** (native object versioning is the audit trail) | needs a seam |
| Traces / feedback logs | JSONL files | stream to **S3** (or Kinesis Firehose); the drift and monitoring readers point at the same records | needs a seam |
| Review queue + auth | SQLite | **DynamoDB** or one small **RDS Postgres** (which can double as the MLflow backend) | needs a seam |
| Weekly CT (`ct.py`) | `make ct` / GitHub Actions cron | **EventBridge Scheduler** → a scheduled **Fargate task**; Step Functions only if CT grows branching | portable today |
| Monitoring pillars + drift NOTIFY | JSON reports + a GitHub issue | **CloudWatch** metrics/alarms + **SNS** (email/mobile); the GitHub-issue notify is the current, lighter equivalent | needs a seam |
| Promotion gate | `make registry-promote` (human) | **GitHub protected environment** with required reviewers, or **CodePipeline manual-approval** action | portable today |
| Voice STT/TTS (Groq Whisper / ElevenLabs) | HTTPS | same, or **Amazon Transcribe / Polly** behind the existing voice seams | portable today |
| Web client | Vercel / static | **S3 + CloudFront** | portable today |

## The seams, sized

The "needs a seam" rows are all small and localized because the coupling was avoided up front:

- **Vector to Aurora pgvector (M):** pgvector 0.8.0 on Aurora handles the dense leg; the app also uses
  a sparse BM25 leg, so a managed swap needs a sparse adapter (OpenSearch for hybrid, or keep the
  in-process sparse encoder against pgvector). Qdrant-on-ECS is the zero-code path.
- **File state to durable stores (S/M):** traces, the review queue, auth, and the registry are the
  only local files. Each is read/written through one module, so each is one adapter: S3 for
  append-only logs and the registry, DynamoDB or RDS for the queue and auth.
- **MLflow backend (S):** set `MLFLOW_TRACKING_URI` to the RDS-backed server and the artifact store
  to an S3 bucket; the logging code is unchanged.

## What is deliberately not built

- No always-on managed inference: the demo calls Groq/Cohere over HTTPS by design (cheap, fast). A
  customer who needs in-account inference swaps the LLM/embedder adapter to Bedrock or SageMaker.
- No IaC in this repo: the mapping above is the design, not a CloudFormation/Terraform module. That
  is the natural next artifact for a real deployment.
- No live canary: shadow replay is offline over historical traffic; a CloudWatch-monitored canary on
  a traffic slice is the documented scale-up (see `mlops/shadow.py`).

## Why it maps cleanly

The architecture is deterministic code with LLM calls at decision points, hosted dependencies behind
adapters, and state in a handful of clearly-owned files. That is the shape AWS-managed services drop
into: stateless compute on Fargate, state in RDS/DynamoDB/S3, schedules on EventBridge, and
observability on CloudWatch. The local-first stack is the same system with the managed pieces
swapped for containers, which is exactly what "runs in your AWS account" should mean.
