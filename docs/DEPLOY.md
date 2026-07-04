# Deploy

How to put the public demo on hosted free tiers and keep it inside a small cost cap. Everything is
config: no engine code changes between local and deploy. The pieces:

| Piece            | Host                        | What it holds                                   |
| ---------------- | --------------------------- | ----------------------------------------------- |
| Web chat + admin | Vercel                      | the Next.js front end                           |
| API              | Cloud Run (min-instances 0) | FastAPI, the brain, auth                        |
| Vectors          | Qdrant Cloud (free)         | hybrid dense + sparse index                     |
| Knowledge graph  | Neo4j (HTTP-enabled)        | nodes and typed edges                           |
| Users            | SQLite in the container     | accounts, re-seeded each cold start (ephemeral) |
| Metrics          | DuckDB gold tables          | governed numbers, only if baked into the image  |
| Tracking         | MLflow + Postgres (optional)| run tracking, RAGAS, drift (off the serve path) |

What is and is not wired matters here, so it is stated plainly:

- **Users are SQLite, not Postgres.** `api/auth.py` uses a local SQLite file (`AUTH_DB_PATH`); no
  code reads a Postgres URL for auth. On Cloud Run that file lives in the container's ephemeral
  disk and is re-seeded from the demo/admin credentials on every cold start. That is fine for a
  read-only demo: accounts are fixed, and there is no per-user state worth persisting.
- **The graph and MLflow are optional.** With `GRAPH_PROVIDER` unset the app degrades to no-graph
  evidence (`api/deps.py`). MLflow is only read by the eval jobs, never by the serving path; if you
  run a hosted MLflow server, a free Supabase/Postgres is a fine backing store for it.
- **Metric answers need the lakehouse baked in.** The DuckDB gold tables are a local build artifact.
  A default `docker build` / `gcloud --source .` does not include them (see step 3), so metric
  answers are off unless you bake them in.

## 1. Provision the stores

- **Qdrant Cloud**: create a free cluster, note its URL and API key.
- **Neo4j**: the adapter speaks Neo4j's HTTP transaction endpoint (`POST /db/<db>/tx/commit`), not
  bolt and not Aura's newer Query API. Use a deployment that exposes that endpoint (self-managed
  Neo4j 5 with HTTP enabled, the same image as `docker-compose.yml`). Aura Free is bolt-only, so it
  will not work without adapter changes; to skip the graph for the demo, leave `GRAPH_PROVIDER`
  unset and the app runs without it.

## 2. Build and deploy the API (Cloud Run)

Put the production values in `.env.yaml` (a flat `KEY: value` map). The `.env.` prefix matters: it
is what `.gitignore` (`.env.*`) and `.dockerignore` (`**/.env.*`) exclude, so the secrets file
cannot be committed or baked into the image. Using a file also avoids the shell-quoting traps of
`--set-env-vars` when a value contains a space (JWT_SECRET) or a comma (ALLOWED_ORIGINS).

```yaml
# .env.yaml  (gitignored and dockerignored; never commit it)
SKEIN_ENV: production            # refuses to boot without a real JWT + Turnstile secret + non-default creds
JWT_SECRET: <a random string of at least 32 characters>
TURNSTILE_SECRET_KEY: <cloudflare turnstile secret>
ADMIN_PASSWORD: <change from the default>
DEMO_PASSWORD: <change from the default>
DEMO_READONLY: "true"            # disables the mutating admin endpoints on the public demo
RATE_LIMIT: 30/minute
ALLOWED_ORIGINS: https://<your-vercel-app>.vercel.app
LLM_PROVIDER: groq
GROQ_API_KEY: <...>
EMBED_PROVIDER: cohere
RERANK_PROVIDER: cohere
COHERE_API_KEY: <...>
VECTOR_PROVIDER: qdrant
QDRANT_URL: <qdrant cloud url>
QDRANT_API_KEY: <...>
GRAPH_PROVIDER: neo4j            # or omit to run without the graph
NEO4J_URL: <neo4j http url>
NEO4J_USER: <...>
NEO4J_PASSWORD: <...>
```

```bash
gcloud run deploy skein-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --min-instances 0 --max-instances 2 --concurrency 20 \
  --env-vars-file .env.yaml
```

`--source .` uses the `Dockerfile`. `--allow-unauthenticated` is required or the public demo gets a
403 from IAM. `min-instances 0` idles the service (and its cost) to zero between demos;
`max-instances 2` and `concurrency 20` bound the blast radius of a spike.

`SKEIN_ENV=production` is the safety catch: the API refuses to start if `JWT_SECRET` is a placeholder
or under 32 characters, if `TURNSTILE_SECRET_KEY` is empty, or if `ADMIN_PASSWORD`/`DEMO_PASSWORD`
are still the documented defaults, so you cannot expose forgeable tokens, a bypassed captcha, or
known credentials.

## 3. Seed the stores

Build the vector index and graph against the hosted stores (same targets as local, pointed at the
cloud URLs):

```bash
make ingest && make graph-load     # DOMAIN=apparel_ecommerce
```

To also serve governed **metric** answers, the DuckDB gold tables must be inside the image, because
DuckDB is an embedded file with no hosted target. The `.dockerignore` already allows it in; the one
extra step is telling `gcloud --source .` to upload it (it derives its ignore list from
`.gitignore`, which excludes `lakehouse.duckdb`). Create a `.gcloudignore` that keeps the
`.gitignore` defaults and un-ignores just the lakehouse:

```bash
make lakehouse                     # writes lakehouse.duckdb (synthetic seed data only)
cat > .gcloudignore <<'EOF'
#!include:.gitignore
!lakehouse.duckdb
EOF
```

The `#!include:.gitignore` line preserves gcloud's default exclusions (so `.env`, `.venv`, and the
local SQLite files are still kept out of the upload) while letting the lakehouse through. Without
this the demo runs chat + graph and logs that metric answers are disabled (`api/deps.py`).

## 4. Deploy the web app (Vercel)

Import the repo, set the root to `web/`, and set the environment variables:

```text
NEXT_PUBLIC_API_URL=https://<your-cloud-run-url>
NEXT_PUBLIC_TURNSTILE_SITE_KEY=<cloudflare turnstile site key>
```

The Turnstile widget renders on both the customer and admin logins only when the site key is set.

## 5. Keep the free tiers warm

Neo4j Aura Free pauses in days; Qdrant free clusters suspend when idle. Set these repo secrets
(Settings -> Secrets and variables -> Actions) and the `keepalive` workflow pings each every three
days (targets with no secret are skipped):

```text
KEEPALIVE_API_URL, QDRANT_URL, QDRANT_API_KEY, GRAPH_PROVIDER, NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD
```

Run it by hand any time with `make keepalive` (locally) or the workflow's "Run workflow" button.

## Ephemeral state on Cloud Run

The users database (`.auth.db`), the review queue (`.review_queue.db`), and `traces/` live on the
container's local disk and are wiped when the service scales to zero. Accounts are re-seeded on the
next cold start, so login keeps working, but the admin dashboards (quality, gaps, review queue)
start empty after each idle period. That is acceptable for a read-only demo; persisting them means
moving the queue and traces to a hosted store, which is out of scope here.

## Cost cap checklist

- Cloud Run `min-instances 0` so idle cost is zero; `max-instances` + `concurrency` bound a spike.
- `DEMO_READONLY=true` so a leaked admin credential cannot trigger re-embedding (Voyage spend).
- Rate limiting is best-effort per client (keyed on `X-Forwarded-For` in production), with a tighter
  `5/minute` bucket on login. The hard ceiling is the Cloud Run instance cap, not the per-client key.
- The request-body size limit (15 MB) and the 10 MB decoded-audio cap block oversized uploads.
- `SKEIN_ENV=production` enforces non-default `ADMIN_PASSWORD`/`DEMO_PASSWORD` at boot.
