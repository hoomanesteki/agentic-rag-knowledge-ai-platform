# Deploy

How to put the public demo on hosted free tiers and keep it inside a small cost cap. Everything is
config: no engine code changes between local and deploy. The pieces:

| Piece            | Host                        | What it holds                                  |
| ---------------- | --------------------------- | ---------------------------------------------- |
| Web chat + admin | Vercel                      | the Next.js front end                          |
| API              | Cloud Run (min-instances 0) | FastAPI, the brain, auth                        |
| Vectors          | Qdrant Cloud (free)         | hybrid dense + sparse index                     |
| Users            | Supabase Postgres (free)    | the accounts table (same schema as local SQLite)|
| Knowledge graph  | Neo4j (HTTP-enabled)        | nodes and typed edges                           |
| Metrics          | DuckDB file in the image    | the gold medallion tables (read-only at serve)  |
| Tracking         | MLflow server (optional)    | run tracking, RAGAS, drift                       |

The graph and MLflow are optional: with `GRAPH_PROVIDER` unset the app degrades to no-graph
evidence (see `api/deps.py`), and MLflow is only read by the eval jobs.

## 1. Provision the stores

- **Qdrant Cloud**: create a free cluster, note its URL and API key.
- **Supabase**: create a project, take the Postgres connection string.
- **Neo4j**: the adapter speaks Neo4j's HTTP Cypher API (`POST /db/<db>/tx/commit`), not bolt, so
  use a deployment that exposes HTTP (self-managed Neo4j, or Aura with the Query API enabled). If
  you would rather skip the graph for the demo, leave `GRAPH_PROVIDER` unset and move on.

## 2. Build and deploy the API (Cloud Run)

```bash
gcloud run deploy skein-api \
  --source . \
  --region us-central1 \
  --min-instances 0 --max-instances 2 --concurrency 20 \
  --set-env-vars "$(grep -v '^#' .env.production | xargs | tr ' ' ',')"
```

`--source .` uses the `Dockerfile`. `min-instances 0` idles the service to zero (and to zero cost)
between demos; `max-instances 2` and `concurrency 20` cap the blast radius of a traffic spike.

Put the production values in `.env.production` (never commit it). The ones that matter:

```bash
SKEIN_ENV=production            # refuses to boot without a real JWT + Turnstile secret
JWT_SECRET=<a long random string>
TURNSTILE_SECRET_KEY=<cloudflare turnstile secret>
DEMO_READONLY=true              # disables the mutating admin endpoints on the public demo
RATE_LIMIT=30/minute
ALLOWED_ORIGINS=https://<your-vercel-app>.vercel.app
LLM_PROVIDER=groq
GROQ_API_KEY=<...>
EMBED_PROVIDER=voyage
RERANK_PROVIDER=voyage
VOYAGE_API_KEY=<...>
VECTOR_PROVIDER=qdrant
QDRANT_URL=<qdrant cloud url>
QDRANT_API_KEY=<...>
GRAPH_PROVIDER=neo4j            # or leave unset to run without the graph
NEO4J_URL=<neo4j http url>
NEO4J_USER=<...>
NEO4J_PASSWORD=<...>
ADMIN_PASSWORD=<change from the default>
```

`SKEIN_ENV=production` is the safety catch: the API refuses to start if `JWT_SECRET` is still the
insecure default or `TURNSTILE_SECRET_KEY` is empty, so you cannot accidentally expose a service
that serves forgeable tokens or a bypassed captcha.

## 3. Seed the stores

Build the lakehouse and indexes against the hosted stores (same targets as local, pointed at the
cloud URLs). Run once after provisioning and after any free-tier reset:

```bash
make lakehouse && make ingest && make graph-load    # DOMAIN=apparel_ecommerce (or saas_support)
```

## 4. Deploy the web app (Vercel)

Import the repo, set the root to `web/`, and set the environment variables:

```
NEXT_PUBLIC_API_URL=https://<your-cloud-run-url>
NEXT_PUBLIC_TURNSTILE_SITE_KEY=<cloudflare turnstile site key>
```

The Turnstile widget renders on both the customer and admin logins only when the site key is set.

## 5. Keep the free tiers warm

Neo4j Aura Free pauses in days; Supabase pauses in about a week; Qdrant free clusters suspend when
idle. Set these repo secrets (Settings -> Secrets and variables -> Actions) and the
`keepalive` workflow pings each every three days (targets with no secret are skipped):

```
KEEPALIVE_API_URL, QDRANT_URL, QDRANT_API_KEY, GRAPH_PROVIDER, NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD
```

Run it by hand any time with `make keepalive` (locally) or the workflow's "Run workflow" button.

## Cost cap checklist

- Cloud Run `min-instances 0` so idle cost is zero; `max-instances` + `concurrency` bound a spike.
- `DEMO_READONLY=true` so a leaked admin credential cannot trigger re-embedding (Voyage spend).
- `RATE_LIMIT` on chat and a tighter `5/minute` bucket on login, enforced per client.
- The request-body size limit (15 MB) and the 10 MB decoded-audio cap block oversized uploads.
- Rotate `ADMIN_PASSWORD` and `DEMO_PASSWORD` off their documented defaults before going public.
