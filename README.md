# agentic-rag-knowledge-ai-platform

Skein is an agentic AI platform that answers questions by weaving together structured and
unstructured data, grounding every response in a knowledge graph, hybrid vector search, and
a governed semantic layer. It escalates to humans when unsure. The infrastructure runs on
your laptop in Docker, the models are hosted APIs, and the whole thing ports to the cloud by
swapping adapters.

## How this is built

We build in small, demoable slices, MVP first. The plan lives in
[docs/BUILD-PLAN.md](docs/BUILD-PLAN.md). Each step there is one focused work session with a
runnable check for when it is done.

The topic is a config folder, not code. A domain (for example apparel ecommerce or saas
support) is a pack under `domains/<name>/`. To scaffold or check one, use the `domain-pack`
skill in
`.claude/skills/domain-pack/`. Switching topics means adding a folder and changing `DOMAIN`.

Models are hosted so nothing heavy runs on your laptop: Groq for the LLM, Voyage for
embeddings and reranking. Qdrant and Postgres run in Docker locally.

## Quick start (dev)

Needs [uv](https://docs.astral.sh/uv/) (it manages Python 3.12 for you) and Docker.

```bash
make setup            # create the venv and install locked dependencies
cp .env.example .env  # then fill in your API keys
make check            # lint, tests, domain validation, and the leak check
```

Then follow [docs/BUILD-PLAN.md](docs/BUILD-PLAN.md) from milestone M0.

### API and web

```bash
make serve                       # API on http://localhost:8000 (needs make up + make ingest)
cd web && npm install && npm run dev   # web chat on http://localhost:3000
```

The web app reads the API URL from `NEXT_PUBLIC_API_URL` and the API allows the web origin via
`ALLOWED_ORIGINS` (see `.env.example`).

Sign in with the seeded demo account (override via `DEMO_USERNAME` / `DEMO_PASSWORD`):

```
username: demo
password: skein-demo-2026
```

The captcha is bypassed when `TURNSTILE_SECRET_KEY` is empty (dev). Set it plus
`NEXT_PUBLIC_TURNSTILE_SITE_KEY` (in `web/.env.local`) to enable Turnstile.

## Development workflow

Work happens on short-lived branches, roughly one milestone step per branch, and merges to
`main` only when green.

1. Branch: `git checkout -b build/<step>` (for example `build/m1-first-answer`).
2. Build the step from [docs/BUILD-PLAN.md](docs/BUILD-PLAN.md).
3. Check: `make check` runs ruff lint, the tests, domain-pack validation, and the leak
   check. This is the same gate CI runs. The `preflight` skill runs it and reports go or
   no-go.
4. Open a pull request. CI (`.github/workflows/ci.yml`) runs on every PR and on `main`.
   Merge once it is green.
