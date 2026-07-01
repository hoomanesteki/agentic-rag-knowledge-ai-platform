# agentic-rag-knowledge-ai-platform

Skein is an agentic AI platform that answers questions by weaving together structured and
unstructured data, grounding every response in a knowledge graph, hybrid vector search, and
a governed semantic layer. It escalates to humans when unsure, and runs locally or scales to
the cloud.

## How this is built

We build in small, demoable slices, MVP first. The plan lives in
[docs/BUILD-PLAN.md](docs/BUILD-PLAN.md). Each step there is one focused work session with a
runnable check for when it is done.

The topic is a config folder, not code. A domain (for example lululemon or saas support) is a
pack under `domains/<name>/`. To scaffold or check one, use the `domain-pack` skill in
`.claude/skills/domain-pack/`. Switching topics means adding a folder and changing `DOMAIN`.

Models are hosted so nothing heavy runs on your laptop: Groq for the LLM, Voyage for
embeddings and reranking. Qdrant and Postgres run in Docker locally.

## Quick start (dev)

```bash
python3 -m venv .venv && source .venv/bin/activate
make setup        # install dependencies
cp .env.example .env   # then fill in your API keys
make test         # run the smoke tests
```

Then follow [docs/BUILD-PLAN.md](docs/BUILD-PLAN.md) from milestone M0.

## Development workflow

Work happens on short-lived branches, roughly one milestone step per branch, and merges to
`main` only when green.

1. Branch: `git checkout -b build/<step>` (for example `build/m1-first-answer`).
2. Build the step from [docs/BUILD-PLAN.md](docs/BUILD-PLAN.md).
3. Check: `make check` runs ruff lint, the tests, and domain-pack validation. This is the
   same gate CI runs. The `preflight` skill runs it and reports go or no-go.
4. Open a pull request. CI (`.github/workflows/ci.yml`) runs on every PR and on `main`.
   Merge once it is green.
