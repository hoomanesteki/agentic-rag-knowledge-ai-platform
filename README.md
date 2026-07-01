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
