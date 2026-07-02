.DEFAULT_GOAL := help
DOMAIN ?= apparel_ecommerce
export q  # pass the ask question through the environment, not the shell command line

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

setup: ## Create the virtualenv and install dependencies (uv, reads .python-version)
	uv sync --extra dev

test: ## Run tests
	uv run pytest

lint: ## Lint with ruff
	uv run ruff check .

validate: ## Validate one domain pack, e.g. make validate DOMAIN=apparel_ecommerce
	uv run python .claude/skills/domain-pack/scripts/validate_domain_pack.py domains/$(DOMAIN)

validate-all: ## Validate every domain pack under domains/ (no-op if none yet)
	@found=0; \
	for d in domains/*/; do \
	  [ -d "$$d" ] || continue; \
	  found=1; echo "validating $${d%/}"; \
	  uv run python .claude/skills/domain-pack/scripts/validate_domain_pack.py "$${d%/}" || exit 1; \
	done; \
	if [ $$found -eq 0 ]; then echo "no domain packs yet, skipping"; fi

leak-check: ## Fail if a domain's vocabulary leaked into engine folders
	uv run python scripts/check_domain_leak.py

check: lint test validate-all leak-check ## Run every check that CI runs
	@echo "all checks passed"

up: ## Start local infrastructure (qdrant, postgres)
	docker compose up -d --wait

down: ## Stop local infrastructure (data volumes are kept)
	docker compose down

ps: ## Show infrastructure status
	docker compose ps

lakehouse: ## Build the active DOMAIN's DuckDB medallion lakehouse and run data contracts
	PYTHONPATH=. uv run python scripts/build_lakehouse.py

graph-load: ## Load the active DOMAIN's knowledge graph from gold into Neo4j (needs make up)
	PYTHONPATH=. uv run python scripts/build_graph.py

ingest: ## Ingest the active DOMAIN into Qdrant (needs keys in .env and make up)
	PYTHONPATH=. uv run python scripts/run_ingest.py

ask: ## Ask a question, e.g. make ask q="What do customers say about sizing?"
	PYTHONPATH=. uv run python scripts/ask.py

eval: ## Score retrieval and the abstain gate against the domain golden set
	PYTHONPATH=. uv run python scripts/run_eval.py

ablation: ## Write docs/eval-report.md comparing dense vs hybrid vs hybrid+rerank
	PYTHONPATH=. uv run python scripts/run_ablation.py

mlflow-log: ## Log the request traces to MLflow (./mlruns locally, or MLFLOW_TRACKING_URI)
	PYTHONPATH=. uv run python scripts/log_mlflow.py

serve: ## Run the API locally on :8000 (needs keys, make up, and an ingest for real answers)
	PYTHONPATH=. uv run uvicorn api.app:app --reload --port 8000

.PHONY: help setup test lint validate validate-all leak-check check up down ps lakehouse graph-load ingest ask eval ablation mlflow-log serve
