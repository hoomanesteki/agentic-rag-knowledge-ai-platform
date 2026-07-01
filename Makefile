.DEFAULT_GOAL := help
DOMAIN ?= lululemon

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

setup: ## Install Python dependencies into the active environment
	pip install -r requirements.txt -r requirements-dev.txt

test: ## Run tests
	pytest

lint: ## Lint with ruff
	ruff check .

validate: ## Validate one domain pack, e.g. make validate DOMAIN=lululemon
	python .claude/skills/domain-pack/scripts/validate_domain_pack.py domains/$(DOMAIN)

validate-all: ## Validate every domain pack under domains/ (no-op if none yet)
	@found=0; \
	for d in domains/*/; do \
	  [ -d "$$d" ] || continue; \
	  found=1; echo "validating $${d%/}"; \
	  python .claude/skills/domain-pack/scripts/validate_domain_pack.py "$${d%/}" || exit 1; \
	done; \
	if [ $$found -eq 0 ]; then echo "no domain packs yet, skipping"; fi

check: lint test validate-all ## Run every check that CI runs
	@echo "all checks passed"

.PHONY: help setup test lint validate validate-all check
