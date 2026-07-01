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

validate: ## Validate a domain pack, e.g. make validate DOMAIN=lululemon
	python .claude/skills/domain-pack/scripts/validate_domain_pack.py domains/$(DOMAIN)

.PHONY: help setup test lint validate
