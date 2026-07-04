# Showcase roadmap

A staged plan to take the platform from "works" to "best-in-class, and legible to a reviewer in
ten minutes." Each stage is a self-contained, committable increment. Ordered by impact-per-effort
for a hiring-manager demo, not by layer.

The guiding principle: most of the hard machinery already exists (medallion ETL, a LangGraph
agentic brain, MLflow + drift, a RAGAS/golden eval suite with a CI gate, resilient fallbacks). The
work is to **deepen the thin parts, close the real gaps, and make the invisible sophistication
visible** through content, UX, dashboards, and honest documentation.

---

## Stage 0, Baseline audit (done as part of this plan)

What already exists, so we deepen instead of rebuild:

| Area | Exists | File(s) |
| --- | --- | --- |
| Medallion ETL (bronze/silver) | Yes | `dbt/models/staging`, `dbt/models/silver` |
| Semantic / metrics layer | Yes | `docs/semantic-layer.md`, metrics governance |
| Agentic brain (LangGraph) | Yes | `rag/{agent,supervisor,graph,specialists,state}.py` |
| Human-in-the-loop + flywheel | Yes | `rag/hitl.py`, `rag/flywheel.py` |
| Eval: RAGAS, golden, CI gate, monitoring | Yes | `evaluation/*` |
| Drift monitoring | Yes | `mlops/drift.py` |
| MLflow tracking | Yes | `mlops/mlflow_sink.py` |
| Resilience / fallbacks | Yes | `api/resilience.py` |
| Back office dashboards | Yes | `web/app/admin/*` |
| CI/CD | Yes | `.github/workflows/ci.yml` |
| Data-architecture notebook | Partial (1) | `notebooks/01-data-architecture.ipynb` |

Gaps to close are listed per stage below.

---

## Stage 1, Assistant intelligence + chat UX (highest visible impact)

The demo lives or dies here. A reviewer types real shopper questions and reads the answers.

- **Content depth across dimensions** shoppers actually ask from: 2026 seasonal trends and colors;
  influencer / celebrity / "off-duty" style; medical and skin needs (eczema, sensitive skin,
  sensory-friendly, chafe-free); occasions (anniversary, Black Friday, back-to-school); gift
  categories; care and materials. Fixes reworded abstains and makes it read like an expert.
- **Scannable answers**: recommend in short bullets, each product named with a one-line *reason*,
  and rendered as clickable product cards (image + link), never a wall of text.
- **Restock + per-size availability**: a restock policy line, and size-level stock lookups.
- **Gift-specific categories/products** surfaced (gift cards, bundles, accessories).

## Stage 2, Data platform: lineage, gold layer, governance

- Confirm/complete the **gold** marts (facts + dims) on top of silver; document grain.
- **dbt tests** (not-null, unique, relationships, accepted values) + a PII-masking test.
- **Data lineage** rendered (dbt docs / a lineage diagram) and linked from the back office.
- **Governance**: PII columns declared, masking in silver, retention notes, the leak-linter that
  keeps engine code domain-agnostic.

## Stage 3, Eval, drift, MLOps, and the "why this model" story

- RAGAS (faithfulness, answer-relevancy, context precision/recall) + a **golden set** the CI gate
  blocks on; add adversarial / jailbreak / multi-lingual edge cases to the golden set.
- **Model-selection rationale** (`docs/model-selection.md`) deepened with a measured comparison
  table (latency, cost/1k, grounding) so "we chose Groq Llama 3.3 70B because …" is evidenced.
- **MLflow**: log every eval run; a registry with dev → staging → prod stages and a promotion gate.
- **Drift**: per-language input/embedding/answer-quality drift with thresholds and an alert path;
  surfaced on a dashboard.

## Stage 4, Back office + BI for two audiences

- **Business dashboard**: funnel, top searches/questions, revenue proxies, and **unmet-demand
  insights** (products/queries people ask for that we don't carry or can't answer), seeded with
  thousands of realistic apparel searches so it looks like a live store.
- **Technical dashboard**: p95 latency, cost/turn, grounding, abstain/escalation, drift trend.
- Every metric has a hover explanation ("why this matters"); clearly labeled per audience.

## Stage 5, Agentic workflow, loops, fallbacks, scale

- A **diagram + doc** of the LangGraph loop (understand → retrieve → govern → answer → verify →
  escalate → flywheel) so the agentic design is visible.
- **Loops with cost caps**: the feedback flywheel (re-index verified answers), a retrain/refresh
  trigger, and capped retries (e.g., 3 attempts) on transient failures.
- **Fallback chain**, documented and tested: reranker → cache → skip; LLM → smaller model →
  degraded; ElevenLabs voice → browser voice. Graceful degradation everywhere.

## Stage 6, CI/CD, skills, AI safety

- CI runs lint, tests, schema/validate, leak-check, and the **eval gate** on every PR; add a
  matrix / edge-case job.
- A **PR-review skill**: on a PR, the assistant reviews the diff, posts suggestions as review
  comments, and the human approves, reviewer-in-the-loop, not auto-merge.
- **AI-safety harness**: a test suite of harmful / jailbreak / PII-extraction / prompt-injection
  cases the assistant must refuse; run in CI.

## Stage 7, Documentation and showcase

- **README**: visual, step-by-step, "what / why / how it runs," with the architecture diagram.
- **Notebooks** for each stage (data architecture, evaluation, drift, model selection) that render
  real outputs and read as a lab report.
- A **decision log** (`docs/`) capturing why each tool was chosen and what was traded off.
- Human-authored voice throughout; no AI-assistant attribution.

## Stage 8, independent end-to-end verification

- Adversarial review of the whole thing, word by word, by an independent model: correctness, UX,
  safety, edge cases, categorized test coverage, and best-practice gaps versus the apparel-ecommerce
  industry.

---

### Working agreement
Ship one stage at a time, each green through `make check` (lint, tests, validate, leak-check,
eval gate) and committed on its own. This document is the running plan; it is updated as stages
complete.

---

## Progress log

- **Stage 0** done, baseline audit (table above).
- **Stage 1** done, assistant content depth (medical/eczema, 2026 trends, occasions, gifts, care),
  scannable bulleted answers with a reason per product, restock/size/price policies.
- **Stage 2** done, governed `stock_by_size` metric closes the per-size stock gap; the rest of the
  medallion stack (bronze/silver/gold, schema tests, PII masking, exposures, semantic layer) was
  already in place.
- **Stage 3** done, model-selection rationale corrected (Voyage → Cohere) and evidenced against the
  live Health view / RAGAS gate / ablation; the MLflow staging→prod promotion gate landed
  (`scripts/promote_model.py`, `make promote`, evidence in `docs/mlops/experiments.md`).
- **Stage 4** done, back-office BI insights with per-metric business and technical hints.
- **Stage 5** done, agentic-loop write-up and the layered provider fallback chains
  (`docs/fallbacks.md`).
- **Stage 6** done, AI-safety test harness (injection, harm decline, PII gate; caught a real
  first-person-gate leak) and a `pr-review` skill (reviewer suggests on the PR, human approves); the
  offline CI eval gate blocks regressions on every push.
- **Stage 7** done, README, notebooks, and decision log rewritten to render real outputs.
- **Stage 8** done, independent end-to-end review across rounds with adversarial verification;
  confirmed findings fixed and the docs reconciled with the code.
