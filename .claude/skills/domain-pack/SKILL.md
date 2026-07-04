---
name: domain-pack
description: Scaffold and validate a Skein Lite domain pack (domains/<name>/) so the same engine works on a new topic with no engine code change. Use when adding or editing a domain (for example apparel_ecommerce), when a new topic needs seed data, ontology, metrics, a source manifest, or a golden eval set, or when checking that a pack meets the reproducibility contract before seeding.
---

# Domain pack skill

A domain in Skein Lite is a folder, not code. The engine reads only what is inside
`domains/<name>/`. Switching topics means adding a folder and changing the `DOMAIN` env
var. This skill scaffolds a pack and checks it against the contract.

## The contract (what every pack must contain)

```
domains/<name>/
  domain.yaml            # identity, languages, glossary, entity types, and the sources manifest
  ontology.cypher        # node labels, relationship types, constraints and indexes
  metrics.yaml           # metric definitions that reference roles from the manifest
  prompts/               # optional node prompt overrides for this topic
  eval/
    golden.jsonl         # hand-written questions with known answers, the eval ground truth
  seed/
    structured/*.csv     # tabular seed, one file per source role
    unstructured/*.jsonl # text seed, one object per line
```

Use a fictional brand and clearly synthetic data. A public portfolio should not put a real
company's trademark on invented sales and reviews. Model the catalog on a real one if you
like, but the names are yours.

The single most important part is the `sources` manifest inside `domain.yaml`. It is what
lets the engine ingest an unknown schema without hardcoding table or column names. The
bronze load, the graph load, and the metrics all read roles from this manifest.

### domain.yaml required keys

- `name`: the domain slug, must match the folder name.
- `brand`: optional. The fictional brand name. The leak linter uses it to catch the brand
  leaking into engine code.
- `languages`: list of language codes the pack ships content in, for example `[en, fr]`.
- `glossary`: map of term to synonyms, used to normalize queries.
- `entity_types`: list of canonical node labels. Every one must appear in `ontology.cypher`.
- `sources`:
  - `structured`: list, each with `file`, `role`, `primary_key`, `columns` (name to type),
    optional `pii_columns`, optional `grain`.
  - `unstructured`: list, each with `file`, `doc_type`, `id_field`, `text_field`,
    `lang_field`, optional `meta_fields`, optional `entity_ref` fields.
- `defaults`: optional. Route hints, confidence thresholds, top_k values for this topic.

### metrics.yaml rules

Each metric has `name`, `grain`, `source` (a role declared in the manifest or a gold model
built from one), `dimensions`, and either `measures` or a parameterized `sql_template`.
A metric may never reference a table or column that is not traceable to a declared source.
Params are validated and the query runs read only. No free SQL. The engine enforces read
only at runtime (a read-only connection that rejects any non-SELECT), so the template is a
convenience, not the security boundary.

### ontology.cypher rules

Declare a constraint or index for every entity type so the graph load is idempotent. Use
`MERGE` friendly constraints (unique keys on the primary id of each label).

### eval/golden.jsonl (the ground truth)

One JSON object per line. Aim for about 20 questions spread across categories and languages,
so retrieval and answer quality can be measured instead of guessed. Fields:

- `id`: stable id.
- `lang`: language code, matching one in `languages`.
- `question`: the user question.
- `type`: `answerable`, `unanswerable` (in scope but not in the data, should abstain), or
  `out_of_domain` (should decline).
- `route`: expected route (`factual`, `relational`, `qualitative`, `analytical`), optional.
- `expected_answer_contains`: list of strings the answer should contain, for answerable ones.
- `expected_entities`: entity ids the answer should cite, optional.

Include the unanswerable and out-of-domain ones on purpose. Abstaining correctly is a quality
signal, and a pack that only has softball questions proves nothing.

## How to scaffold a new pack

1. Ask for the topic name and its two or three core structured sources and one or two text
   sources. Keep the first seed tiny (about 8 to 20 rows per table, 20 short text records).
2. Copy the files in `templates/` into `domains/<name>/` and fill them in for the topic.
   Real, plausible, clearly synthetic data. Do not use a real brand's proprietary data for
   a public demo.
3. Make sure `entity_types` in `domain.yaml` match the labels in `ontology.cypher`, and that
   every metric `source` matches a manifest role.
4. Run the validator (below). Fix everything it reports before seeding.

## How to validate a pack

```
python .claude/skills/domain-pack/scripts/validate_domain_pack.py domains/<name>
```

It checks: required files exist, `domain.yaml` has the required keys, every structured and
unstructured seed file referenced in the manifest exists, entity types are declared in the
ontology, every metric source resolves to a manifest role, a sample of the jsonl records carry
the fields the manifest promises, and the golden set exists with the required fields. Exit
code 0 means the pack meets the contract.

## Reproducibility guardrail

Two layers keep the engine domain agnostic:

1. Automated: `make check` runs `scripts/check_domain_leak.py`, which loads each pack's
   vocabulary (brand, product names, metric names, glossary terms) and fails if any of it
   appears in engine folders (`adapters/`, `retrieval/`, `pipeline/`, `rag/`, `ingest/`,
   `api/`, `data/`, `mlops/`). This runs on every commit.
2. Manual: when you add a pack, still eyeball the diff. If a product name, metric, brand, or
   ontology label ended up in engine code, move it into the pack. That leak is the thing that
   breaks topic swapping.
