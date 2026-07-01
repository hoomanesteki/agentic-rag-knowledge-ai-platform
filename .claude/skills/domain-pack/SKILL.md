---
name: domain-pack
description: Scaffold and validate a Skein Lite domain pack (domains/<name>/) so the same engine works on a new topic with no engine code change. Use when adding or editing a domain (for example lululemon or saas_support), when a new topic needs seed data, ontology, metrics, or a source manifest, or when checking that a pack meets the reproducibility contract before seeding.
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
  seed/
    structured/*.csv     # tabular seed, one file per source role
    unstructured/*.jsonl # text seed, one object per line
```

The single most important part is the `sources` manifest inside `domain.yaml`. It is what
lets the engine ingest an unknown schema without hardcoding table or column names. The
bronze load, the graph load, and the metrics all read roles from this manifest.

### domain.yaml required keys

- `name`: the domain slug, must match the folder name.
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
Params are validated and the query runs read only. No free SQL.

### ontology.cypher rules

Declare a constraint or index for every entity type so the graph load is idempotent. Use
`MERGE` friendly constraints (unique keys on the primary id of each label).

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
ontology, every metric source resolves to a manifest role, and a sample of the jsonl records
carry the fields the manifest promises. Exit code 0 means the pack meets the contract.

If PyYAML is not installed the script still checks file structure and prints how to install
it (`pip install pyyaml`) for the full check.

## Reproducibility guardrail

Before finishing, scan the change: no product name, metric name, or ontology label should
have leaked into engine folders (`adapters/`, `retrieval/`, `pipeline/`, `rag/`, `ingest/`).
If one did, move it into the pack. That leak is the thing that breaks topic swapping.
