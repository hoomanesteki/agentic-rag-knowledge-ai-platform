# The semantic layer

A semantic layer is the single place a business metric is defined, so an analyst, a dashboard, and
the AI agent all get the same number for the same question. Here that layer is each domain's
`metrics.yaml`.

## One definition, many readers

A metric in `domains/<domain>/metrics.yaml` is a named, governed object: a metric name, the
dimensions it can be sliced by, its typed parameters, and a validated SQL template over the gold
tables. For example, `return_rate_by_size` is defined once and every consumer reads that one
definition:

- the **RAG agent** fills the metric's slots through a validated call and cites the number as its
  own evidence, never writing free-form SQL;
- the **eval and RAGAS harness** score answers against the same governed numbers;
- the **backoffice dashboard** reads the same metrics for its quality and health views.

Because the definition lives in one file, "what is the return rate for size M" has exactly one
answer across the whole system. Change the definition once and every reader moves together.

## Governed, not free SQL

The AI never emits SQL. It selects a metric name and fills declared parameters; the engine binds
them as named parameters and runs the template on a read-only DuckDB connection with external
access off, rejecting anything that is not a single SELECT over gold (`data/metrics.py`). A missing
parameter binds to NULL, which the `(... is null or ...)` guard turns into "all values", so a
half-specified question still returns a sensible, safe result.

## Sitting on tested, documented models

The gold tables the metrics read are dbt models (`dbt/`, generated from the manifest). So the
semantic layer sits on a tested, lineage-traced foundation: schema tests, relationships tests, and
the PII-masking governance test all run on every `dbt build`, and dbt exposures name the assistant,
the dashboard, and the eval as the downstream consumers, so a change to a gold model shows its
blast radius. See [plan/stages/T1.1-dbt-medallion.md](plan/stages/T1.1-dbt-medallion.md).

## Why a metrics file and not dbt metrics

The metric definitions have to be swappable per domain and readable by the running app at request
time, so they live in the pack as data (`metrics.yaml`), not as dbt project code. dbt owns the
tables and their tests and lineage; the pack owns the business definitions on top. The two meet at
the gold tables, which are the same tables either way (a parity test proves the dbt build and the
in-app build produce identical gold).
