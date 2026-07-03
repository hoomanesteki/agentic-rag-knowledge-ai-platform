# T5 Reproducibility and tests

**Theme:** T5 Reproducibility and tests. **Status:** done.

## What I set out to do

Make the guarantees the platform claims into things CI actually checks, and make the whole thing
reproducible from a clean clone. The dbt tests only ran locally, and there was no explicit test for
a deterministic build or for the semantic layer being well-formed across every domain.

## How I did it

- **dbt runs in CI.** A new `dbt` job installs the dbt extra, builds the medallion for both domains
  (so the schema, relationships, and PII-masking governance tests run on every change), and runs the
  parity test that asserts the dbt gold equals the in-app builder's gold. The dbt tests are no longer
  a local-only thing.
- **Semantic-layer test** (`tests/test_semantic_layer.py`): for every domain, every governed metric
  is checked to be well-formed (a single read-only SELECT, never touching the raw bronze/silver
  layers) and to have its declared parameters match the SQL placeholders exactly, with no dead or
  undeclared parameter. Then each metric is resolved against a freshly built gold, so the single
  source of truth is executed, not just trusted.
- **Reproducibility test** (`tests/test_reproducibility.py`): building a domain twice must produce
  byte-identical gold. The PII pseudonyms are a stable hash and the CSV read is ordered, so a rerun
  can never silently change a governed number.
- **One-command reproduce** (`make reproduce`): setup then the full offline check, so anyone can
  clone and get the same result deterministically, with a pointer to the keyed live-stack path.

## What I tested

- The six new tests pass (two domains times semantic-layer and reproducibility checks), and they run
  offline as part of `make check`, so they are in the default CI job too. No keys, no dbt required.
- `make check` stays green. The CI dbt job builds and tests both domains and asserts parity.

## Where this leaves the guarantees

The claims the project makes now have a test behind each: the metric layer is read-only and
well-formed (semantic layer), PII never reaches gold raw (governance, in dbt and the contracts), the
two build paths agree (parity), the build is deterministic (reproducibility), and a retrieval or
abstain regression is blocked (the eval gate). All of it runs in CI.
