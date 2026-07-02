# Retrieval ablation (apparel_ecommerce)

OFFLINE PLACEHOLDER: the index is empty, so retrieval numbers are zero and the gate trivially abstains on everything (so the gate columns read 1.000, which is not a real score). Run `make up && make ingest && make ablation` on a machine with the API keys to fill real numbers.

Retrieval quality across variants on the measurable (qualitative) golden questions. The +chunking column needs a re-ingest (toggle the manifest context_fields) written to a separate report; compare the two.

provenance: embed=fake, rerank=voyage, top_k_in=50, generated=2026-07-02 04:29 UTC
top_k=8, measured 4 qualitative question(s), 7 deferred to M4/M5, abstain-set 6.

| variant | scope | hit@k | entity_recall@k | mrr | false_abstain | abstain_recall |
|---|---|---|---|---|---|---|
| dense | overall | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| dense | en | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| dense | fr | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| hybrid | overall | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| hybrid | en | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| hybrid | fr | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| hybrid+rerank(voyage) | overall | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| hybrid+rerank(voyage) | en | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| hybrid+rerank(voyage) | fr | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
