# T2 Sample data and company knowledge

**Theme:** T2 Sample data and knowledge. **Status:** done.

## What I set out to do

Make the domain answer well the moment you pick it. The structured data already drives the
governed metrics and the graph, but a real user mostly asks how-to and policy questions, and there
was no company-knowledge corpus for those. Add one without breaking the governed
numbers or the eval.

## How I did it

- **A new unstructured source**, `company_knowledge.jsonl`, declared in the manifest as
  `doc_type: guide`. The ingest, eval, and ablation runners already iterate every unstructured
  source, so it flows through with no engine change.
  - Apparel (Aster Athletics): 16 docs covering shipping, returns and exchanges, warranty, sizing,
    care, payment, loyalty, gift cards, price adjustments, sustainability, tracking, and support,
    with French versions of returns and shipping.
- **Kept the governed numbers stable.** I did not touch the structured tables, so the metric values
  the golden set pins (for example return rate for size M = 0.5) are unchanged.
- **Kept the eval honest.** Adding a warranty guide made the apparel golden's "warranty period"
  probe answerable, so I changed it from unanswerable to answerable (expects "one year") rather than
  leave a now-answerable question marked unanswerable, and swapped in a genuinely-unanswerable
  French probe (a student discount we do not offer) to keep abstain coverage. Added company-knowledge
  golden items in both languages (returns window, express shipping cost).

## What I tested

- The pack still meets the contract (`make validate`), and `make check` is green: the domain
  retrieval test still surfaces its evidence, the route heuristic stays above its floor, and the
  golden type coverage holds.
- An offline ingest check confirms the new docs are chunked and retrievable (lexical matches land,
  for example an API-key question surfaces the API-key guide). Semantic ranking quality is a keyed
  Voyage run, since the offline fake embedder is deterministic hashing, not meaning.

## Note

Structured data stayed small on purpose: it feeds the governed metrics and the golden pins their
values, so growing it means re-deriving those numbers. The high-value, low-risk win for "answers
well as a sample" was the company-knowledge corpus, which is where real user questions land.
