# Dev notes

Process and pacing notes, kept out of the build plan so that file reads clean for anyone
evaluating the project.

## Toolchain

- Package manager: uv. It manages the Python version too (pinned to 3.12 in `.python-version`).
- Dependencies: ranges in `pyproject.toml`, exact versions pinned in `uv.lock` (committed).
  That is what makes installs reproducible. Add a dep with `uv add <pkg>`, then commit the
  updated lock.
- One gate: `make check` runs ruff lint, pytest, domain-pack validation, and the domain-leak
  linter. CI runs the same target. The `/preflight` skill runs it and reports go or no-go.

## Working on Claude Pro without burning tokens

Claude Pro has usage limits, so keep each building session small and let the repo hold the
state instead of the chat.

- One milestone step per session. Start by reading BUILD-PLAN.md plus only the files that
  step touches. Do not ask Claude to re-read the whole repo.
- Keep state in the repo, not in chat. After each step, run `make check` and commit. When the
  conversation gets long, reset it safely: the plan and code hold the state.
- Use the `/domain-pack` skill to generate and check packs instead of reasoning them out each
  time. Deterministic scaffolding is cheaper than fresh generation.
- Let the app make its own model calls (Groq, Voyage). That work does not touch your Claude
  quota. Claude is for building, not for serving answers.
- Write each "Done when" as a command you can run. A runnable check ends the debate about
  whether a step is finished.
- Prefer small diffs. If a step feels like an L, split it into two S sessions.

## Evaluation and the rerank delta

`make eval` scores the golden set. To record the reranker's effect (M2.2 done-when), run it
both ways against a real Voyage + Qdrant index and compare `hit@k` / `mrr` / `entity_recall@k`:

```bash
make up && make ingest
PYTHONPATH=. uv run python scripts/run_eval.py --no-rerank   # hybrid baseline
make eval                                                    # hybrid + rerank
```

Only the real Voyage run counts. The offline `RERANK_PROVIDER=fake` reranker scores by word
overlap, the same signal as the sparse leg and the abstain gate, so its "delta" is circular
and must not be recorded as the M2.2 number. Paste the two scorecards and the delta here once
you have run them (this environment has no network, so the numbers come from your machine).

## M2.3 grounding and injection (known limits)

The abstain gate holds unchanged, so abstain precision holds by construction; still run
`make eval` on a real index and record the number here to close the done-when. The injection
sanitizer is defense in depth next to the system prompt: it redacts to sentence end (a payload
split into a separate sentence from its trigger survives), and covers English and French plus
zero-width bypasses, not every phrasing. The grounding score measures citation discipline, not
faithfulness (RAGAS at M8). Two follow-ups for M8: compute the gate on sanitized text so an
injection cannot inflate confidence, and add an adversarial question to the golden set.

## M2.4 chunking (scope and follow-ups)

The seed reviews are single sentences, so the sentence packer produces one chunk each and is
a retrieval no-op on this corpus; it is exercised by unit tests, and the real recall delta is
recorded via the M2.5 ablation on a real index. The contextual prefix is opt-in per pack
(`context_fields` in the manifest) and is embedded with the text but kept out of the stored
and displayed text, so it cannot pollute citations or the abstain gate. Whether the prefix
helps is measured, not assumed. Tracked follow-ups (M4, when data grows): delete points by
`record_id` before re-upsert so a record that shrinks from N to fewer chunks does not orphan
old points; an intra-sentence window fallback for a single sentence over the token budget;
and preserving paragraph structure.

## Git and attribution

Commits use your own git identity. No assistant attribution goes into commit messages or PR
bodies. Work on short-lived `build/<step>` branches, run `make check`, then open a PR (CI runs
on it) and merge when green.
