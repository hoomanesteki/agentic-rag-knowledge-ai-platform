---
name: preflight
description: Run every project check before committing or merging (lint, tests, domain-pack validation) and give a clear go or no-go. Use before opening a branch for merge, before committing a milestone step, or whenever asked to confirm everything passes.
---

# Preflight

Run the same gate that CI runs, locally, and report the result plainly.

## Steps

1. From the repo root with the virtualenv active, run `make check`. It runs ruff lint,
   pytest, and validates every domain pack under `domains/`.
2. Run `git status --short` to see exactly what would be committed.
3. Report a short go or no-go:
   - Go: name what passed (lint clean, N tests passed, packs validated) and say whether the
     tree is clean or list the intended changes.
   - No-go: show the first failing check with its exact error, and stop. Do not commit or
     merge on a red check.

## Notes

- If `make` is unavailable, run the pieces directly: `ruff check .`, `pytest`, and
  `python .claude/skills/domain-pack/scripts/validate_domain_pack.py domains/<name>` per pack.
- Never commit or merge while a check is red. Fix it first, or surface it to the user.
