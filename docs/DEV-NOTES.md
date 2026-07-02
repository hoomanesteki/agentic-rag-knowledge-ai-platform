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

## Git and attribution

Commits use your own git identity. No assistant attribution goes into commit messages or PR
bodies. Work on short-lived `build/<step>` branches, run `make check`, then open a PR (CI runs
on it) and merge when green.
