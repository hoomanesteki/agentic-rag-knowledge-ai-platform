---
name: ship
description: Ship one build step end to end - branch, build, make check, independent review, chunked commits under the owner's identity, and a no-ff merge to main. Use when implementing a milestone sub-step (an M-dot-x), completing a fix, or whenever asked to "ship", "build and merge", or "do the next step". Encodes the project's standing rules so they are never re-derived.
---

# Ship a build step

The repeatable loop for every unit of work. It keeps the trunk green, every change reviewed by
an independent model, and every commit under the owner's identity with no assistant attribution.

## Standing rules (never break these)

- Commits are authored by the owner, not the assistant:
  `git -c user.name="hoomanesteki" -c user.email="esteki.net@gmail.com" commit --no-gpg-sign ...`
- ZERO assistant attribution anywhere: no `Co-Authored-By`, no "generated with", no trailer.
- Write plainly, low AI signal. NEVER use an em dash. Conventional-commit subjects
  (`feat(scope): ...`, `fix(scope): ...`, `test(...)`, `docs(...)`, `chore(...)`).
- Commit in small, standard chunks (one coherent change each) so the graph grows and history reads
  well. Do not sweep unrelated files into one commit; always `git add` explicit paths.
- Never commit or merge on a red `make check`.

## Steps

1. **Branch.** From an up-to-date `main`: `git checkout -b build/<step>` (use `docs/<name>` for
   docs-only work). If work was already started on `main`, `git checkout -b build/<step>` carries
   the uncommitted changes onto the branch.
2. **Build.** Implement the smallest slice that satisfies the step's "Done when". Match the
   surrounding code's style, comment density, and idioms. Add or extend tests in the same step.
3. **Gate.** Run `make check` (lint, tests, domain validation, leak check). It must be green
   before review. Fix and re-run until green.
4. **Review.** Invoke the `review` skill (independent Fable pass, Opus fallback on stall) scoped to
   the files this step changed. Apply P0 and P1 findings now; defer P2 only with a one-line
   rationale recorded in `docs/DEV-NOTES.md`. Add a regression test for every real bug fixed.
5. **Re-gate.** Run `make check` again after the fixes. Green.
6. **Commit in chunks.** Stage explicit paths per logical change and commit each with the owner
   identity command above. Typical split: core change, wiring, tests, docs.
7. **Merge.** `git checkout main` then
   `git -c user.name="hoomanesteki" -c user.email="esteki.net@gmail.com" merge --no-ff --no-gpg-sign -m "Merge build/<step>: <summary>" build/<step>`,
   then `git branch -d build/<step>`.
8. **Confirm.** Run `make check` on `main` and report: step done, test count, what merged, and any
   deferred follow-ups.

## Notes

- Real infrastructure (Groq, Cohere, Qdrant, Neo4j) is not reachable in this environment; build
  and test against the offline fakes, and hand the owner the one real-infra command to run
  locally (documented in DEV-NOTES), rather than claiming an unverified real run.
- Push and open a PR only when the owner asks. The default is branch, review, merge locally.
- Keep the todo list current: one in-progress item, marked done when merged.
