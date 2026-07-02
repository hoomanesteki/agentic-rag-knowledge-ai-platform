---
name: review
description: Run an independent, adversarial code review with Fable 5 and fall back to Opus if the reviewer stalls. Use before merging a build step, when asked to "check with Fable", "review this", or to sanity-check a diff or subsystem. Returns concrete P0/P1/P2 findings with file:line and a minimal fix.
---

# Independent review

A second model reviews the change with fresh eyes, so bugs are caught before they merge. Fable is
the reviewer of record; if its stream stalls (a known watchdog failure, not a code problem), fall
back to an inline Opus review so the step is never blocked.

## Steps

1. **Scope it.** List the exact files this change touches (a `git diff --name-only` against the
   base is a good source). Review the changed files, not the whole repo.
2. **Launch Fable.** Spawn a subagent with `model: "fable"`, read-only (it must not edit or run
   `make check`; the gate is already green). Give it:
   - the files to read (named explicitly, "read only these"),
   - what the change is meant to do,
   - a request for a concise reply (under ~400 words) grouping findings as **P0** (fix now),
     **P1** (soon), **P2** (nice), each with `file:line`, the bug, and the one-line fix,
   - priority order: correctness and integration bugs first, then security, then cost, then style.
   - For untested paths (a real-infra adapter with no offline test), tell it to focus there, since
     no local test can catch those.
3. **Handle a stall.** If the notification says the agent stalled ("no progress for 600s / stream
   watchdog"), that is an infrastructure fault, not the task. Retry Fable ONCE with a tighter
   scope. If it stalls again, do the review yourself with Opus: read the same files and produce the
   same P0/P1/P2 structure. Say plainly in the report that Fable stalled and Opus did the review.
4. **Triage and apply.** Fix P0 and P1 in this step. Defer a P2 only with a one-line rationale in
   `docs/DEV-NOTES.md`; do not defer a real correctness or security bug. When a fix would trade one
   real failure for another (a brittle heuristic that regresses a common case), say so and defer to
   the milestone that solves it properly, rather than shipping the worse fix.
5. **Regression-test.** Add a test for each real bug fixed, then re-run `make check`.
6. **Report.** A short verdict line, the findings grouped P0/P1/P2, what was fixed vs deferred, and
   which reviewer produced the result.

## Notes

- A whole-subsystem "end-to-end" pass after several sub-steps catches cross-cutting bugs the
  per-step reviews miss (contract drift between modules, a shared-resource wipe, ordering). Run one
  at the end of a milestone.
- The reviewer's final message is the tool result and is not shown to the user; relay what matters.
- Keep the review read-only. It reports; this session applies the fixes.
