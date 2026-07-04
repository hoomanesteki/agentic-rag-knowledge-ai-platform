---
name: pr-review
description: Review a GitHub pull request and post the findings back as PR review comments and suggested changes, then leave the merge decision to the human. Use when asked to "review this PR", "review PR <n>", "check my PR before I merge", or when a PR is opened and you want an AI reviewer-in-the-loop. Reviewer suggests; a person approves and merges.
---

# PR review (reviewer-in-the-loop)

An AI reviewer reads the pull request, finds real problems, and posts them **on the PR** as review
comments and GitHub suggested-changes, so the author sees them inline and stays in control. The AI
never approves or merges; it recommends, the human decides. This is the safe division of labor:
machine catches the bugs, person owns the call.

## Preconditions
- `gh` is installed and authenticated (`gh auth status`; if not, tell the user to run `gh auth login`).
- A PR exists for the branch. If not, offer to open one with `gh pr create` first.

## Steps

1. **Resolve the PR.** Use the number the user gave, else the PR for the current branch:
   `gh pr view --json number,title,headRefName,baseRefName,url`.

2. **Get the diff and changed files.**
   - `gh pr diff <n>` for the unified diff.
   - `gh pr diff <n> --name-only` for the file list.
   Review only the changed files, in the context of how they are meant to be used.

3. **Run the adversarial review.** Reuse the `review` skill's approach: spawn a Fable subagent
   (read-only, no edits, no `make check`) scoped to the changed files, asking for concise findings
   grouped **P0** (fix now) / **P1** (soon) / **P2** (nice), each with `file:line`, the concrete
   failure (input to wrong output), and a minimal fix. If Fable stalls twice, do the review inline
   with Opus and say so. Priority order: correctness and integration, then security, then cost,
   then style.

4. **Post the findings on the PR.** Keep the author in control:
   - For each concrete, localized fix, post an inline **suggested change** so the author can accept
     it with one click. Build a review with the GitHub API so suggestions land on the right lines:
     ```bash
     gh api repos/{owner}/{repo}/pulls/<n>/reviews -f event=COMMENT \
       -f body="AI review: N findings (P0/P1/P2). Suggestions inline; you decide what to take." \
       -f 'comments[][path]=path/to/file.py' -F 'comments[][line]=42' \
       -f 'comments[][body]=**P1** why this is wrong.

```suggestion
the corrected line(s)
```'
     ```
   - For findings that are not a one-line fix (design, missing test, security), post them as plain
     review comments with the `file:line` and the reasoning.
   - Summarize P0/P1/P2 counts in the review body.

5. **Do NOT approve or merge.** Explicitly leave `event=COMMENT` (never `APPROVE`). Tell the user:
   "Posted N findings on <PR url> as comments and suggestions. Review them and, if you agree,
   accept the suggestions and merge." The human approves.

6. **Report back** in chat: the PR url, the P0/P1/P2 counts, and the one or two findings most worth
   their attention, so they can act without opening every thread.

## Guardrails
- Never push commits to the PR branch or resolve conversations; only comment and suggest.
- Never post secrets or full file contents; reference `file:line`.
- If there are zero real findings, post a short "looks good, no blocking issues" review comment
  (still `event=COMMENT`) rather than inventing nits.
- A P0 correctness or security bug is always called out, even on a small PR.
