"""The NOTIFY step of the human-gated MLOps loop: turn a CT/drift outcome into ONE deduped GitHub
issue that is the page, the review surface, and the audit trail in one place.

Today notify is an exit code and an uploaded artifact nobody is paged on. This builds a labelled
issue (deduped by label+title so a recurring signal updates the same thread instead of spamming new
ones) carrying the drift signals, the signal-to-action decision, the registered candidate and its
MLflow link, and the EXACT next human commands. Watchers get email/mobile push for free, and the
thread becomes the approval discussion and the record.

The issue body is built purely here (unit-tested); posting it shells out to the gh CLI and is a
no-op dry run when gh or a token is absent, so a notify never breaks the CT run.
"""
from __future__ import annotations

import json
import os
import subprocess


def build_issue(report: dict) -> dict:
    """A GitHub issue (title, label, body) from a CT report dict. The label+title are stable per
    domain so the same signal updates one thread; the body carries everything a human needs to act
    and the exact commands to run, so nothing about the decision lives only in a log."""
    domain = report.get("domain", "default")
    signals = report.get("signals", {}) or {}
    classification = signals.get("classification", {}) or {}
    warranted = classification.get("experiment_warranted")
    label = "ct-signal:{}".format(domain)
    outcome = "candidate proposed" if warranted else "notify only"
    title = "CT signal: {} ({})".format(domain, outcome)

    lines = [
        "Automated by the CT loop. This is the NOTIFY step; a human decides everything downstream.",
        "",
        "## Signal",
        "- drift: {}".format(signals.get("drift_note", "n/a")),
        "- new verified answers: {} (min {})".format(
            signals.get("new_labeled", 0), signals.get("min_new_labeled", 0)),
        "- quality signals: {}".format(classification.get("quality_signals") or "none"),
        "- data/index signals: {}".format(classification.get("data_signals") or "none"),
        "- action: {}".format(classification.get("action", "n/a")),
        "",
        "## Candidate",
    ]
    if warranted and report.get("candidate_path"):
        gain = report.get("gain")
        lines += [
            "- candidate: `{}`".format(report["candidate_path"]),
            "- held-out score: {} vs baseline {} (gain {})".format(
                report.get("candidate_score"), report.get("baseline_score"), gain),
            "- gate passed: {} | safety passed: {}".format(
                report.get("gate_passed"), report.get("safety_passed")),
            "- promotion recommended: {} (PROPOSED only, never auto-promoted)".format(
                report.get("promote_recommended")),
            "- MLflow: experiment `skein-ct` (this run's metrics + candidate artifact)",
            "",
            "## Next (a human runs these)",
            "1. `make shadow N=50`: replay real traffic through the candidate, read the deltas",
            "2. `make registry-promote V=<n>`: promote only if the shadow evidence supports it",
        ]
    else:
        lines += [
            "- no candidate opened: the signal is data/index drift a prompt cannot fix.",
            "",
            "## Next (a human runs these)",
            "1. Investigate retrieval/ingest (is the index stale? has the intent mix shifted?)",
            "2. `make ingest` / refresh the corpus if the data moved; do NOT open a prompt "
            "experiment",
        ]
    return {"title": title, "label": label, "body": "\n".join(lines)}


def _gh_available() -> bool:
    if not (os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")):
        return False
    try:
        subprocess.run(["gh", "--version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def post_issue(issue: dict, *, dry_run: bool | None = None) -> dict:
    """Create or update the labelled issue via gh, deduped by label. A dry run (default when gh or a
    token is absent) prints what it would post and changes nothing, so notify never breaks CT."""
    if dry_run is None:
        dry_run = not _gh_available()
    if dry_run:
        return {"posted": False, "dry_run": True, "issue": issue}
    try:  # find an open issue with this label to update, else create a new one
        found = subprocess.run(
            ["gh", "issue", "list", "--label", issue["label"], "--state", "open",
             "--json", "number", "--limit", "1"],
            capture_output=True, text=True, check=True)
        existing = json.loads(found.stdout or "[]")
        if existing:
            number = str(existing[0]["number"])
            subprocess.run(["gh", "issue", "comment", number, "--body", issue["body"]], check=True)
            return {"posted": True, "updated": number}
        subprocess.run(["gh", "issue", "create", "--title", issue["title"],
                        "--label", issue["label"], "--body", issue["body"]], check=True)
        return {"posted": True, "created": True}
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        return {"posted": False, "error": str(exc)[:200]}
