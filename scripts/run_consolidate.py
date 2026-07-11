#!/usr/bin/env python3
"""Human-triggered knowledge consolidation: distill recent traffic (repeats, thumbs-down, abstains)
and closed review answers into a PROPOSED knowledge pack under evaluation/reports/, for a human to
review and approve. Nothing is indexed here; on approval the existing flywheel does that.

    make consolidate

Reads the trace log, the feedback log, and the review queue. Runs offline; no keys needed.
"""
from __future__ import annotations

import json
import os
import sys
import time

from adapters.config import get_settings
from mlops.consolidate import propose_pack
from rag.hitl import ReviewQueue

_TRACES = os.getenv("TRACE_PATH", "traces/requests.jsonl")
_FEEDBACK = os.getenv("FEEDBACK_PATH", "traces/feedback.jsonl")
_OUT = "evaluation/reports/consolidation_proposal.json"


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def main() -> int:
    settings = get_settings()
    traces = _read_jsonl(_TRACES)
    feedback = _read_jsonl(_FEEDBACK)
    try:
        rq = ReviewQueue(settings.review_queue_db)
        closed = rq.closed_since(0.0, domain=settings.domain)
    except Exception:
        closed = []

    pack = propose_pack(traces, feedback, closed)
    pack["domain"] = settings.domain
    pack["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    with open(_OUT, "w", encoding="utf-8") as f:
        json.dump(pack, f, indent=2, ensure_ascii=False)

    c = pack["counts"]
    print("consolidation proposal -> {}".format(_OUT))
    print("  candidate chunks (from human-verified answers): {}".format(c["chunks"]))
    print("  candidate eval rows: {}".format(c["eval_rows"]))
    print("  knowledge gaps (frequent abstains / thumbs-down): {}".format(c["gaps"]))
    print("  frequent repeat queries: {}".format(c["frequent"]))
    print("\nreview and approve the proposal, then the flywheel indexes it. Nothing indexed here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
