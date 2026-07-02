#!/usr/bin/env python3
"""Run the CI eval gate on the recorded fixtures. Exits non-zero (blocking a merge) when the
score falls below the threshold. Fully offline, so it runs in CI. Run: make gate
"""
from __future__ import annotations

import os
import sys
import tempfile

from evaluation.ci_gate import load_gate, run_gate


def main() -> int:
    path = os.getenv("GATE_FIXTURES", "evaluation/fixtures/gate.json")
    min_score = float(os.getenv("GATE_MIN_SCORE", "1.0"))
    trace_path = os.path.join(tempfile.mkdtemp(), "gate.jsonl")  # never touch the real traces
    result = run_gate(load_gate(path), min_score=min_score, trace_path=trace_path)
    for row in result["results"]:
        print("  {} {} -> {}".format(row["id"], row["expect"],
                                     "PASS" if row["passed"] else "FAIL"))
    print("gate score {} (min {}) -> {}".format(
        result["score"], min_score, "OK" if result["passed"] else "BLOCKED"))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
