"""Provable self-learning: a time series of the assistant's eval score over Continuous Training
cycles, so "it gets smarter over time" is a graph a person can read, not a claim. CT appends one
point per cycle (the held-out score against the growing verified eval set) and the showcase plots
it. Pure and append-only, so it is unit-testable and auditable.
"""
from __future__ import annotations

import json
import os


def append_point(path: str, *, at: str, score, n_examples: int, version=None) -> None:
    """Append one learning-curve point. `at` is an injected timestamp, so runs stay reproducible."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": at, "score": score, "n_examples": n_examples,
                            "version": version}) + "\n")


def read_curve(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    points = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    points.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return sorted(points, key=lambda p: p.get("at", ""))


def trend(points: list[dict]) -> dict:
    """The direction of the curve, comparing the first and last scored cycles."""
    scored = [p for p in points if isinstance(p.get("score"), (int, float))]
    if len(scored) < 2:
        return {"points": len(scored), "delta": None, "improving": None}
    delta = round(scored[-1]["score"] - scored[0]["score"], 4)
    return {"points": len(scored), "first": scored[0]["score"], "last": scored[-1]["score"],
            "delta": delta, "improving": delta > 0}
