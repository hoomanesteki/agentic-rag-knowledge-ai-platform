"""A lightweight, versioned, staged model registry, so every retrain leaves an auditable record and
promotion stays human-gated (nothing self-ships).

A "model" here is a serving artifact this project owns and versions: a prompt candidate, or a
serving config (LLM + embedder + reranker + brain). Each registered version carries its metrics
(baseline vs candidate, gate, safety), a stage, and provenance (what produced it, when). The stages
mirror the MLflow Model Registry:

    proposed  ->  staging  ->  production        (and any version can be archived)

It is JSON-backed so it works offline in CI and in the demo, where scripts/promote_model.py also
writes the real MLflow Registry when a tracking server is configured; the two stay consistent by the
same rule, register as `proposed`, a human transitions to `production`. Continuous Training calls
record_cycle() every weekly run (an audit entry, promotable or not) and register() only when the
cycle produced a promotable candidate, which is what makes "the model is registered on a weekly
cadence" true and reviewable without ever auto-promoting.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

STAGES = ("proposed", "staging", "production", "archived")


@dataclass
class ModelVersion:
    version: int
    name: str
    kind: str            # "prompt" | "config"
    stage: str           # one of STAGES
    created_at: str
    source: str          # what produced it, e.g. "ct-cycle" or "prompt_opt"
    metrics: dict = field(default_factory=dict)
    notes: str = ""


class ModelRegistry:
    """A JSON-backed registry of model versions and CT-cycle audit records. All mutations persist
    immediately, and time is injected (created_at / at), so it stays reproducible and unit-testable
    with no wall clock."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, encoding="utf-8") as f:
                    data = json.load(f)
                data.setdefault("versions", [])
                data.setdefault("cycles", [])
                return data
            except (OSError, json.JSONDecodeError):
                pass
        return {"versions": [], "cycles": []}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    def register(self, *, name: str, kind: str, source: str, created_at: str,
                 metrics: dict | None = None, notes: str = "") -> int:
        """Register a new version as `proposed` (never auto-promoted) and return its number."""
        version = len(self._data["versions"]) + 1
        self._data["versions"].append(asdict(ModelVersion(
            version=version, name=name, kind=kind, stage="proposed", created_at=created_at,
            source=source, metrics=metrics or {}, notes=notes)))
        self._save()
        return version

    def transition(self, version: int, stage: str, *, at: str) -> bool:
        """Move a version to a stage. Promoting to `production` archives the incumbent, so there is
        exactly one production model at a time. Returns False if the version is unknown."""
        if stage not in STAGES:
            raise ValueError("unknown stage: {}".format(stage))
        target = next((v for v in self._data["versions"] if v["version"] == version), None)
        if target is None:
            return False
        if stage == "production":
            for other in self._data["versions"]:
                if other["stage"] == "production" and other["version"] != version:
                    other["stage"] = "archived"
        target["stage"] = stage
        target["transitioned_at"] = at
        self._save()
        return True

    def champion(self) -> dict | None:
        """The current production version, or None if nothing has been promoted yet."""
        prod = [v for v in self._data["versions"] if v["stage"] == "production"]
        return prod[-1] if prod else None

    def record_cycle(self, cycle: dict) -> None:
        """Append a CT-cycle audit entry (kept for every cycle, promotable or not)."""
        self._data["cycles"].append(cycle)
        self._save()

    def versions(self) -> list[dict]:
        return list(self._data["versions"])
