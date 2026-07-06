#!/usr/bin/env python3
"""Inspect the model registry, and make the deliberate manual step: promote a version.

Continuous Training registers a retrained candidate as `proposed`; retraining and evaluation are
automated, deployment is NOT. A human reviews the proposal and runs this to move it along a stage:

  make registry                          # show every version, its stage, and the current champion
  make registry-promote V=3              # move version 3 to production (archives the incumbent)
  make registry-promote V=3 STAGE=staging

Promotion to production archives the previous champion, so exactly one model is live at a time.
When an MLflow tracking server is configured, this also mirrors the transition to the real Registry.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from mlops.model_registry import ModelRegistry

_REGISTRY = os.getenv("MODEL_REGISTRY_PATH", "evaluation/reports/model_registry.json")


def _show(reg: ModelRegistry) -> None:
    versions = reg.versions()
    champ = reg.champion()
    print("model registry: {} version(s), {} recorded CT cycle(s)".format(
        len(versions), len(reg._data.get("cycles", []))))
    for v in versions:
        print("  v{} [{}] {} '{}' {}".format(
            v["version"], v["stage"], v["kind"], v["name"], v.get("metrics") or ""))
    print("champion:", "v{}".format(champ["version"]) if champ else "none (nothing promoted yet)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-V", "--version", type=int, default=int(os.getenv("V") or 0),
                    help="version to transition; 0 or omitted just lists the registry")
    ap.add_argument("--stage", default=os.getenv("STAGE", "production"),
                    help="target stage: proposed | staging | production | archived")
    ap.add_argument("--list", action="store_true", help="show the registry and exit")
    args = ap.parse_args()

    reg = ModelRegistry(_REGISTRY)
    if args.list or not args.version:
        _show(reg)
        return 0

    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if not reg.transition(args.version, args.stage, at=stamp):
        print("no version {} in the registry".format(args.version))
        return 1
    print("transitioned v{} -> {}".format(args.version, args.stage))
    _show(reg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
