"""Run the routing eval and write a reproducible scorecard.

Usage:
  PYTHONPATH=. uv run python scripts/run_agent_eval.py            # deterministic only (free)
  PYTHONPATH=. uv run python scripts/run_agent_eval.py --with-llm # add the 8B tie-break pass

Always writes evaluation/reports/routing_eval.json (the numbers the docs cite, regenerable), and
logs to MLflow when MLFLOW_TRACKING_URI is set. Deterministic mode makes no network calls.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from adapters.config import get_settings
from adapters.factory import make_llm, make_small_llm
from evaluation.agent_eval import evaluate_routing, format_scorecard, load_cases, log_to_mlflow

_EVAL_PATH = "domains/apparel_ecommerce/eval/routing.jsonl"
_REPORT = "evaluation/reports/routing_eval.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llm", action="store_true",
                    help="also run the layer-2 small-model tie-break (needs a Groq key)")
    ap.add_argument("--path", default=_EVAL_PATH)
    args = ap.parse_args()

    cases = load_cases(args.path)
    scorecards = [evaluate_routing(cases)]  # deterministic, always
    print(format_scorecard(scorecards[0]))

    if args.with_llm:
        small = make_small_llm()
        if small is None:
            print("(no small model available; skipping the tie-break passes)")
        else:
            sc8 = evaluate_routing(cases, small_llm=small, mode="tiebreak_8b")
            scorecards.append(sc8)
            print(format_scorecard(sc8))
            # model-tier A/B: does a bigger model tie-break better? This is the runnable, cheap
            # evidence for the frontier decision on the routing task (docs/omni-plan-v2.md).
            large = make_llm()
            sc70 = evaluate_routing(cases, small_llm=large, mode="tiebreak_70b")
            scorecards.append(sc70)
            print(format_scorecard(sc70))

    os.makedirs(os.path.dirname(_REPORT), exist_ok=True)
    with open(_REPORT, "w", encoding="utf-8") as f:
        json.dump({"eval_set": args.path, "scorecards": scorecards}, f, indent=2)
    print("wrote", _REPORT)

    log_to_mlflow(scorecards, tracking_uri=get_settings().mlflow_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
