"""Agent-level evaluation for the omni brain: does the router send each turn to the right lane?

This is the routing tier of the ground-truth eval (domains/<domain>/eval/routing.jsonl), the cheap,
high-volume half of the plan. It exercises the router only, no retrieval and no answer generation,
so it runs in seconds and costs at most one small-model call per genuinely ambiguous turn. It
reports overall and per-stratum lane accuracy, a confusion matrix, escalation precision and recall,
and the deterministic decision rate: how often layers 0 and 1 decide without paying for the
small-model tie-break, which is the routing-is-cheap claim made measurable.

Some strata are scored on intent, not a single lane. An ambiguous turn is correct when the brain
asks (clarify) or safely falls back to answers instead of guessing a confident wrong lane. A
multitask turn is correct when the router surfaces the two-intent plan or lands the primary lane.
A pii_probe is about the deterministic order gate (graded by the safety suite, not here), so its
lane is only reported. The eval set is curated synthetic, labeled and spot-checked; in production
this tier would be human-annotated (two annotators plus adjudication, with an agreement score).
"""
from __future__ import annotations

import json
import os

from rag.router import route

_CLEAN = ("stylist", "care", "complaint", "escalation", "answers")
# per-1M-token price of the small classifier, for the (tiny) cost of the tie-break calls
_SMALL_PRICE_IN, _SMALL_PRICE_OUT = 0.05, 0.08


def load_cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _graded_correct(case: dict, decision) -> bool | None:
    """True/False if the case is graded on lane, None if it is only reported (pii_probe)."""
    st, want, lane = case["stratum"], case["intended_lane"], decision.lane
    if st == "pii_probe":
        return None
    if st == "ambiguous":
        return decision.clarify is not None or lane == "answers"
    if st == "multitask":
        return lane == want or bool(decision.tasks) or lane == "escalation"
    return lane == want


def evaluate_routing(cases: list[dict], *, small_llm=None, mode: str | None = None,
                     tiebreak_system: str | None = None) -> dict:
    """Route every case and aggregate. With small_llm=None only layers 0 and 1 run (free,
    deterministic); pass a model to include the layer-2 tie-break on ambiguous turns. mode overrides
    the label so a model-tier comparison (8B vs 70B tie-break) can name each run. tiebreak_system
    overrides the tie-break prompt so the prompt-optimization loop can score a candidate."""
    graded = correct = det_decisions = llm_calls = 0
    per: dict[str, list[int]] = {}
    confusion: dict[str, int] = {}
    esc_tp = esc_fp = esc_fn = 0
    for c in cases:
        d = route(c["query"], signed_in=c.get("signed_in", False), small_llm=small_llm,
                  tiebreak_system=tiebreak_system)
        if d.layer < 2:
            det_decisions += 1
        else:
            llm_calls += 1
        ok = _graded_correct(c, d)
        if ok is not None:
            graded += 1
            correct += int(ok)
            row = per.setdefault(c["stratum"], [0, 0])
            row[0] += int(ok)
            row[1] += 1
            if c["stratum"] in _CLEAN and not ok:
                key = "{}->{}".format(c["intended_lane"], d.lane)
                confusion[key] = confusion.get(key, 0) + 1
        # escalation precision/recall, excluding multitask (a multitask turn that asks for a human
        # legitimately routes to escalation, so counting it as a false positive would mislead)
        if c["stratum"] != "multitask":
            want_esc = c["intended_lane"] == "escalation"
            got_esc = d.lane == "escalation"
            esc_tp += int(want_esc and got_esc)
            esc_fp += int(got_esc and not want_esc)
            esc_fn += int(want_esc and not got_esc)
    n = len(cases)
    esc_prec = esc_tp / (esc_tp + esc_fp) if (esc_tp + esc_fp) else 1.0
    esc_rec = esc_tp / (esc_tp + esc_fn) if (esc_tp + esc_fn) else 1.0
    # the tie-break calls are tiny (a ~60-token prompt, a ~10-token reply); estimate conservatively
    tie_cost = round(llm_calls * (80 / 1e6 * _SMALL_PRICE_IN + 12 / 1e6 * _SMALL_PRICE_OUT), 6)
    return {
        "mode": mode or ("with_tiebreak" if small_llm is not None else "deterministic"),
        "n": n,
        "accuracy": round(correct / graded, 4) if graded else 0.0,
        "graded": graded,
        "deterministic_decision_rate": round(det_decisions / n, 4) if n else 0.0,
        "tiebreak_calls": llm_calls,
        "tiebreak_cost_usd": tie_cost,
        "by_stratum": {k: round(v[0] / v[1], 4) for k, v in sorted(per.items())},
        "escalation": {"precision": round(esc_prec, 4), "recall": round(esc_rec, 4),
                       "tp": esc_tp, "fp": esc_fp, "fn": esc_fn},
        "confusion": dict(sorted(confusion.items(), key=lambda kv: -kv[1])),
    }


def format_scorecard(sc: dict) -> str:
    lines = [
        "Routing eval ({}), n={}, graded={}".format(sc["mode"], sc["n"], sc["graded"]),
        "  accuracy={:.1%}  det_decision_rate={:.1%}  tiebreak_calls={} (~${:.4f})".format(
            sc["accuracy"], sc["deterministic_decision_rate"], sc["tiebreak_calls"],
            sc["tiebreak_cost_usd"]),
        "  escalation precision={:.1%} recall={:.1%}".format(
            sc["escalation"]["precision"], sc["escalation"]["recall"]),
        "  by stratum: " + ", ".join("{} {:.0%}".format(k, v)
                                     for k, v in sc["by_stratum"].items()),
    ]
    if sc["confusion"]:
        top = list(sc["confusion"].items())[:6]
        lines.append("  top confusions: " + ", ".join("{} ({})".format(k, v) for k, v in top))
    return "\n".join(lines)


def log_to_mlflow(scorecards: list[dict], *, tracking_uri: str, experiment: str = "skein-omni-eval",
                  run_name: str | None = None) -> None:
    """Log each scorecard as an MLflow run (mode as a param, the scalars as metrics). No-op if no
    tracking uri is set, so the eval still runs and writes its JSON artifact offline."""
    if not tracking_uri:
        return
    if not tracking_uri.startswith(("http://", "https://")):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        _log_runs(scorecards, tracking_uri, experiment, run_name)
    except Exception as exc:  # a down or unauthorized tracking server must not fail the eval; the
        print("  (mlflow logging skipped: {})".format(str(exc)[:120]))  # JSON report still stands


def _log_runs(scorecards, tracking_uri, experiment, run_name):
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=tracking_uri)
    found = client.get_experiment_by_name(experiment)
    exp_id = found.experiment_id if found else client.create_experiment(experiment)
    for sc in scorecards:
        run = client.create_run(exp_id, run_name=run_name or ("routing-" + sc["mode"]))
        rid = run.info.run_id
        client.log_param(rid, "mode", sc["mode"])
        client.log_param(rid, "n", sc["n"])
        for k in ("accuracy", "deterministic_decision_rate", "tiebreak_calls", "tiebreak_cost_usd"):
            client.log_metric(rid, k, float(sc[k]))
        client.log_metric(rid, "escalation_precision", float(sc["escalation"]["precision"]))
        client.log_metric(rid, "escalation_recall", float(sc["escalation"]["recall"]))
        for lane, acc in sc["by_stratum"].items():
            client.log_metric(rid, "acc_" + lane, float(acc))
        client.set_terminated(rid)
