"""Optimize the router's tie-break prompt against the routing ground truth, safety-gated.

The routing eval (docs/eval-routing-findings.md) found the tie-break prompt guesses a specialist
lane on turns it should defer on. This runs the OPRO loop on that exact prompt: a Groq model
proposes more conservative rewrites from the failing cases, each is scored on a TRAIN split, and the
winner is reported on a held-out TEST split so the gain has to generalize. Nothing is promoted
automatically; the winner is written to mlops/prompt_registry as a candidate for a human to review.

  PYTHONPATH=. uv run python scripts/run_prompt_opt.py   # needs a Groq key

Deterministic split (fixed seed), so the run is reproducible.
"""
from __future__ import annotations

import json
import os
import random
import sys

from adapters.factory import make_llm, make_small_llm
from evaluation.agent_eval import _graded_correct, load_cases
from mlops.prompt_opt import optimize_prompt, save_candidate
from rag.router import _TIEBREAK_SYSTEM, _model_tiebreak, route

_SET = "domains/apparel_ecommerce/eval/routing.jsonl"
_REPORT = "evaluation/reports/prompt_opt.json"
# the strata the tie-break actually affects, so the loop spends its calls where the prompt matters
_TIEBREAK_STRATA = ("ambiguous", "answers", "complaint", "multilingual", "multitask", "stylist")


def _evaluate(prompt, cases, small):
    graded = correct = 0
    misrouted = []
    for c in cases:
        d = route(c["query"], signed_in=c.get("signed_in", False), small_llm=small,
                  tiebreak_system=prompt)
        ok = _graded_correct(c, d)
        if ok is None:
            continue
        graded += 1
        correct += int(ok)
        if not ok:
            misrouted.append({"query": c["query"], "want": c["intended_lane"], "got": d.lane})
    return (correct / graded if graded else 0.0), misrouted


def _propose(current, misrouted, n, proposer):
    ex = "\n".join("- {!r} should be {} but got {}".format(m["query"], m["want"], m["got"])
                   for m in misrouted) or "(none)"
    ask = (
        "You are improving a shopping assistant's router SYSTEM PROMPT. The current prompt sorts a "
        "message into one of stylist, care, complaint, answers, unclear as JSON {\"lane\": L}. It "
        "misroutes the examples below. Rewrite the prompt to fix them. The key problem is that it "
        "GUESSES a specialist lane on vague or general questions; it should DEFER: choose answers "
        "for general or policy questions, and unclear only when a message genuinely reads as two "
        "different specific intents. Keep it short, keep the exact JSON output contract, do not "
        "invent lanes. Reply with ONLY the new system prompt text, no preamble.\n\n"
        "CURRENT PROMPT:\n" + current + "\n\nMISROUTED EXAMPLES:\n" + ex)
    out = []
    for i in range(n):
        try:
            r = proposer.generate(ask + "\n\nWrite variant {}.".format(i + 1),
                                  system="You write concise, precise router system prompts.",
                                  max_tokens=320)
            out.append(r.text.strip())
        except Exception:
            continue
    return out


# Messages that clearly belong to different lanes, used to prove a candidate is not degenerate.
_PROBE = ["can you suggest a gift for my mum", "you charged me twice and I'm upset",
          "what is your return policy", "it is not quite right, not sure"]


def _make_safety(small):
    """The hard gate. Checks the candidate's TEXT (it must keep the JSON contract and the lanes) AND
    its BEHAVIOR: a prompt that collapses every turn to one lane (a reward-hacking candidate that
    beats the metric by always saying 'answers') produces one lane on the probe and is rejected."""
    def safety(prompt: str) -> bool:
        low = prompt.lower()
        if not ('"lane"' in prompt and len(prompt) < 1600
                and all(w in low for w in ("stylist", "care", "complaint", "answers", "unclear"))):
            return False
        lanes = set()
        for q in _PROBE:
            d = _model_tiebreak(q, small, system=prompt)
            lanes.add(d.lane if d else "none")
        return len(lanes) >= 3  # not degenerate: it distinguishes at least three lanes
    return safety


def main() -> int:
    small = make_small_llm()
    if small is None:
        print("no small model (set LLM_PROVIDER=groq with a key); cannot run the loop")
        return 1
    proposer = make_llm()

    cases = [c for c in load_cases(_SET) if c["stratum"] in _TIEBREAK_STRATA]
    rng = random.Random(13)
    rng.shuffle(cases)
    cut = int(len(cases) * 0.6)
    train, test = cases[:cut], cases[cut:]

    safety = _make_safety(small)
    result = optimize_prompt(
        _TIEBREAK_SYSTEM,
        evaluate=lambda p: _evaluate(p, train, small),
        propose=lambda p, m, n: _propose(p, m, n, proposer),
        safety=safety, rounds=2, n_candidates=3)

    base_test, _ = _evaluate(_TIEBREAK_SYSTEM, test, small)
    best_test, _ = _evaluate(result["best_prompt"], test, small)
    # Decide promotion and safety and WRITE the candidate BEFORE building the report, so the report
    # records the candidate path actually written (or "") and the explicit safety result. The CT
    # loop reads those instead of guessing from a possibly-stale file. A real held-out margin is
    # required (not a single flipped case), and the hard safety gate is re-checked.
    promote = result["improved"] and best_test > base_test + 0.01
    safety_ok = bool(safety(result["best_prompt"]))
    candidate_path = ""
    if promote and safety_ok:
        candidate_path = save_candidate("tiebreak_system", result["best_prompt"],
                                        {"baseline_test": round(base_test, 4),
                                         "candidate_test": round(best_test, 4)})
    report = {
        "target": "rag.router._TIEBREAK_SYSTEM",
        "train_size": len(train), "test_size": len(test),
        "baseline_train": result["baseline_score"], "best_train": result["best_train_score"],
        "baseline_test": round(base_test, 4), "candidate_test": round(best_test, 4),
        "improved_on_train": result["improved"],
        "generalizes": best_test > base_test,
        "safety_passed": safety_ok,
        "candidate_path": candidate_path,  # "" when nothing beat the baseline safely
        "candidates_tried": result["candidates_tried"], "history": result["history"],
    }
    os.makedirs(os.path.dirname(_REPORT), exist_ok=True)
    with open(_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("tie-break prompt optimization")
    print("  baseline: train {:.1%}  test {:.1%}".format(result["baseline_score"], base_test))
    print("  candidate: train {:.1%}  test {:.1%}".format(result["best_train_score"], best_test))
    if candidate_path:
        print("  candidate beats baseline and generalizes -> written to", candidate_path,
              "\n  (human-gated: review it, then promote by editing rag/router.py)")
    else:
        print("  no candidate beat the baseline on the held-out test; keeping the baseline")
    print("wrote", _REPORT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
