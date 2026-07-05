"""A lightweight OPRO/APE prompt-optimization loop: propose, score, keep the best, repeat.

This is the industry-standard idea (an LLM proposes better prompts, a metric selects them) without
the weight of a framework like DSPy: our prompts are hand tuned and safety critical, so we keep the
loop small and the control obvious. The loop NEVER writes a winner into the code. It surfaces a
candidate for a human to review and promote, because a system that rewrites its own production
prompts invites drift and reward hacking. See docs/prompt-optimization.md.

optimize_prompt is generic. The caller supplies:
  evaluate(prompt) -> (score: float, misrouted: list)   how good is this prompt, and its failures
  propose(prompt, misrouted, n) -> list[str]            n candidate rewrites from the failures
  safety(prompt) -> bool                                 a hard gate a candidate must pass
so the same loop optimizes any prompt against any ground-truth metric.
"""
from __future__ import annotations

import json
import os


def optimize_prompt(baseline: str, *, evaluate, propose, safety=None, rounds: int = 2,
                    n_candidates: int = 3, margin: float = 0.005, sample: int = 12) -> dict:
    """Run the loop on a TRAIN metric (evaluate). Returns a result dict; the caller reports the
    winner on a held-out test set separately, so an improvement has to generalize, not memorize."""
    best = baseline
    best_score, best_fail = evaluate(baseline)
    history = [{"round": 0, "candidate": "baseline", "score": round(best_score, 4)}]
    for r in range(1, rounds + 1):
        candidates = propose(best, best_fail[:sample], n_candidates) or []
        for i, cand in enumerate(candidates):
            if not cand or cand == best:
                continue
            if safety is not None and not safety(cand):
                history.append({"round": r, "candidate": i, "rejected": "safety"})
                continue
            score, fail = evaluate(cand)
            history.append({"round": r, "candidate": i, "score": round(score, 4)})
            if score > best_score + margin:
                best, best_score, best_fail = cand, score, fail
    return {
        "baseline_score": round(history[0]["score"], 4),
        "best_train_score": round(best_score, 4),
        "improved": best != baseline,
        "best_prompt": best,
        "rounds": rounds,
        "candidates_tried": sum(1 for h in history if isinstance(h.get("candidate"), int)),
        "history": history,
    }


def save_candidate(name: str, prompt: str, meta: dict, *,
                   root: str = "mlops/prompt_registry") -> str:
    """Write a proposed prompt as a review artifact (status: proposed). Promotion is a human editing
    the source; this only records what the loop found, with the numbers that justify it."""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "{}.candidate.json".format(name))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"name": name, "status": "proposed", "prompt": prompt, **meta}, f, indent=2)
    return path
