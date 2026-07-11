"""A reproducible, data-driven cost model for a served turn and a served user.

Every number in the cost doc comes from here, not from a hand-typed figure, so it regenerates:

  PYTHONPATH=. uv run python -m mlops.cost_model

The model is a transparent bottom-up estimate: token and usage assumptions are explicit constants
with their sources, and the per-turn cost is the sum of the stages a turn actually runs (routing,
retrieval, generation, and for voice the speech in and out). It is an ESTIMATE for a demo, not a
metered production bill; the assumptions are stated so anyone can change them and rerun.

Prices are list prices per the vendors as of 2026 (Groq, Cohere, ElevenLabs, Anthropic) and a
fully-loaded human-agent rate; verify against current pricing before quoting externally.
"""
from __future__ import annotations

import json
import os
from collections import Counter

# LLM prices per 1M tokens (input, output). Groq values match pipeline.answer._PRICES; the frontier
# rows are list prices used only for the comparison, the app itself is Groq-only.
LLM = {
    "llama-3.1-8b-instant": (0.05, 0.08),      # Groq
    "llama-3.3-70b-versatile": (0.59, 0.79),   # Groq
    "claude-sonnet-5": (3.00, 15.00),          # Anthropic, comparison only
    "claude-opus-4-8": (5.00, 25.00),          # Anthropic, comparison only
}

# Non-LLM unit costs, list prices, approximate.
COHERE_EMBED_PER_1M = 0.12        # embed-v4 input tokens
COHERE_RERANK_PER_SEARCH = 0.002  # rerank v3.5, about $2.00 per 1000 searches
GROQ_WHISPER_PER_MIN = 0.0007     # hosted whisper-large-v3, about $0.04 per audio hour
ELEVENLABS_PER_1K_CHARS = 0.10    # flash v2.5 list; cheaper on higher paid tiers
HUMAN_AGENT_PER_HOUR = 20.0       # fully-loaded customer-service agent

# Per-turn usage assumptions (explicit so they can be changed and rerun).
PROMPT_TOKENS = 1500     # retrieved context + history + system, a typical grounded turn
COMPLETION_TOKENS = 250  # a concise grounded answer
QUERY_EMBED_TOKENS = 25  # the shopper's query, embedded once for retrieval
SPOKEN_CHARS = 320       # a short spoken reply (voice strips to one or two sentences)
UTTERANCE_SECONDS = 10   # a spoken shopper turn transcribed
SESSION_TURNS = 8        # turns in a typical assistant session


def _gen_cost(model: str) -> float:
    pin, pout = LLM[model]
    return PROMPT_TOKENS / 1e6 * pin + COMPLETION_TOKENS / 1e6 * pout


def _retrieval_cost() -> float:
    return QUERY_EMBED_TOKENS / 1e6 * COHERE_EMBED_PER_1M + COHERE_RERANK_PER_SEARCH


def text_turn_cost(answer_model: str = "llama-3.3-70b-versatile") -> dict:
    """A text turn: routing (deterministic or a tiny 8B tie-break, rounding error), retrieval, and
    the answer generation. Returns the breakdown so the doc can show where the money goes."""
    routing = _gen_cost("llama-3.1-8b-instant") * 0.15  # only ~15% of turns pay the 8B tie-break
    retrieval = _retrieval_cost()
    generation = _gen_cost(answer_model)
    return {"routing": round(routing, 6), "retrieval": round(retrieval, 6),
            "generation": round(generation, 6),
            "total": round(routing + retrieval + generation, 6)}


def voice_turn_cost(answer_model: str = "llama-3.3-70b-versatile") -> dict:
    base = text_turn_cost(answer_model)
    stt = UTTERANCE_SECONDS / 60 * GROQ_WHISPER_PER_MIN
    tts = SPOKEN_CHARS / 1000 * ELEVENLABS_PER_1K_CHARS
    total = base["total"] + stt + tts
    return {**base, "stt": round(stt, 6), "tts": round(tts, 6), "total": round(total, 6)}


def human_turn_cost(minutes: float = 4.0) -> float:
    return round(minutes / 60 * HUMAN_AGENT_PER_HOUR, 4)


def build() -> dict:
    per_turn = {
        "text_8b": text_turn_cost("llama-3.1-8b-instant")["total"],
        "text_70b": text_turn_cost("llama-3.3-70b-versatile")["total"],
        "text_sonnet5": text_turn_cost("claude-sonnet-5")["total"],
        "text_opus": text_turn_cost("claude-opus-4-8")["total"],
        "voice_70b": voice_turn_cost("llama-3.3-70b-versatile")["total"],
        "human_agent": human_turn_cost(),
    }
    return {
        "assumptions": {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": COMPLETION_TOKENS,
                        "spoken_chars": SPOKEN_CHARS, "session_turns": SESSION_TURNS},
        "text_turn_breakdown_70b": text_turn_cost("llama-3.3-70b-versatile"),
        "voice_turn_breakdown_70b": voice_turn_cost("llama-3.3-70b-versatile"),
        "per_turn": {k: round(v, 6) for k, v in per_turn.items()},
        "per_session": {
            "text_70b": round(per_turn["text_70b"] * SESSION_TURNS, 4),
            "voice_70b": round(per_turn["voice_70b"] * SESSION_TURNS, 4),
            "human_agent": round(per_turn["human_agent"] * SESSION_TURNS, 4),
        },
        "ratios_vs_text_70b": {
            "sonnet5": round(per_turn["text_sonnet5"] / per_turn["text_70b"], 1),
            "opus": round(per_turn["text_opus"] / per_turn["text_70b"], 1),
            "human_agent": round(per_turn["human_agent"] / per_turn["text_70b"], 0),
        },
    }


# --- measured mode: reconcile the bottom-up estimate above against real metered traffic ---------
# The estimate uses hand-set token assumptions. Once turns are metered (streamed usage or the
# non-streamed cost path), this reads the trace log and reports p50/p95 tokens and cost per turn,
# side by side with the assumptions, flagging any that drift. That is what turns the 430x claim
# from a stated assumption into a measured number.

def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def measured_from_traces(trace_path: str | None = None) -> dict | None:
    """Per-turn tokens and cost measured from the metered trace log. Returns None when no turn has
    recorded usage yet, so the caller can say 'not measured' instead of inventing a zero."""
    path = trace_path or os.getenv("TRACE_PATH", "traces/requests.jsonl")
    if not os.path.exists(path):
        return None
    metered = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("cost") is not None and r.get("prompt_tokens"):
                metered.append(r)
    if not metered:
        return None
    pt = [r["prompt_tokens"] for r in metered]
    ct = [r.get("completion_tokens", 0) or 0 for r in metered]
    cost = [r["cost"] for r in metered]
    return {
        "n": len(metered),
        "prompt_tokens": {"p50": round(_percentile(pt, 0.5)), "p95": round(_percentile(pt, 0.95))},
        "completion_tokens": {"p50": round(_percentile(ct, 0.5)),
                              "p95": round(_percentile(ct, 0.95))},
        "cost_per_turn": {"p50": round(_percentile(cost, 0.5), 6),
                          "p95": round(_percentile(cost, 0.95), 6),
                          "mean": round(sum(cost) / len(cost), 6)},
        "models": dict(Counter(r.get("model") for r in metered)),
    }


def estimate_vs_measured(trace_path: str | None = None, tolerance: float = 0.25) -> dict:
    """The assumptions next to the measured p50, flagging any assumption off by more than tol."""
    m = measured_from_traces(trace_path)
    if m is None:
        return {"measured": None,
                "note": "no metered turns yet; run live traffic to populate traces"}

    def off_by(assumed: float, measured: float | None) -> float | None:
        return None if not measured else round((measured - assumed) / assumed, 3)

    pt_off = off_by(PROMPT_TOKENS, m["prompt_tokens"]["p50"])
    ct_off = off_by(COMPLETION_TOKENS, m["completion_tokens"]["p50"])
    flags = []
    if pt_off is not None and abs(pt_off) > tolerance:
        flags.append("prompt_tokens assumption {} off by {:+.0%}".format(PROMPT_TOKENS, pt_off))
    if ct_off is not None and abs(ct_off) > tolerance:
        flags.append("completion_tokens assumption {} off by {:+.0%}".format(
            COMPLETION_TOKENS, ct_off))
    return {
        "measured": m,
        "assumptions": {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": COMPLETION_TOKENS},
        "prompt_tokens_off_by": pt_off,
        "completion_tokens_off_by": ct_off,
        "flags": flags,
    }


if __name__ == "__main__":
    model = build()
    model["measured_vs_estimate"] = estimate_vs_measured()
    os.makedirs("evaluation/reports", exist_ok=True)
    with open("evaluation/reports/cost_model.json", "w", encoding="utf-8") as f:
        json.dump(model, f, indent=2)
    print(json.dumps(model, indent=2))
