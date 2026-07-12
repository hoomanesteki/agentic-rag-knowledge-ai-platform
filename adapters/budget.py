"""A per-turn budget that turns "never loops, never burns tokens" from a hope into an enforced,
queryable fact.

The budget is CHECKED before each model call and CHARGED after. A BudgetedLLM wraps the real LLM
so every generate/stream on the omni path passes through one choke point sharing one TurnBudget.
On breach it raises BudgetExceeded and the orchestrator finishes with the answers already in hand
(or the zero-LLM escalation) instead of looping.

Why check-before-call and not a recursion limit: LangGraph's recursion_limit and a bare step cap
only fire AFTER the wasted steps, and a step cap cannot express a token or dollar or wall-clock
ceiling. This checks before the spend and caps calls, tokens, cost, and elapsed time together.
"""
from __future__ import annotations

import time

# Sized for a demo turn: a single answer is 1-2 calls; a 3-clause multi-task with one reroute is at
# most ~5. The ceilings sit above that and below anything that would read as a loop or a runaway.
DEFAULT_MAX_CALLS = 8
DEFAULT_MAX_TOKENS = 24_000
DEFAULT_MAX_USD = 0.05
DEFAULT_MAX_SECONDS = 30.0


class BudgetExceeded(Exception):
    """Raised before a model call a turn's budget can no longer afford. Carries the tripped rule."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class TurnBudget:
    """One turn's ceiling on model calls, tokens, cost, and wall-clock time."""

    def __init__(self, *, max_calls: int = DEFAULT_MAX_CALLS, max_tokens: int = DEFAULT_MAX_TOKENS,
                 max_usd: float = DEFAULT_MAX_USD, max_seconds: float = DEFAULT_MAX_SECONDS,
                 clock=time.monotonic) -> None:
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.max_usd = max_usd
        self.max_seconds = max_seconds
        self._clock = clock
        self.started = clock()
        self.calls = 0
        self.tokens = 0
        self.usd = 0.0

    def elapsed(self) -> float:
        return self._clock() - self.started

    def check(self) -> None:
        """Raise BudgetExceeded if the next model call cannot be afforded."""
        if self.calls >= self.max_calls:
            raise BudgetExceeded("max_calls")
        if self.tokens >= self.max_tokens:
            raise BudgetExceeded("max_tokens")
        if self.usd >= self.max_usd:
            raise BudgetExceeded("max_usd")
        if self.elapsed() >= self.max_seconds:
            raise BudgetExceeded("deadline")

    def charge(self, *, tokens: int = 0, usd: float = 0.0) -> None:
        self.calls += 1
        self.tokens += max(tokens, 0)
        self.usd += max(usd, 0.0)

    def snapshot(self) -> dict:
        """The turn's spend, for logging into the final event and the trace so it is queryable. The
        usd figure tracks Groq generation spend, the ceiling this budget enforces; retrieval (Cohere
        embed and rerank) is not charged here, and the full-turn cost lives in the cost model."""
        return {"calls": self.calls, "tokens": self.tokens, "usd": round(self.usd, 6),
                "elapsed_ms": round(self.elapsed() * 1000, 1),
                "limits": {"calls": self.max_calls, "tokens": self.max_tokens,
                           "usd": self.max_usd, "seconds": self.max_seconds}}


def _estimate(model, prompt_tokens: int, completion_tokens: int) -> float:
    # lazy import: pipeline.answer imports adapters, so a module-load import would be circular
    try:
        from pipeline.answer import _estimate_cost
        return _estimate_cost(model, prompt_tokens or 0, completion_tokens or 0) or 0.0
    except Exception:
        return 0.0


class BudgetedLLM:
    """Wrap an LLM so every call is budget-checked before it runs and charged after. Exposes the
    same generate/stream/model surface the router and pipeline read, and delegates anything else to
    the wrapped client."""

    def __init__(self, llm, budget: TurnBudget) -> None:
        self._llm = llm
        self.budget = budget
        self.model = getattr(llm, "model", None)

    def generate(self, prompt, *, system=None, max_tokens: int = 512):
        self.budget.check()
        res = self._llm.generate(prompt, system=system, max_tokens=max_tokens)
        self.budget.charge(
            tokens=(getattr(res, "prompt_tokens", 0) or 0)
            + (getattr(res, "completion_tokens", 0) or 0),
            usd=_estimate(getattr(res, "model", None), getattr(res, "prompt_tokens", 0),
                          getattr(res, "completion_tokens", 0)))
        return res

    def stream(self, prompt, *, system=None, max_tokens: int = 512, usage_out=None):
        self.budget.check()
        local: dict = {}
        for piece in self._llm.stream(prompt, system=system, max_tokens=max_tokens,
                                      usage_out=local):
            yield piece
        if usage_out is not None:
            usage_out.update(local)
        self.budget.charge(
            tokens=(local.get("prompt_tokens", 0) or 0) + (local.get("completion_tokens", 0) or 0),
            usd=_estimate(local.get("model"), local.get("prompt_tokens", 0),
                          local.get("completion_tokens", 0)))

    def __getattr__(self, name):
        # reached only when normal lookup fails; delegate to the wrapped llm without recursing
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "_llm"), name)
