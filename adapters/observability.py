"""Optional Langfuse tracing for the LLM path and the LangGraph brain.

Enabled only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set; otherwise every hook is a
no-op passthrough, so the offline engine and the tests are unaffected and there is no noisy
"client disabled" logging. The project's LLM calls go through the custom Groq adapter (not a
LangChain LLM), so tracing is done with @observe and context spans rather than the LangChain
callback handler: a request opens one root span, and every LLM generation nests under it with its
model, token counts, and latency. That gives one trace per turn, viewable in the Langfuse UI.
"""
from __future__ import annotations

import os
from contextlib import nullcontext

from dotenv import load_dotenv

load_dotenv()

_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))

if _ENABLED:  # import the SDK only when configured, so a base install without keys stays quiet
    from langfuse import get_client
    from langfuse import observe as _lf_observe


def enabled() -> bool:
    return _ENABLED


def observe(func=None, *, name=None, as_type=None):
    """Decorator: trace the wrapped function as a Langfuse span or generation when enabled, else
    return it unchanged (zero overhead, no logging). Usable as @observe or @observe(as_type=...)."""
    def wrap(fn):
        if not _ENABLED:
            return fn
        return _lf_observe(name=name, as_type=as_type)(fn)
    return wrap(func) if callable(func) else wrap


def request_span(name: str, **kwargs):
    """Context manager for a request's root span, so the LLM generations inside it group into one
    trace. A plain nullcontext when disabled."""
    if not _ENABLED:
        return nullcontext()
    return get_client().start_as_current_observation(name=name, as_type="span", **kwargs)


def update_generation(**kwargs) -> None:
    """Attach model, input, output, and usage to the current generation span (no-op if disabled or
    called outside a generation)."""
    if _ENABLED:
        get_client().update_current_generation(**kwargs)


def update_span(**kwargs) -> None:
    if _ENABLED:
        get_client().update_current_span(**kwargs)


def flush() -> None:
    """Send buffered traces. Call at the end of a request so nothing is lost when a run exits."""
    if _ENABLED:
        get_client().flush()
