# T3 Observability with Langfuse

**Theme:** T3 Observability. **Status:** Langfuse tracing done; dashboard visuals continue in T4.

## What I set out to do

See every answer end to end: for each turn, one trace with the LLM calls inside it, showing the
model, token counts, latency, and cost, viewable in a real observability UI. And do it without
breaking the offline engine or the tests.

## How I did it

The project's LLM calls go through the custom Groq adapter, not a LangChain LLM, so the LangChain
callback handler is the wrong tool (it also needs the full langchain package). Instead I trace with
Langfuse's `@observe` decorators and context spans, which capture the real generations:

- `adapters/observability.py` is the seam. It reads the Langfuse keys once at import. If they are
  set, `observe`, `request_span`, and the update/flush helpers delegate to Langfuse; if not, every
  one is a no-op passthrough (the decorator returns the function unchanged, the span is a null
  context), so there is zero overhead and no "client disabled" noise offline.
- `adapters/groq.py`: `generate` is decorated as a Langfuse generation and reports its model,
  prompt, answer, and input/output token counts. Langfuse derives cost from the model and tokens.
- `rag/graph.py` and `rag/supervisor.py`: each brain opens a `request_span` around the LangGraph
  invocation, so all the generations in a turn nest under one trace named `chat.linear` or
  `chat.agent`.
- `api/app.py`: the streaming chat handler wraps the whole turn in a `chat` request span and
  flushes at the end, so both the linear and the agent serving paths produce one trace per turn.
- The admin console gets the Langfuse URL (`/api/admin/domain`) so the backoffice can jump straight
  to the traces, next to the existing MLflow link.

## What I tested

- `make check` stays green (254+ tests). A new `tests/test_observability.py` proves the disabled
  path is a true no-op: the decorator returns the function unchanged, the span is a usable null
  context, and the update/flush calls never raise.
- Enabled-path smoke test with dummy keys against an unreachable server: the decorated call still
  returns its value and the app does not crash. Tracing is resilient. If Langfuse is down, the
  answer is unaffected and the trace is simply not delivered.
- I could not run against a real Langfuse server here (no keys), so the enabled path is verified for
  correct API use and resilience, not for the rendered trace. That is a keyed run.

## Set up

Put `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST` in `.env` (see `.env.example`)
and every turn shows up in Langfuse. Leave the keys empty to disable tracing entirely.

## What is next

T4 adds the richer backoffice dashboard visuals (the dbt test status, drift and cost charts) and
the guided user experience.
