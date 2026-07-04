# Fallbacks: how the demo stays up

Every external provider can rate-limit, expire a key, or go down. A hiring manager testing the demo
should never see a dead screen, so each provider has layers that degrade quietly instead of failing.
The rule is the same everywhere: try the best option, fall to the next on failure, and only ever
lose a little quality, never the whole turn.

## The chains

| Capability | Layer 1 (best) | Layer 2 | Layer 3 (always works) | What the user notices |
|-----------|----------------|---------|------------------------|-----------------------|
| **Embeddings** | Cohere trial key | Cohere paid key | Local BM25 sparse retrieval | Nothing, or slightly coarser matches on layer 3 |
| **Reranker** | Cohere rerank-v3.5 | Cohere paid key | Skip rerank, keep hybrid order | Slightly less sharp ranking |
| **LLM** | Groq Llama 3.3 70B | Groq Llama 3.1 8B | Static abstain or handoff message | A shorter answer, or a graceful "let me get a human" |
| **Voice out (TTS)** | ElevenLabs premium voice | Browser `speechSynthesis` | Text only | A more robotic voice, then plain text |
| **Voice in (STT)** | Groq Whisper | Browser `SpeechRecognition` | Type instead | Falls back to the keyboard |
| **Vector store** | Qdrant (retry on 429/503) | Same call retried with backoff | Graceful degraded reply | A brief pause; if the store stays down, an honest "trouble reaching results" note, not an error |
| **Knowledge graph** | Neo4j facts enrich the answer | Skip graph, answer from vectors | | Nothing; graph is additive |
| **Governed metrics** | DuckDB semantic layer | Skip metric, answer from vectors | | Nothing; metric is additive |

## How each one works

**Embeddings and reranker (Cohere).** One helper, `adapters/cohere._post(api_key, fallback_key)`,
handles both keys. A `429` (a capped trial or a per-minute limit) retries once on the fallback key.
A `401/403` (a bad or expired key) is remembered in `_dead_keys` for the rest of the process, so the
next request skips the doomed primary and goes straight to the paid key. If both keys are down, the
retrieval path drops to sparse-only (BM25), which needs no API at all, so search still returns.

**LLM (Groq).** `api/resilience.ResilientLLM` wraps the primary model with retries and a fallback
client. A failure that survives the retries falls to the smaller, cheaper model. If even that fails,
the turn returns the static abstain or human-handoff message rather than an error, so the shopper
always gets a sensible reply.

**Voice.** Voice is progressive enhancement. Text-to-speech uses ElevenLabs when a key is set
(`TTS_PROVIDER=elevenlabs`) and the browser's built-in `speechSynthesis` otherwise, so the assistant
always speaks. Speech-to-text uses Groq Whisper when configured and the browser's `SpeechRecognition`
otherwise. Both degrade to plain typed chat with no dead ends.

**Store, graph, and metrics.** `ResilientStore` retries a Qdrant `429/503` with backoff instead of
losing the turn. The graph and metric layers are additive: each is wrapped so a Neo4j or DuckDB
blip is caught and the answer falls back to the vector evidence it already has.

## Seeing it happen

The drift run in [mlops/experiments.md](mlops/experiments.md) logged
`Cohere primary key failed (auth), using the fallback key` mid-run and finished normally: the
embedding fallback firing on real traffic, with no interruption to the result.

Defaults are offline. A fresh checkout with no keys runs entirely on the local fakes (fake embedder,
in-memory store, browser voice), so the whole thing boots and the tests pass with nothing installed.
