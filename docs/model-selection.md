# Model selection

The rule I used everywhere: pick the model that gives a good result at low cost and low latency, not
the biggest or most expensive one. RAG does most of the heavy lifting with retrieval and grounding,
so the model's job is careful synthesis over evidence I already found, not open-ended reasoning. That
lets me use smaller, faster, cheaper models without losing answer quality.

Every model sits behind an adapter (`adapters/`), so any of these is a config swap, not a rewrite.
Nothing runs locally: this is a laptop with limited disk, and hosted inference is cheaper than the
electricity and disk a local 70B would cost.

## App LLM: Groq (Llama 3.x)

- **Why:** Groq's LPU serves tokens faster than any GPU host I tested, which matters for a streaming
  chat where time-to-first-token is what the user feels. Llama-3.3-70B is strong enough for grounded
  synthesis and citation, and Llama-3.1-8B-instant handles the small jobs (query rewrite, metric
  slot-fill, routing) for a fraction of the cost.
- **Two-tier:** small model for the cheap, frequent calls; large model only for the final answer.
  Most tokens go through the cheap model.
- **Cost and latency:** among the lowest available for open-weight models of this quality. A typical
  answer is well under a cent.

## Embeddings and rerank: Cohere

- **Why:** `embed-v4.0` is a top multilingual embedder (this project is English and French), which
  is exactly what a hybrid retriever needs, and `rerank-v3.5` cleans up the candidate set before the
  model ever sees it. Both are cheap per call and **one account covers both**, so the retrieval
  stack is a single vendor. Cohere also has generous free/trial limits, which suits a portfolio
  demo, and a resilient wrapper (`api/resilience.py`) caches and retries so a rate limit degrades
  gracefully instead of breaking the chat.
- **Migrated from Voyage:** the project started on Voyage `voyage-3-large` + `rerank-2.5` (also
  strong and multilingual), but hit tight requests-per-minute limits on the free tier mid-demo.
  Cohere gave the same retrieval quality with headroom to actually run the live demo, so the whole
  stack moved over with a config swap (adapter seam) and no engine change. Embeddings are 1536-dim.
- **Alternative considered:** OpenAI `text-embedding-3-small` is cheaper but scores lower on
  multilingual retrieval and does not bundle a reranker.

## Speech to text: Groq hosted Whisper

- **Why:** the voice feature needs one short transcription per clip. Groq hosts `whisper-large-v3`
  behind the same account and key as the app LLM, so there is no extra vendor and the latency is low.
  The browser's Web Speech API is the offline fallback.

## Text to speech (voice): ElevenLabs, browser fallback

- **Why:** a sales-agent demo lives on sounding human. ElevenLabs `eleven_flash_v2_5` is near the top
  for realism at ~75 ms latency, which is what makes the assistant feel like a person rather than a
  robot. The key stays server-side (the browser calls `/api/tts`), and the assistant and the human
  specialist get distinct voices via a persona flag.
- **Cost control and fallback:** free-tier premade voices only (a config default that avoids the
  paid-only Voice Library), and when `TTS_PROVIDER=none` or the key is missing or a call fails, the
  endpoint returns 204 and the browser's built-in `speechSynthesis` speaks instead. Voice never
  hard-fails; it degrades to free.

## Judge for RAGAS: a separate small model

- **Why:** an answer-quality judge should be independent of the model that wrote the answer, so a
  weak answer is not graded by the same weak model. `JUDGE_MODEL` points at a small, cheap Groq
  model; leaving it empty falls back to the app LLM for offline runs.

## How the choice is evidenced (not just asserted)

The point of a model choice is that you can show it was right, not just claim it. Every request
writes a trace (`traces/requests.jsonl`) with tier, grounding, latency, and cost, and every eval
run logs to MLflow. So the decision is backed by numbers you can see live:

- **Back-office Health view** (`/admin/health`): p95 latency, throughput, error rate, average cost
  per turn, and grounding, broken out by language, with a retrieval-quality trend. This is where
  "Groq is fast enough / an answer costs well under a cent" stops being a claim.
- **Answer-quality gate** (`make gate`, `make ragas`): the RAGAS scores (faithfulness,
  answer-relevancy, context precision/recall) and a golden fixture set that CI blocks on, so a model
  or prompt swap that drops quality fails the build.
- **Ablation** (`make ablation`): dense-only vs hybrid vs hybrid+rerank, so the rerank vendor earns
  its place with a measured lift, not a hunch.

To reproduce the comparison for a new model, point the provider env vars at it, run `make ragas`
and `make gate`, and read the Health view; the delta in grounding, latency, and cost is the answer.

## What I did not choose, and why

- **A local model (Ollama, ONNX):** disk and memory cost on a laptop, slower, and no quality gain at
  this size. The adapter seam is there if a private-data deployment ever needs it.
- **AWS Bedrock:** a good enterprise-governance story, but more setup and higher latency than Groq
  for the same open models. Worth it when the deployment is already AWS-native; not for this demo.
- **A single frontier model for everything:** overkill and expensive. Reasoning-heavy frontier models
  add cost and latency that grounded RAG synthesis does not need.

## Swapping

Set the provider env vars and go. The engine and tests run fully offline on deterministic fakes, so
none of this is required to develop or to pass CI: `LLM_PROVIDER`, `EMBED_PROVIDER`,
`RERANK_PROVIDER`, `TRANSCRIBE_PROVIDER`, and the matching model names in `.env`.
