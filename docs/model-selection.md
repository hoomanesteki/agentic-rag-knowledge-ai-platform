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

## Embeddings and rerank: Voyage

- **Why:** `voyage-3-large` is at or near the top of the retrieval benchmarks and is multilingual
  (this project is English and French), which is exactly what a hybrid retriever needs. `rerank-2.5`
  cleans up the candidate set before the model ever sees it. Both are cheap per call and one account
  covers both.
- **Alternative considered:** OpenAI `text-embedding-3-small` is cheaper but scores lower on
  multilingual retrieval; Cohere is comparable but adds a third vendor. Voyage won on quality per
  dollar for a bilingual RAG index.

## Speech to text: Groq hosted Whisper

- **Why:** the voice feature needs one short transcription per clip. Groq hosts `whisper-large-v3`
  behind the same account and key as the app LLM, so there is no extra vendor and the latency is low.
  The browser's Web Speech API is the offline fallback.

## Judge for RAGAS: a separate small model

- **Why:** an answer-quality judge should be independent of the model that wrote the answer, so a
  weak answer is not graded by the same weak model. `JUDGE_MODEL` points at a small, cheap Groq
  model; leaving it empty falls back to the app LLM for offline runs.

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
