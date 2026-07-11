"""The per-job model map: which model does which job, and the named reason for each choice.

An assistant that spends well uses the cheapest model that clears the bar for each JOB, not one
model for everything. This module is the single, documented source of that decision, so every model
choice is defensible with a reason and a source rather than an implicit default buried in env vars.
Each row resolves from an environment variable, so a change is one line of config, gated by the eval
harness before it ships.

Prices are Groq list prices per 1M tokens (input/output). The app stays Groq-only by decision; the
frontier rows on cost.qmd exist only as a measured comparison, not a dependency.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelJob:
    job: str          # what the model is for
    env: str          # the environment variable that selects it
    default: str      # the model used when the env var is unset
    reason: str       # why this model is right for this job
    source: str       # the research/benchmark backing the choice


# The canonical map. Ordered cheapest-job first. Read by resolve() and published on the site.
MODEL_MAP: tuple[ModelJob, ...] = (
    ModelJob(
        job="Routing tie-break, metric slot-fill, typo repair",
        env="GROQ_MODEL_SMALL", default="llama-3.1-8b-instant",
        reason="A classification job with a tiny output; the 8B is accurate here, and the routing "
               "eval shows a 70B tie-break is no better, so a bigger model would buy nothing.",
        source="routing_eval.json (70B 85.6% vs 8B 85.9%); RouteLLM cost-routing policy."),
    ModelJob(
        job="Answers lane draft (policy / FAQ), behind the grounding gate",
        env="GROQ_MODEL_ANSWERS_DRAFT", default="llama-3.1-8b-instant",
        reason="Grounded FAQ answers are easy: draft on the small model, accept when the existing "
               "grounding + confidence gate clears, else regenerate once on the workhorse.",
        source="FrugalGPT LLM cascades (Chen, Zaharia, Zou)."),
    ModelJob(
        job="Stylist / complaint / care generation (the workhorse)",
        env="GROQ_MODEL_LARGE", default="llama-3.3-70b-versatile",
        reason="Tone-sensitive grounded generation. gpt-oss-120b is ~70% cheaper ($0.15/$0.60 vs "
               "$0.59/$0.79) with a 0.5x prompt cache, but a live RAGAS A/B HELD the swap: aggregate "
               "quality flat and faithfulness regressed 0.023 on the hallucination guard. Revisit "
               "after the Phase 2 online faithfulness net catches the extra ungrounding.",
        source="evaluation/reports/gpt_oss_swap_ab.json (live A/B, judge fixed to llama-3.3-70b)."),
    ModelJob(
        job="RAGAS answer-quality judge",
        env="JUDGE_MODEL", default="llama-3.3-70b-versatile",
        reason="A judge must be a different family from the generator, or it grades its own style "
               "up. Once generation moves to gpt-oss-120b, the freed llama-3.3-70b judges free.",
        source="Self-Preference Bias in LLM-as-a-Judge (arXiv:2410.21819)."),
    ModelJob(
        job="Embeddings (dense retrieval)",
        env="EMBED_MODEL", default="embed-v4.0",
        reason="Strong multilingual retrieval; the reranker does the precision work downstream.",
        source="Cohere embed-v4 retrieval benchmarks."),
    ModelJob(
        job="Rerank (precision), skip-gated",
        env="RERANK_MODEL", default="rerank-v3.5",
        reason="A cross-encoder is the single largest per-turn cost line, so it is skipped when it "
               "cannot change the answer (own-order lookups, governed-metric turns, clean score "
               "margins) and paid only where it earns its latency.",
        source="When cross-encoders earn latency (rerank-strategies research)."),
)


def resolve(job_env: str, default: str) -> str:
    """The model actually selected for a job, from its env var or the documented default."""
    return os.getenv(job_env, default)


def active_map() -> list[dict]:
    """The map with the currently-resolved model per row, for the site table and audit."""
    return [{"job": m.job, "env": m.env, "model": resolve(m.env, m.default),
             "reason": m.reason, "source": m.source} for m in MODEL_MAP]
