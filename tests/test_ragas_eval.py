"""M8.2 RAGAS eval: the faithful RAGAS algorithms (statement decompose/verify, question
generation + embedding similarity, per-chunk precision), offline with a fake judge and the fake
embedder (the real judge is Groq)."""
from adapters.base import LLMResult
from adapters.factory import make_embedder
from evaluation.ragas_eval import evaluate_ragas, faithfulness, ragas_scores


class RagasFakeLLM:
    """Returns the structured output each RAGAS step expects, keyed off the judge system prompt."""

    def generate(self, prompt, *, system=None, max_tokens=512):
        s = system or ""
        if "atomic factual statements" in s:
            text = '["claim one", "claim two"]'
        elif "supported by the context" in s or "attributed to" in s:
            text = "[true, true]"
        elif "Generate three questions" in s:
            text = '["what is one", "what is two", "what is three"]'
        elif "which retrieved chunks were useful" in s:
            text = "[true, true]"       # both chunks useful (booleans, one per chunk)
        else:
            text = "[]"
        return LLMResult(text=text, prompt_tokens=1, completion_tokens=1, model="fake")


def test_faithful_ragas_metrics():
    item = {"question": "q", "answer": "a", "contexts": ["c1", "c2"], "ground_truth": "gt",
            "lang": "en"}
    scores = ragas_scores(item, RagasFakeLLM(), make_embedder("fake"))
    assert scores["faithfulness"] == 1.0        # 2/2 statements supported by context
    assert scores["context_recall"] == 1.0      # 2/2 reference statements attributable
    assert scores["context_precision"] == 1.0   # both chunks judged relevant, rank-weighted
    assert 0.0 <= scores["answer_relevance"] <= 1.0  # generated-question embedding similarity


def test_faithfulness_none_on_verdict_length_mismatch():
    class Mismatch:
        def generate(self, prompt, *, system=None, max_tokens=512):
            s = system or ""
            text = '["a", "b", "c"]' if "atomic factual statements" in s else "[true]"
            return LLMResult(text=text, prompt_tokens=1, completion_tokens=1, model="fake")

    assert faithfulness(Mismatch(), "answer", "context") is None  # 3 statements, 1 verdict


def test_verdicts_normalize_non_bool_values():
    # a judge that emits 1/0 (not true/false) must be counted, not read as all-unsupported 0.0
    class IntVerdicts:
        def generate(self, prompt, *, system=None, max_tokens=512):
            s = system or ""
            if "atomic factual statements" in s:
                text = '["a", "b"]'
            elif "supported by the context" in s:
                text = "[1, 0]"           # one supported, one not
            else:
                text = "[]"
            return LLMResult(text=text, prompt_tokens=1, completion_tokens=1, model="fake")

    assert faithfulness(IntVerdicts(), "answer", "context") == 0.5


def test_abstention_skips_faithfulness_and_relevance():
    item = {"question": "q", "answer": "I do not have enough information", "contexts": ["c"],
            "abstained": True}
    scores = ragas_scores(item, RagasFakeLLM(), make_embedder("fake"))
    assert scores["faithfulness"] is None and scores["answer_relevance"] is None


def test_malformed_judge_output_is_skipped():
    class Junk:
        def generate(self, prompt, *, system=None, max_tokens=512):
            return LLMResult(text="not json", prompt_tokens=1, completion_tokens=1, model="fake")

    scores = ragas_scores({"question": "q", "answer": "a", "contexts": ["c"]}, Junk(),
                          make_embedder("fake"))
    assert scores["faithfulness"] is None and scores["context_precision"] is None


def test_scorecard_by_language():
    items = [
        {"question": "q1", "answer": "a1", "contexts": ["c"], "ground_truth": "g", "lang": "en"},
        {"question": "q2", "answer": "", "contexts": [], "lang": "fr"},  # abstain, no metrics
    ]
    card = evaluate_ragas(items, RagasFakeLLM(), make_embedder("fake"))
    assert card["count"] == 2
    assert card["overall"]["faithfulness"] == 1.0
    # fr had nothing to score, so it gets no bucket
    assert "en" in card["by_language"] and "fr" not in card["by_language"]
