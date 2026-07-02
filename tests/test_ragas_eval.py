"""M8.2 RAGAS-style eval: the four metrics and the by-language scorecard, offline with a fake
judge (the real judge is Groq)."""
from adapters.base import LLMResult
from evaluation.ragas_eval import evaluate_ragas, ragas_scores


class FixedJudge:
    def __init__(self, score="0.8"):
        self.score = score

    def generate(self, prompt, *, system=None, max_tokens=512):
        return LLMResult(text='{{"score": {}}}'.format(self.score), prompt_tokens=1,
                         completion_tokens=1, model="fake")


def test_all_four_metrics_when_inputs_present():
    item = {"question": "q", "answer": "a", "contexts": ["c1", "c2"], "ground_truth": "gt",
            "lang": "en"}
    scores = ragas_scores(item, FixedJudge("0.8"))
    # both chunks judged relevant -> rank-weighted precision is 1.0; the rest reflect the judge
    assert scores == {"faithfulness": 0.8, "answer_relevance": 0.8,
                      "context_precision": 1.0, "context_recall": 0.8}


def test_abstention_skips_faithfulness_and_relevance():
    item = {"question": "q", "answer": "I do not have enough information", "contexts": ["c"],
            "abstained": True}
    scores = ragas_scores(item, FixedJudge("0.9"))
    # an abstention has no claims and does not "address" the question, so both are not applicable
    assert scores["faithfulness"] is None and scores["answer_relevance"] is None


def test_irrelevant_chunks_score_zero_precision():
    item = {"question": "q", "answer": "a", "contexts": ["c1", "c2"]}
    scores = ragas_scores(item, FixedJudge("0.1"))   # every chunk judged not relevant
    assert scores["context_precision"] == 0.0


def test_missing_inputs_skip_metrics():
    # no contexts and no answer -> only the metrics that apply (none here) are scored
    scores = ragas_scores({"question": "q", "answer": "", "contexts": []}, FixedJudge())
    assert all(v is None for v in scores.values())
    # an answer with no reference -> recall skipped, relevance still scored
    scores = ragas_scores({"question": "q", "answer": "a", "contexts": ["c"]}, FixedJudge("0.9"))
    assert scores["answer_relevance"] == 0.9 and scores["context_recall"] is None


def test_malformed_judge_output_is_skipped():
    class Junk:
        def generate(self, prompt, *, system=None, max_tokens=512):
            return LLMResult(text="not json", prompt_tokens=1, completion_tokens=1, model="fake")

    scores = ragas_scores({"question": "q", "answer": "a", "contexts": ["c"]}, Junk())
    assert scores["answer_relevance"] is None


def test_scorecard_by_language():
    items = [
        {"question": "q1", "answer": "a1", "contexts": ["c"], "ground_truth": "g", "lang": "en"},
        {"question": "q2", "answer": "", "contexts": [], "lang": "fr"},  # abstain, no metrics
    ]
    card = evaluate_ragas(items, FixedJudge("0.7"))
    assert card["count"] == 2
    assert card["overall"]["faithfulness"] == 0.7
    # fr had nothing to score, so it gets no bucket
    assert "en" in card["by_language"] and "fr" not in card["by_language"]
