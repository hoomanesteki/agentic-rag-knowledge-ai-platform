"""GEPA-style reflection in the prompt optimizer's proposer: it diagnoses the failure pattern in
natural language BEFORE rewriting, and feeds that diagnosis into the rewrite, which is what makes
reflective prompt evolution generalize where a blind rewrite flatlines on held-out data."""
from adapters.base import LLMResult
from scripts.run_prompt_opt import _propose


class _RecordingProposer:
    def __init__(self, diagnosis: str):
        self.prompts: list[str] = []
        self.diagnosis = diagnosis

    def generate(self, prompt, *, system=None, max_tokens=512):
        self.prompts.append(prompt)
        is_diagnosis = "rewrite the prompt yet" in prompt.lower()
        text = self.diagnosis if is_diagnosis else "NEW ROUTER PROMPT {\"lane\": L}"
        return LLMResult(text=text, prompt_tokens=1, completion_tokens=1, model="fake")


def test_propose_diagnoses_before_and_feeds_the_rewrite():
    rec = _RecordingProposer(diagnosis="DIAG: it guesses a specialist lane on vague questions")
    variants = _propose("CURRENT PROMPT",
                        [{"query": "what is your return policy", "want": "answers",
                          "got": "stylist"}], 2, rec)
    # the first call is the reflection step, not a rewrite
    assert "diagnose" in rec.prompts[0].lower()
    # every rewrite ask after it carries the diagnosis, so the fix is informed by the pattern
    assert all("DIAG: it guesses a specialist lane" in p for p in rec.prompts[1:])
    assert len(variants) == 2


def test_propose_survives_a_failed_diagnosis():
    class _NoDiagnose(_RecordingProposer):
        def generate(self, prompt, *, system=None, max_tokens=512):
            self.prompts.append(prompt)
            if "rewrite the prompt yet" in prompt.lower():
                raise RuntimeError("diagnosis call failed")
            return LLMResult(text="NEW PROMPT {\"lane\": L}", prompt_tokens=1, completion_tokens=1,
                             model="fake")

    rec = _NoDiagnose(diagnosis="")
    # a failed diagnosis must not break proposing: it just proceeds without the diagnosis block
    assert len(_propose("CURRENT", [{"query": "x", "want": "answers", "got": "care"}], 1, rec)) == 1
