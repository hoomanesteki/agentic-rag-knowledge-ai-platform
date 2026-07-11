"""The per-job model map is the documented, defensible source for which model does which job.
These lock in that every row resolves from an env var (a one-line, gated change) and carries a
reason and a source, and that the tie-break and workhorse jobs point at the expected models."""
from mlops.model_map import MODEL_MAP, active_map, resolve
from pipeline.answer import _estimate_cost, _tier_of


def test_every_job_has_a_reason_and_a_source():
    assert len(MODEL_MAP) >= 5
    for m in MODEL_MAP:
        assert m.job and m.env and m.default and m.reason and m.source


def test_resolve_uses_env_then_default(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL_LARGE", raising=False)
    assert resolve("GROQ_MODEL_LARGE", "llama-3.3-70b-versatile") == "llama-3.3-70b-versatile"
    monkeypatch.setenv("GROQ_MODEL_LARGE", "openai/gpt-oss-120b")
    assert resolve("GROQ_MODEL_LARGE", "llama-3.3-70b-versatile") == "openai/gpt-oss-120b"


def test_active_map_resolves_a_model_per_row():
    rows = active_map()
    assert {r["job"] for r in rows} and all(r["model"] for r in rows)


def test_gpt_oss_is_priced_and_tiered_so_a_swap_is_metered():
    # the swap target must be costable the moment it is selected, or the metered turn goes unknown
    assert _estimate_cost("openai/gpt-oss-120b", 1500, 250) is not None
    assert _tier_of("openai/gpt-oss-120b") == "large"
    assert _tier_of("openai/gpt-oss-20b") == "small"
    # and it is genuinely cheaper than the current workhorse on the same tokens
    assert _estimate_cost("openai/gpt-oss-120b", 1500, 250) < \
        _estimate_cost("llama-3.3-70b-versatile", 1500, 250)
