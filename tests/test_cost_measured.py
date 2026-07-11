"""The measured cost mode reconciles the bottom-up estimate against real metered traffic, so the
cost claim is a measured number, not only an assumption. These pin that it reads metered traces,
computes percentiles, flags a drifted assumption, and stays honest (None) when nothing metered."""
import json

from mlops.cost_model import estimate_vs_measured, measured_from_traces


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_no_metered_turns_reports_none_not_zero(tmp_path):
    p = tmp_path / "empty.jsonl"
    # lines that predate metering (cost null) must not be counted as free turns
    _write(p, [{"tier": "auto", "cost": None, "prompt_tokens": None},
               {"tier": "abstain", "cost": 0.0, "prompt_tokens": 0}])
    assert measured_from_traces(str(p)) is None
    assert estimate_vs_measured(str(p))["measured"] is None


def test_measures_percentiles_from_metered_traces(tmp_path):
    p = tmp_path / "t.jsonl"
    rows = [{"tier": "auto", "model": "llama-3.3-70b-versatile", "prompt_tokens": 1000 + 100 * i,
             "completion_tokens": 200, "cost": 0.001 * (i + 1)} for i in range(11)]
    _write(p, rows)
    m = measured_from_traces(str(p))
    assert m["n"] == 11
    assert m["prompt_tokens"]["p50"] == 1500  # median of 1000..2000 step 100
    assert m["prompt_tokens"]["p95"] > m["prompt_tokens"]["p50"]
    assert m["models"] == {"llama-3.3-70b-versatile": 11}


def test_flags_an_assumption_that_drifted(tmp_path):
    p = tmp_path / "t.jsonl"
    # measured prompt tokens ~3000, double the 1500 assumption -> must flag (> 25% off)
    rows = [{"tier": "auto", "model": "m", "prompt_tokens": 3000, "completion_tokens": 250,
             "cost": 0.003} for _ in range(5)]
    _write(p, rows)
    cmp = estimate_vs_measured(str(p))
    assert cmp["prompt_tokens_off_by"] == 1.0  # +100%
    assert cmp["flags"] and "prompt_tokens" in cmp["flags"][0]


def test_close_assumptions_do_not_flag(tmp_path):
    p = tmp_path / "t.jsonl"
    rows = [{"tier": "auto", "model": "m", "prompt_tokens": 1550, "completion_tokens": 260,
             "cost": 0.003} for _ in range(5)]
    _write(p, rows)
    assert estimate_vs_measured(str(p))["flags"] == []
