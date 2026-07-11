"""M8.3 CI eval gate: passes on the recorded fixtures, blocks a seeded regression."""
from evaluation.ci_gate import load_gate, run_gate

_GATE = "evaluation/fixtures/gate.json"


def test_gate_passes_on_recorded_fixtures(tmp_path):
    result = run_gate(load_gate(_GATE), trace_path=str(tmp_path / "t.jsonl"))
    assert result["passed"] and result["score"] == 1.0  # default min_score is 1.0


def test_gate_blocks_a_single_dropped_doc(tmp_path):
    gate = load_gate(_GATE)
    # drop ONE corpus doc so exactly one retrieves-fixture fails: the strict gate must still block
    broken = {"corpus": gate["corpus"][1:], "fixtures": gate["fixtures"]}
    result = run_gate(broken, trace_path=str(tmp_path / "t.jsonl"))
    assert not result["passed"] and result["score"] < 1.0  # any regression blocks


def test_gate_has_a_populated_abstain_slice():
    # false-premise, third-party, and unanswerable questions must be CI-gated to abstain, so an
    # un-abstain change (a heuristic that starts answering them) cannot land silently
    abstain = [f for f in load_gate(_GATE)["fixtures"] if f["expect"] == "abstain"]
    assert len(abstain) >= 4
