"""M8.3 drift monitors: PSI, query-embedding drift, and the five-monitor report by language."""
from adapters.factory import make_embedder
from mlops.drift import drift_report, embedding_drift, psi


def test_psi_is_low_for_same_and_high_for_shifted():
    reference = [i / 100 for i in range(100)]  # 0.00 .. 0.99
    assert psi(reference, reference) < 0.01           # identical distribution -> no drift
    shifted = [x + 0.5 for x in reference]
    assert psi(reference, shifted) > 0.2              # shifted -> drift


def test_psi_none_with_too_little_data():
    assert psi([1.0, 2.0], [1.0]) is None


def test_psi_constant_reference_catches_downward_shift():
    reference = [0.9] * 20  # pinned confidence: quantile binning alone would miss a drop
    assert psi(reference, [0.9] * 20) == 0.0     # no change
    assert psi(reference, [0.4] * 20) > 0.2      # a real downward shift is caught by the fallback


def test_embedding_drift_detects_topic_shift():
    embedder = make_embedder("fake")
    reference = ["how long does shipping take", "when will my order arrive"]
    assert embedding_drift(reference, reference, embedder) < 0.05     # same queries -> ~0
    other = ["explain quantum physics", "what is the meaning of life"]
    assert embedding_drift(reference, other, embedder) > 0.1          # different topics -> drift


def test_drift_report_flags_retrieval_drop_and_stratifies():
    reference = [{"lang": "en", "retrieved": [{"id": "a", "score": 0.80 + (i % 10) * 0.01}],
                  "confidence": 0.8} for i in range(20)]
    current = [{"lang": "en", "retrieved": [{"id": "a", "score": 0.10 + (i % 10) * 0.01}],
                "confidence": 0.8} for i in range(20)]
    report = drift_report(reference, current)
    assert report["drifted"] is True
    assert report["monitors"]["retrieval_score"]["drift"] is True
    assert "en" in report["by_language"]


def test_drift_report_flags_a_grounding_slide_ignoring_abstains():
    # answered turns slide from well-grounded to poorly-grounded -> grounding PSI must flag it
    reference = [{"lang": "en", "tier": "auto", "grounding": 0.95} for _ in range(20)]
    current = [{"lang": "en", "tier": "auto", "grounding": 0.45} for _ in range(20)]
    # a pile of abstains (grounding 0.0) must NOT be read as a groundedness collapse
    current += [{"lang": "en", "tier": "abstain", "grounding": 0.0} for _ in range(20)]
    report = drift_report(reference, current)
    assert report["monitors"]["grounding"]["drift"] is True
    assert report["drifted"] is True


def test_grounding_monitor_ignores_abstain_only_traffic():
    # abstains carry grounding 0.0 but are excluded, so an abstain-heavy window is not false drift
    reference = [{"lang": "en", "tier": "auto", "grounding": 0.9} for _ in range(20)]
    current = [{"lang": "en", "tier": "abstain", "grounding": 0.0} for _ in range(20)]
    report = drift_report(reference, current)
    assert report["monitors"]["grounding"]["psi"] is None  # nothing answered to compare against


def test_drift_report_flags_feedback_rate_rise():
    report = drift_report([], [], feedback_ref=[{"verdict": "up"}] * 10,
                          feedback_cur=[{"verdict": "down"}] * 10)
    assert report["monitors"]["feedback_rate"]["drift"] is True


def test_drift_report_query_embedding_monitor_when_embedder_given():
    embedder = make_embedder("fake")
    reference = [{"query": "how long does shipping take", "lang": "en"}]
    current = [{"query": "explain quantum physics", "lang": "en"}]
    report = drift_report(reference, current, embedder=embedder)
    assert report["monitors"]["query_embedding"]["drift"] is True
