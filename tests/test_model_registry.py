"""The model registry versions and stages a retrained artifact so promotion stays human-gated and
auditable. These lock in: register lands in `proposed` (never auto-promoted), promotion to
production archives the incumbent (one champion at a time), and the CT-cycle audit log persists."""
from mlops.model_registry import ModelRegistry


def test_register_lands_in_proposed_never_auto_promoted(tmp_path):
    reg = ModelRegistry(str(tmp_path / "registry.json"))
    v = reg.register(name="tiebreak_system", kind="prompt", source="ct-cycle",
                     created_at="2026-07-06T00:00:00Z",
                     metrics={"baseline": 0.80, "candidate": 0.86})
    assert v == 1
    assert reg.versions()[0]["stage"] == "proposed"  # a human still has to promote it
    assert reg.champion() is None  # nothing is in production until a human transitions it


def test_promotion_archives_the_incumbent_so_one_champion_at_a_time(tmp_path):
    reg = ModelRegistry(str(tmp_path / "registry.json"))
    v1 = reg.register(name="cfg", kind="config", source="prompt_opt", created_at="t1")
    reg.transition(v1, "production", at="t2")
    assert reg.champion()["version"] == v1
    v2 = reg.register(name="cfg", kind="config", source="ct-cycle", created_at="t3")
    reg.transition(v2, "production", at="t4")
    champ = reg.champion()
    assert champ["version"] == v2  # the new one is champion
    incumbent = next(x for x in reg.versions() if x["version"] == v1)
    assert incumbent["stage"] == "archived"  # exactly one production version
    assert sum(1 for x in reg.versions() if x["stage"] == "production") == 1


def test_transition_rejects_unknown_stage_and_version(tmp_path):
    import pytest
    reg = ModelRegistry(str(tmp_path / "registry.json"))
    v = reg.register(name="p", kind="prompt", source="s", created_at="t")
    with pytest.raises(ValueError):
        reg.transition(v, "shipped", at="t")  # not a real stage
    assert reg.transition(999, "staging", at="t") is False  # unknown version


def test_cycle_audit_log_and_reload_persist(tmp_path):
    path = str(tmp_path / "registry.json")
    reg = ModelRegistry(path)
    reg.record_cycle({"triggered": True, "promoted": False, "at": "t1"})
    reg.record_cycle({"triggered": False, "at": "t2"})
    reg.register(name="p", kind="prompt", source="ct-cycle", created_at="t3")
    # a fresh instance reads the same file: the registry is durable, not in-memory
    reloaded = ModelRegistry(path)
    assert len(reloaded._data["cycles"]) == 2
    assert len(reloaded.versions()) == 1


def test_champion_and_challenger_aliases(tmp_path):
    # MLflow's post-2.9 vocabulary: the production version is @champion, the latest proposal the
    # @challenger a human is deciding on
    reg = ModelRegistry(str(tmp_path / "aliases.json"))
    assert reg.aliases() == {"champion": None, "challenger": None}
    v1 = reg.register(name="tiebreak", kind="prompt", source="ct", created_at="t1")
    assert reg.aliases()["challenger"] == v1 and reg.champion() is None
    reg.transition(v1, "production", at="t2")
    assert reg.aliases() == {"champion": v1, "challenger": None}  # promoted, nothing left to review
    v2 = reg.register(name="tiebreak", kind="prompt", source="ct", created_at="t3")
    assert reg.aliases() == {"champion": v1, "challenger": v2}  # next proposal is the challenger
