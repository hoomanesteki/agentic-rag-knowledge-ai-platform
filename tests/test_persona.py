"""The engine must carry no brand or persona of its own: the assistant name, specialist name,
brand, and industry all come from the active pack's domain.yaml. This is the domain-swap thesis, so
the same answer pipeline introduces itself with each pack's identity and never bleeds one pack's
brand into another. If these fail, the leak linter would also flag hardcoded vocab in an engine dir.
"""
from pipeline.answer import _agent_system, _persona, _smalltalk, _system

APPAREL = "apparel_ecommerce"
SAAS = "saas_support"


def test_persona_is_read_from_each_pack():
    a = _persona(APPAREL)
    assert a["assistant"] == "Aria" and a["specialist"] == "Sara"
    assert a["brand"] == "Aster" and a["industry"] == "an athletic apparel brand"
    s = _persona(SAAS)
    assert s["assistant"] == "Nova" and s["specialist"] == "Remy"
    assert s["brand"] == "Northwind"


def test_system_prompts_use_the_active_pack_identity():
    apparel = _system(APPAREL) + _agent_system(APPAREL)
    assert "Aria" in apparel and "Sara" in apparel and "Aster" in apparel
    saas = _system(SAAS) + _agent_system(SAAS)
    assert "Nova" in saas and "Remy" in saas and "Northwind" in saas
    # no cross-contamination: the saas prompts must not carry the apparel identity, and vice versa
    for token in ("Aria", "Sara", "Aster"):
        assert token not in saas, token
    for token in ("Nova", "Remy", "Northwind"):
        assert token not in apparel, token


def test_greetings_and_intros_speak_as_the_active_pack():
    assert "Nova" in _smalltalk("hi", None, SAAS)
    assert "Aria" in _smalltalk("hi", None, APPAREL)
    # the human specialist greeting uses the specialist name
    assert "Remy" in _smalltalk("hi", "agent", SAAS)
    assert "Sara" in _smalltalk("hi", "agent", APPAREL)
    # "who are you" introduces the pack's assistant, never the other pack's
    saas_intro = _smalltalk("who are you", None, SAAS)
    assert "Nova" in saas_intro and "Aria" not in saas_intro


def test_greeting_recognizes_either_persona_name():
    # "hey <assistant>" and "hey <specialist>" must both be caught as greetings: the trailing-name
    # group has to space-prefix every alternative, not just the first.
    for q in ("hey aria", "hi sara", "hey there sara", "hello aria"):
        assert _smalltalk(q, None, APPAREL) is not None, q
    for q in ("hey nova", "hi remy"):
        assert _smalltalk(q, None, SAAS) is not None, q


def test_missing_persona_falls_back_to_neutral_voice():
    # a pack slug with no manifest must not crash; it yields neutral placeholders
    p = _persona("does_not_exist")
    assert p["assistant"] and p["brand"] and p["industry"]
    assert "Aria" not in _system("does_not_exist")
