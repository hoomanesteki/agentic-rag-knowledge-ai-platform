"""The engine must carry no brand or persona of its own: the assistant name, specialist name,
brand, and industry all come from the active pack's domain.yaml. This is the domain-swap thesis, so
the answer pipeline introduces itself with the pack's identity, and a pack with no manifest falls
back to a neutral voice instead of a hardcoded one. If these fail, the leak linter would also flag
hardcoded vocab in an engine dir.
"""
from pipeline.answer import _agent_system, _persona, _smalltalk, _system

APPAREL = "apparel_ecommerce"


def test_persona_is_read_from_the_pack():
    a = _persona(APPAREL)
    assert a["assistant"] == "Aria" and a["specialist"] == "Sara"
    assert a["brand"] == "Aster" and a["industry"] == "an athletic apparel brand"


def test_system_prompts_use_the_active_pack_identity():
    apparel = _system(APPAREL) + _agent_system(APPAREL)
    assert "Aria" in apparel and "Sara" in apparel and "Aster" in apparel
    # the identity comes from the pack, not from engine code: a slug with no manifest falls back to
    # a neutral voice and never carries the apparel persona, so nothing is hardcoded in the prompt.
    neutral = _system("does_not_exist") + _agent_system("does_not_exist")
    for token in ("Aria", "Sara", "Aster"):
        assert token not in neutral, token


def test_greetings_and_intros_speak_as_the_active_pack():
    assert "Aria" in _smalltalk("hi", None, APPAREL)
    # the human specialist greeting uses the specialist name
    assert "Sara" in _smalltalk("hi", "agent", APPAREL)
    # "who are you" introduces the pack's assistant
    assert "Aria" in _smalltalk("who are you", None, APPAREL)


def test_greeting_recognizes_either_persona_name():
    # "hey <assistant>" and "hey <specialist>" must both be caught as greetings: the trailing-name
    # group has to space-prefix every alternative, not just the first.
    for q in ("hey aria", "hi sara", "hey there sara", "hello aria"):
        assert _smalltalk(q, None, APPAREL) is not None, q


def test_missing_persona_falls_back_to_neutral_voice():
    # a pack slug with no manifest must not crash; it yields neutral placeholders
    p = _persona("does_not_exist")
    assert p["assistant"] and p["brand"] and p["industry"]
    assert "Aria" not in _system("does_not_exist")
