"""The semantic answer cache serves a repeat FAQ without retrieval or generation, but only for
anonymous answers-lane turns and only for grounded answers. These pin the cache mechanics (hit,
miss, TTL, invalidate) and the omni scoping (a repeat hits without re-generating; a signed-in turn
and a non-answers lane bypass it)."""
import rag.omni as omni
from adapters.factory import make_embedder
from rag.answer_cache import SemanticAnswerCache
from rag.router import RouteDecision

_Q = "what is the general company policy regarding product returns or exchanges"


def _cache(**kw):
    return SemanticAnswerCache(make_embedder("fake"), **kw)


def test_put_then_identical_query_hits():
    c = _cache()
    assert c.get(_Q) is None  # cold miss
    c.put(_Q, "you have 30 days", [{"id": "CK02"}], grounding=0.9, confidence=0.8)
    hit = c.get(_Q)
    assert hit is not None and hit["answer"] == "you have 30 days"
    assert hit["similarity"] >= 0.95 and hit["citations"] == [{"id": "CK02"}]


def test_unrelated_query_misses():
    c = _cache()
    c.put(_Q, "you have 30 days", [])
    assert c.get("do you sell waterproof jackets for winter hiking trips") is None


def test_ttl_expires_an_entry():
    now = [1000.0]
    c = _cache(ttl_s=10.0, clock=lambda: now[0])
    c.put(_Q, "you have 30 days", [])
    now[0] = 1005.0
    assert c.get(_Q) is not None  # still fresh
    now[0] = 1020.0
    assert c.get(_Q) is None  # past the TTL


def test_invalidate_clears_everything():
    c = _cache()
    c.put(_Q, "you have 30 days", [])
    c.invalidate()
    assert c.get(_Q) is None
    assert c.stats()["entries"] == 0


def _answers_decision(*a, **k):
    return RouteDecision("answers", 0.9, 1, "test", [], None)


class _Rec:
    def __init__(self):
        self.calls = 0

    def __call__(self, query, **kwargs):
        self.calls += 1
        yield {"type": "token", "text": "policy answer"}
        yield {"type": "final", "answer": "policy answer", "lane": kwargs.get("lane"),
               "tier": "auto", "grounding": 0.9, "confidence": 0.8, "citations": [{"id": "c1"}]}


def test_repeat_anonymous_answers_turn_is_served_from_cache(monkeypatch):
    rec = _Rec()
    monkeypatch.setattr(omni, "route", _answers_decision)
    monkeypatch.setattr(omni, "stream_answer", rec)
    cache = _cache()
    kw = dict(embedder=make_embedder("fake"), store=None, llm=None, answer_cache=cache)

    first = list(omni.stream_omni(_Q, **kw))
    assert rec.calls == 1 and not first[-1].get("cache_hit")  # miss: generated and stored

    second = list(omni.stream_omni(_Q, **kw))
    assert rec.calls == 1  # hit: generation was NOT called again
    assert second[-1]["cache_hit"] is True and second[-1]["answer"] == "policy answer"


def test_signed_in_turn_bypasses_the_cache(monkeypatch):
    rec = _Rec()
    monkeypatch.setattr(omni, "route", _answers_decision)
    monkeypatch.setattr(omni, "stream_answer", rec)
    cache = _cache()
    kw = dict(embedder=make_embedder("fake"), store=None, llm=None, answer_cache=cache)

    list(omni.stream_omni(_Q, **kw))  # anonymous: populates the cache
    list(omni.stream_omni(_Q, auth_identity=("Aaron", "a@b.com"), **kw))  # signed in: must bypass
    assert rec.calls == 2  # the signed-in turn generated instead of serving the cached answer
