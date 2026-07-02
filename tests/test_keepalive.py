"""M9.3 keepalive: the target plan is built purely from the environment, so a paused free tier is
checked only when it is actually configured (and local stores are never pinged)."""
from scripts.keepalive import plan


def _names(env):
    return [name for name, _check in plan(env)]


def test_empty_env_has_no_targets():
    assert plan({}) == []


def test_api_url_adds_health():
    assert _names({"KEEPALIVE_API_URL": "https://api.example.com"}) == ["api-health"]


def test_localhost_stores_are_skipped():
    env = {"KEEPALIVE_API_URL": "http://127.0.0.1:8000",
           "QDRANT_URL": "http://localhost:6333", "GRAPH_PROVIDER": "neo4j",
           "NEO4J_URL": "http://0.0.0.0:7474"}
    assert _names(env) == []


def test_hosted_qdrant_and_neo4j_are_checked():
    env = {"QDRANT_URL": "https://xyz.cloud.qdrant.io", "GRAPH_PROVIDER": "neo4j",
           "NEO4J_URL": "https://xyz.databases.neo4j.io"}
    assert _names(env) == ["qdrant", "neo4j"]


def test_neo4j_only_when_provider_is_neo4j():
    env = {"GRAPH_PROVIDER": "memory", "NEO4J_URL": "https://xyz.databases.neo4j.io"}
    assert "neo4j" not in _names(env)
