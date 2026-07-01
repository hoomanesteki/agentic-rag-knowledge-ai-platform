"""Smoke tests for the M0 scaffold. They guard the project's basic contract so that
`make test` has something real to check from day one."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_env_example_declares_core_keys():
    env = (ROOT / ".env.example").read_text()
    for key in ["DOMAIN", "LLM_PROVIDER", "GROQ_API_KEY", "EMBED_PROVIDER", "QDRANT_URL"]:
        assert key in env, "missing {} in .env.example".format(key)


def test_build_plan_exists():
    assert (ROOT / "docs" / "BUILD-PLAN.md").is_file()


def test_domain_pack_skill_present():
    assert (ROOT / ".claude" / "skills" / "domain-pack" / "SKILL.md").is_file()
    assert (ROOT / ".claude" / "skills" / "domain-pack" / "scripts"
            / "validate_domain_pack.py").is_file()
