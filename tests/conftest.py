"""Shared test fixtures.

Settings are read once from the environment and lru_cached. A developer's local .env (with, say,
DEMO_READONLY=true or SKEIN_ENV=production) must not silently break or mask the suite, so pin the
security-sensitive toggles to safe defaults and clear the cache around every test. Tests that need
a specific value set it explicitly (see the _env helper in test_api.py).
"""
import os
import tempfile

# Redirect the trace/feedback/verified paths to a throwaway dir BEFORE anything imports them.
# pipeline.answer binds DEFAULT_TRACE_PATH from TRACE_PATH at import, and api.app reads
# FEEDBACK_PATH at import, so setting these first keeps `make check` from writing fake test
# traffic into the real traces/ that drift, MLflow, and the admin dashboards read. Set before
# load_dotenv runs (via the adapters.config import below); load_dotenv does not override.
_TRACE_DIR = tempfile.mkdtemp(prefix="skein-test-traces-")
os.environ["TRACE_PATH"] = os.path.join(_TRACE_DIR, "requests.jsonl")
os.environ["FEEDBACK_PATH"] = os.path.join(_TRACE_DIR, "feedback.jsonl")
os.environ["VERIFIED_PATH"] = os.path.join(_TRACE_DIR, "verified_answers.jsonl")

import pytest  # noqa: E402

from adapters import config  # noqa: E402

# Run at collection time, before any test module imports api.app (whose module-level
# `app = create_app()` would otherwise raise if the developer's .env sets SKEIN_ENV=production
# with an insecure secret). JWT_SECRET is left untouched so the tokens minted at import in
# test_api.py still verify.
os.environ.pop("SKEIN_ENV", None)
os.environ.pop("DEMO_READONLY", None)
config.get_settings.cache_clear()


@pytest.fixture(autouse=True)
def hermetic_settings(monkeypatch):
    monkeypatch.delenv("DEMO_READONLY", raising=False)
    monkeypatch.delenv("SKEIN_ENV", raising=False)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()
