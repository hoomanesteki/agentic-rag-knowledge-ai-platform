"""Shared test fixtures.

Settings are read once from the environment and lru_cached. A developer's local .env (with, say,
DEMO_READONLY=true or SKEIN_ENV=production) must not silently break or mask the suite, so pin the
security-sensitive toggles to safe defaults and clear the cache around every test. Tests that need
a specific value set it explicitly (see the _env helper in test_api.py).
"""
import os

import pytest

from adapters import config

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
