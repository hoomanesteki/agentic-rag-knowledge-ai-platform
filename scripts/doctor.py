"""Doctor: a fast preflight that catches the environment problems that otherwise hang or fail
cryptically. It bounds the Docker check with a timeout, so `make up` never hangs forever on a
stopped daemon, and it flags a malformed .env (a stray word at the top breaks docker compose and
dotenv). Run `make doctor` any time; it also runs before `make up`.

Usage:
  python scripts/doctor.py                 # full report, exit 1 if anything failed
  python scripts/doctor.py --require docker # gate a single check (used by make up), quiet-ish
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

_GREEN, _RED, _YEL, _RST = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _line(status: str, msg: str) -> None:
    color = {"ok": _GREEN, "FAIL": _RED, "warn": _YEL}[status]
    print("  {}{:4}{} {}".format(color, status, _RST, msg))


def check_docker(timeout: int = 8) -> bool:
    """The daemon must answer within a few seconds, or `docker compose up --wait` hangs. `docker
    info` returns fast (non-zero) when Docker Desktop is stopped; the timeout catches the rarer
    case where it is mid-start and the call would otherwise block."""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=timeout)
    except FileNotFoundError:
        _line("FAIL", "Docker CLI not found. Install Docker Desktop (docker.com).")
        return False
    except subprocess.TimeoutExpired:
        _line("FAIL", "Docker did not answer in {}s. Docker Desktop is starting or stuck; wait for "
                      "the whale icon to settle, then retry.".format(timeout))
        return False
    if result.returncode != 0:
        _line("FAIL", "Docker is installed but the daemon is not running. Open Docker Desktop and "
                      "wait for it to be ready.")
        return False
    _line("ok", "Docker daemon is up")
    return True


def check_env_file(path: str = ".env") -> bool:
    if not os.path.exists(path):
        _line("warn", ".env not found (copy .env.example to .env for real runs)")
        return True
    bad = []
    with open(path, encoding="utf-8") as f:
        for i, raw in enumerate(f, 1):
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            if "=" not in s or not _KEY_RE.match(s.split("=", 1)[0].strip()):
                bad.append((i, s[:50]))
    if bad:
        for i, s in bad:
            _line("FAIL", ".env line {} is not KEY=VALUE or a # comment: {!r}".format(i, s))
        _line("FAIL", "Fix those lines (a stray word at the top of .env is the usual cause); "
                      "docker compose and dotenv both choke on them.")
        return False
    _line("ok", ".env parses cleanly")
    return True


def _load_env() -> dict:
    env = {}
    if os.path.exists(".env"):
        with open(".env", encoding="utf-8") as f:
            for raw in f:
                s = raw.strip()
                if s and not s.startswith("#") and "=" in s:
                    key, _, value = s.partition("=")
                    if _KEY_RE.match(key.strip()):
                        env[key.strip()] = value.strip().strip('"').strip("'")
    env.update(os.environ)  # a real shell/CI env wins over the file
    return env


def check_keys() -> bool:
    env = _load_env()
    needs = []
    if env.get("LLM_PROVIDER", "fake") == "groq":
        needs.append(("GROQ_API_KEY", "gsk_"))
    if "voyage" in (env.get("EMBED_PROVIDER", "fake"), env.get("RERANK_PROVIDER", "none")):
        needs.append(("VOYAGE_API_KEY", "pa-"))
    if not needs:
        _line("ok", "providers are offline fakes; no keys needed")
        return True
    ok = True
    for key, prefix in needs:
        value = env.get(key, "")
        if not value:
            _line("FAIL", "{} is empty but its provider is enabled in .env".format(key))
            ok = False
        elif not value.startswith(prefix):
            _line("warn", "{} does not start with '{}' (double-check it)".format(key, prefix))
        else:
            _line("ok", "{} is set ({} chars)".format(key, len(value)))
    return ok


_CHECKS = {"docker": check_docker, "env": check_env_file, "keys": check_keys}


def main() -> int:
    if "--require" in sys.argv:
        which = sys.argv[sys.argv.index("--require") + 1]
        return 0 if _CHECKS[which]() else 1
    print("Skein doctor:")
    results = [check_env_file(), check_keys(), check_docker()]
    print()
    print("all good." if all(results) else "some checks failed (see above).")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
