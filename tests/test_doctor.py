"""The doctor's checks: the .env sanity guard (the exact corruption that broke `make up` twice)
and the anti-hang Docker check (the whole point of the doctor, so its failure paths are pinned)."""
import subprocess
from unittest.mock import patch

from scripts.doctor import check_docker, check_env_file


def _run(returncode=0, side_effect=None):
    if side_effect is not None:
        return patch("scripts.doctor.subprocess.run", side_effect=side_effect)
    return patch("scripts.doctor.subprocess.run",
                 return_value=subprocess.CompletedProcess(args=[], returncode=returncode))


def test_docker_up_returns_true():
    with _run(returncode=0):
        assert check_docker() is True


def test_docker_daemon_down_returns_false():
    with _run(returncode=1):
        assert check_docker() is False


def test_docker_timeout_does_not_hang_and_returns_false():
    # the anti-hang guarantee: a blocking daemon must be bounded and return False, not hang
    with _run(side_effect=subprocess.TimeoutExpired("docker", 8)):
        assert check_docker(timeout=1) is False


def test_docker_cli_missing_returns_false():
    with _run(side_effect=FileNotFoundError):
        assert check_docker() is False


def _write(tmp_path, text):
    p = tmp_path / ".env"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_clean_env_passes(tmp_path):
    env = _write(tmp_path, "# a comment\nGROQ_API_KEY=gsk_abc\nDOMAIN=apparel_ecommerce\n")
    assert check_env_file(env) is True


def test_missing_env_is_a_warning_not_a_failure(tmp_path):
    assert check_env_file(str(tmp_path / "nope.env")) is True


def test_stray_word_at_the_top_fails(tmp_path):
    # exactly the corruption we hit: "do M3# Copy this file ..." on line 1
    env = _write(tmp_path, "do M3# Copy this file to .env\nGROQ_API_KEY=gsk_abc\n")
    assert check_env_file(env) is False


def test_a_bare_word_line_fails(tmp_path):
    env = _write(tmp_path, "GROQ_API_KEY=gsk_abc\nnonsense line without equals\n")
    assert check_env_file(env) is False


def test_comments_and_blank_lines_are_fine(tmp_path):
    env = _write(tmp_path, "\n# header\n\nVOYAGE_API_KEY=pa-x\n\n# trailing comment\n")
    assert check_env_file(env) is True
