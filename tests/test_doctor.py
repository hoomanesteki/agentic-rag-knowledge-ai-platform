"""The doctor's .env sanity check. This is the guard for the exact corruption that broke
`make up` twice: a stray word at the top of .env (which docker compose and dotenv both reject)."""
from scripts.doctor import check_env_file


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
