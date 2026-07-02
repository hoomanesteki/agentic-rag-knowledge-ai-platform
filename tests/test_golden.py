"""Every domain pack must ship a golden set that actually tests the system: answerable,
unanswerable, and out-of-domain questions. A pack of softballs proves nothing, so this is
enforced in CI, not left to good intentions."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ROOT / "domains"
PACKS = sorted(DOMAINS.glob("*/eval/golden.jsonl")) if DOMAINS.is_dir() else []


def _load(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def test_golden_sets_cover_all_types():
    # Vacuously passes before any pack exists; bites once a pack is added.
    for path in PACKS:
        rows = _load(path)
        types = {r.get("type") for r in rows}
        for needed in ("answerable", "unanswerable", "out_of_domain"):
            assert needed in types, "{} is missing '{}' questions".format(path, needed)
        assert len(rows) >= 10, "{} has only {} questions, add more".format(path, len(rows))
