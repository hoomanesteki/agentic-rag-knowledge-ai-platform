"""Load a domain pack's golden set."""
from __future__ import annotations

import json
import os


def load_golden(pack_dir: str) -> list[dict]:
    path = os.path.join(pack_dir, "eval", "golden.jsonl")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
