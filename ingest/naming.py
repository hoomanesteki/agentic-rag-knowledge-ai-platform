"""Collection naming shared by ingest and query, so both point at the same Qdrant collection."""
from __future__ import annotations

import re


def collection_name(domain: str, model: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", "{}__{}".format(domain, model).lower()).strip("_")
    return "{}__v1".format(safe)
