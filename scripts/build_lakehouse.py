#!/usr/bin/env python3
"""Build the active domain's medallion lakehouse (DuckDB) and run its data contracts.

Run: make lakehouse
Needs no keys or network; DuckDB is embedded and reads the domain pack's CSVs.
"""
from __future__ import annotations

import os
import sys

from adapters.config import get_settings
from data.contracts import check_contracts
from data.lakehouse import build_lakehouse

_DB = os.getenv("LAKEHOUSE_DB", "lakehouse.duckdb")


def main() -> int:
    domain = get_settings().domain
    if not os.path.isdir(os.path.join("domains", domain)):
        print("no domain pack at domains/{}".format(domain))
        return 1
    built = build_lakehouse(domain, _DB)
    print("built gold tables for {}: {}".format(domain, ", ".join(built) or "(none)"))
    violations = check_contracts(domain, _DB)
    if violations:
        print("\ndata contract violations:")
        for v in violations:
            print("  - " + v)
        return 1
    print("data contracts passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
