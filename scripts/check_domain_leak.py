#!/usr/bin/env python3
"""Fail if domain-specific vocabulary leaked into the engine.

The whole reproducibility thesis is that engine folders are domain agnostic. This linter
loads the specific vocabulary from every domain pack (brand, product names, metric names,
glossary terms) and greps the engine folders for it. If an engine file names a domain thing,
that is a leak: it should live in domains/<name>/ instead.

Runs clean today because the engine folders do not exist yet. It starts catching leaks as
soon as they do. Add to `make check` so every commit is guarded.
"""
import csv
import os
import re
import sys

# Folders that must stay domain agnostic. Only the ones that exist are scanned.
ENGINE_DIRS = ["adapters", "ingest", "retrieval", "pipeline", "rag", "api", "data", "mlops",
               "knowledge", "scripts", "web/app", "web/components", "web/lib"]
CODE_EXT = (".py", ".ts", ".tsx", ".js", ".jsx", ".sql", ".yaml", ".yml")

# Generic words that are legitimate in engine code even though a domain might use them.
STOPWORDS = {"product", "products", "review", "reviews", "store", "stores", "supplier",
             "suppliers", "size", "sales", "issue", "issues", "name", "price", "date",
             "category", "rating", "text", "lang", "quantity"}


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def vocab_for_pack(pack):
    """Collect high-signal, domain-specific strings from one pack."""
    terms = set()
    dom = load_yaml(os.path.join(pack, "domain.yaml"))

    # The brand is domain content and must not leak. The domain `name` (slug) is the selector
    # the engine uses to load a pack (for example a default DOMAIN), so it is allowed.
    brand = dom.get("brand")
    if isinstance(brand, str):
        terms.add(brand)

    # glossary canonical keys (domain jargon)
    terms.update((dom.get("glossary") or {}).keys())

    # metric names
    metrics = load_yaml(os.path.join(pack, "metrics.yaml"))
    for m in metrics.get("metrics", []) or []:
        if m.get("name"):
            terms.add(m["name"])

    # product names from the structured seed
    for src in (dom.get("sources", {}) or {}).get("structured", []) or []:
        if src.get("role") != "products":
            continue
        path = os.path.join(pack, src.get("file", ""))
        if os.path.isfile(path):
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    if row.get("name"):
                        terms.add(row["name"])

    # keep terms that are specific enough to be a real signal
    cleaned = set()
    for t in terms:
        t = (t or "").strip()
        if len(t) >= 4 and t.lower() not in STOPWORDS:
            cleaned.add(t)
    return cleaned


def engine_files():
    for base in ENGINE_DIRS:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for name in files:
                if name.endswith(CODE_EXT):
                    yield os.path.join(root, name)


def main():
    if not os.path.isdir("domains"):
        print("no domains/ yet, nothing to check")
        return 0

    terms = set()
    for name in sorted(os.listdir("domains")):
        pack = os.path.join("domains", name)
        if os.path.isdir(pack) and os.path.isfile(os.path.join(pack, "domain.yaml")):
            terms |= vocab_for_pack(pack)

    if not terms:
        print("no domain vocabulary found, nothing to check")
        return 0

    patterns = [(t, re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE)) for t in terms]
    leaks = []
    for path in engine_files():
        with open(path, encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, 1):
                for term, rx in patterns:
                    if rx.search(line):
                        leaks.append((path, i, term, line.strip()))

    if leaks:
        print("Domain vocabulary leaked into engine code. Move it into domains/<name>/:\n")
        for path, i, term, line in leaks:
            print("  {}:{}  '{}'  ->  {}".format(path, i, term, line[:100]))
        print("\n{} leak(s).".format(len(leaks)))
        return 1

    scanned = sum(1 for _ in engine_files())
    print("no leaks: {} engine file(s) scanned against {} domain term(s)".format(
        scanned, len(terms)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
