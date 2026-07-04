#!/usr/bin/env python3
"""Fail if domain-specific vocabulary leaked into the engine.

The whole reproducibility thesis is that engine folders are domain agnostic. This linter
loads the specific vocabulary from every domain pack (brand, product names, metric names,
glossary terms) and greps the engine folders for it. If an engine file names a domain thing,
that is a leak: it should live in domains/<name>/ instead.

Scans every engine folder that exists (107 files today) against the pack vocabulary. Wired
into `make check`, so every commit is guarded against a domain term leaking into the engine.
"""
import csv
import os
import re
import sys

# Folders that must stay domain agnostic. Only the ones that exist are scanned.
ENGINE_DIRS = ["adapters", "ingest", "retrieval", "pipeline", "rag", "api", "data", "mlops",
               "knowledge", "evaluation", "scripts", "web/app", "web/components", "web/lib"]
# The reusable server engine: the code that must serve ANY pack unchanged. High-signal but short
# tokens (the short brand form, the persona names) are only checked here, not against the Next.js
# storefront in web/app, which is demo presentation you would fork per client. This keeps the strict
# check honest (it catches brand or persona baked into the pipeline) without flagging demo chrome.
BACKEND_DIRS = ["adapters", "ingest", "retrieval", "pipeline", "rag", "api", "data", "mlops",
                "knowledge", "evaluation", "scripts"]
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


def strict_vocab_for_pack(pack):
    """High-signal short tokens checked only against the reusable backend (BACKEND_DIRS): the
    persona names the engine speaks as and the short brand form used in copy (the first word of a
    brand). The broad check only knew the full brand string, so a short brand baked into a prompt
    slipped through; and it had no persona source at all. Both are the exact things that break the
    domain-swap thesis if they live in engine code, so they are guarded here."""
    dom = load_yaml(os.path.join(pack, "domain.yaml"))
    terms = set()
    persona = dom.get("persona") or {}
    for key in ("assistant", "specialist", "brand_short"):
        val = persona.get(key)
        if isinstance(val, str) and val.strip():
            terms.add(val.strip())
    brand = dom.get("brand")
    if isinstance(brand, str) and brand.strip():
        terms.add(brand.strip().split()[0])  # the short brand form (first word of the full brand)
    # persona names can be short (3+ chars); still drop generic stopwords
    return {t for t in terms if len(t) >= 3 and t.lower() not in STOPWORDS}


def engine_files(dirs=ENGINE_DIRS):
    for base in dirs:
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

    terms, strict_terms = set(), set()
    for name in sorted(os.listdir("domains")):
        pack = os.path.join("domains", name)
        if os.path.isdir(pack) and os.path.isfile(os.path.join(pack, "domain.yaml")):
            terms |= vocab_for_pack(pack)
            strict_terms |= strict_vocab_for_pack(pack)

    if not terms and not strict_terms:
        print("no domain vocabulary found, nothing to check")
        return 0

    # broad vocab (products, metrics, glossary, full brand) is scanned everywhere; the short
    # persona/brand tokens only against the reusable backend (see BACKEND_DIRS).
    patterns = [(t, re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE)) for t in terms]
    strict_patterns = [(t, re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE))
                       for t in strict_terms]
    strict_files = set(engine_files(BACKEND_DIRS))
    leaks = []
    for path in engine_files():
        active = patterns + (strict_patterns if path in strict_files else [])
        with open(path, encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f, 1):
                for term, rx in active:
                    if rx.search(line):
                        leaks.append((path, i, term, line.strip()))

    if leaks:
        print("Domain vocabulary leaked into engine code. Move it into domains/<name>/:\n")
        for path, i, term, line in leaks:
            print("  {}:{}  '{}'  ->  {}".format(path, i, term, line[:100]))
        print("\n{} leak(s).".format(len(leaks)))
        return 1

    scanned = sum(1 for _ in engine_files())
    print("no leaks: {} engine file(s) scanned against {} broad + {} strict domain term(s)".format(
        scanned, len(terms), len(strict_terms)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
