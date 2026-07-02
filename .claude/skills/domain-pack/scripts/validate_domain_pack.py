#!/usr/bin/env python3
"""Validate a Skein Lite domain pack against the reproducibility contract.

Usage:
    python validate_domain_pack.py domains/<name>

Exit code 0 means the pack meets the contract. Non zero means fix what is printed.
Works without PyYAML for structure checks. Install PyYAML for the full cross reference
check: pip install pyyaml
"""
import json
import os
import sys

REQUIRED_FILES = ["domain.yaml", "ontology.cypher", "metrics.yaml", "eval/golden.jsonl"]
REQUIRED_DIRS = ["seed/structured", "seed/unstructured"]
REQUIRED_KEYS = ["name", "languages", "entity_types", "sources"]
GOLDEN_TYPES = {"answerable", "unanswerable", "out_of_domain"}

errors = []
warnings = []


def err(msg):
    errors.append(msg)


def warn(msg):
    warnings.append(msg)


def load_yaml(path):
    try:
        import yaml
    except ImportError:
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def check_structure(pack):
    for rel in REQUIRED_FILES:
        if not os.path.isfile(os.path.join(pack, rel)):
            err("missing required file: {}".format(rel))
    for rel in REQUIRED_DIRS:
        if not os.path.isdir(os.path.join(pack, rel)):
            err("missing required dir: {}".format(rel))


def check_domain_yaml(pack, dom):
    for key in REQUIRED_KEYS:
        if key not in dom:
            err("domain.yaml missing key: {}".format(key))

    name = dom.get("name")
    folder = os.path.basename(os.path.normpath(pack))
    if name and name != folder and name != "CHANGE_ME":
        warn("domain.yaml name '{}' does not match folder '{}'".format(name, folder))
    if name == "CHANGE_ME":
        err("domain.yaml name is still the template placeholder CHANGE_ME")

    sources = dom.get("sources", {}) or {}
    declared_roles = set()

    for group in ("structured", "unstructured"):
        for src in sources.get(group, []) or []:
            f = src.get("file")
            if not f:
                err("a {} source has no 'file'".format(group))
                continue
            if not os.path.isfile(os.path.join(pack, f)):
                err("source file listed in manifest does not exist: {}".format(f))
            role = src.get("role") if group == "structured" else src.get("doc_type")
            if role:
                declared_roles.add(role)

    # entity types must appear in the ontology
    onto_path = os.path.join(pack, "ontology.cypher")
    onto_text = ""
    if os.path.isfile(onto_path):
        with open(onto_path) as f:
            onto_text = f.read()
    for et in dom.get("entity_types", []) or []:
        if et not in onto_text:
            err("entity_type '{}' not found as a label in ontology.cypher".format(et))

    return declared_roles


def check_metrics(pack, declared_roles):
    metrics = load_yaml(os.path.join(pack, "metrics.yaml"))
    if metrics is None:
        return
    for m in (metrics or {}).get("metrics", []) or []:
        nm = m.get("name", "<unnamed>")
        src = m.get("source")
        if not src:
            err("metric '{}' has no source".format(nm))
        elif declared_roles and src not in declared_roles:
            # gold models built from a role are allowed, so this is a warning
            warn("metric '{}' source '{}' is not a declared manifest role "
                 "(ok only if it is a gold model built from one)".format(nm, src))


def check_unstructured_fields(pack, dom):
    sources = (dom.get("sources", {}) or {}).get("unstructured", []) or []
    for src in sources:
        f = src.get("file")
        path = os.path.join(pack, f) if f else None
        if not path or not os.path.isfile(path):
            continue
        needed = [src.get("id_field"), src.get("text_field"), src.get("lang_field")]
        needed = [n for n in needed if n]
        with open(path) as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    err("{}: line {} is not valid JSON".format(f, i + 1))
                    break
                for field in needed:
                    if field not in rec:
                        err("{}: record {} missing promised field '{}'".format(f, i + 1, field))
                if i >= 4:  # sample the first few lines
                    break


def check_golden(pack):
    path = os.path.join(pack, "eval", "golden.jsonl")
    if not os.path.isfile(path):
        return  # missing file already reported by the structure check
    count = 0
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                err("eval/golden.jsonl: line {} is not valid JSON".format(i))
                continue
            for field in ("id", "lang", "question", "type"):
                if field not in rec:
                    err("eval/golden.jsonl: record {} missing '{}'".format(i, field))
            t = rec.get("type")
            if t is not None and t not in GOLDEN_TYPES:
                err("eval/golden.jsonl: record {} has bad type '{}' (use {})".format(
                    i, t, ", ".join(sorted(GOLDEN_TYPES))))
    if count == 0:
        err("eval/golden.jsonl is empty, add some questions")


def main():
    if len(sys.argv) != 2:
        print("usage: python validate_domain_pack.py domains/<name>")
        return 2
    pack = sys.argv[1]
    if not os.path.isdir(pack):
        print("not a directory: {}".format(pack))
        return 2

    check_structure(pack)
    check_golden(pack)  # independent of PyYAML

    dom = load_yaml(os.path.join(pack, "domain.yaml"))
    if dom is None:
        warn("PyYAML not installed, skipping content checks. Run: pip install pyyaml")
    elif errors:
        pass  # structure already broken, still try content
    if dom is not None:
        roles = check_domain_yaml(pack, dom)
        check_metrics(pack, roles)
        check_unstructured_fields(pack, dom)

    for w in warnings:
        print("WARN: {}".format(w))
    for e in errors:
        print("FAIL: {}".format(e))

    if errors:
        print("\n{} problem(s) to fix.".format(len(errors)))
        return 1
    print("OK: domain pack meets the contract"
          + (" (structure only, install pyyaml for full check)" if dom is None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
