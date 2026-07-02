"""Render an ablation (several eval variants) as a markdown table for docs/eval-report.md."""
from __future__ import annotations


def _f(value, n: int) -> str:
    return "n/a" if n == 0 else "{:.3f}".format(value)


def build_ablation_report(results: list[tuple[str, dict]], *, domain: str, note: str = "",
                          meta: dict | None = None) -> str:
    """results: list of (variant_label, scorecard) from evaluation.harness.evaluate."""
    lines = ["# Retrieval ablation ({})".format(domain), ""]
    if note:
        lines += [note, ""]
    if not results:
        return "\n".join(lines + ["(no variants run)", ""])

    if meta:
        lines.append("provenance: embed={}, rerank={}, top_k_in={}, generated={}".format(
            meta.get("embed"), meta.get("rerank"), meta.get("top_k_in"), meta.get("generated")))
    first = results[0][1]
    cov = first["coverage"]
    lines.append("top_k={}, measured {} qualitative question(s), {} deferred to M4/M5, "
                 "abstain-set {}.".format(first["top_k"], cov["measured"], cov["deferred"],
                                          cov["abstain_set"]))
    lines += [
        "",
        "| variant | scope | hit@k | entity_recall@k | mrr | false_abstain | abstain_recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, sc in results:
        scopes = [("overall", sc["overall"])] + list(sc["by_language"].items())
        for scope, block in scopes:
            r, g = block["retrieval"], block["gate"]
            lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
                label, scope, _f(r["hit_rate_at_k"], r["n"]), _f(r["entity_recall_at_k"], r["n"]),
                _f(r["mrr"], r["n"]), _f(r["false_abstain_rate"], r["n"]),
                _f(g["abstain_recall"], g["n"])))
    return "\n".join(lines) + "\n"
