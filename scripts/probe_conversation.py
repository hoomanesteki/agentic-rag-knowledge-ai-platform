#!/usr/bin/env python3
"""Drive real multi-turn conversations through the chat brain and record what happened per turn.

This is the behavioral probe harness: it runs a scripted conversation (a list of shopper turns)
against the LIVE pipeline (Qdrant + Groq + Cohere, exactly what api/deps builds), maintaining
history the way the web client does, and captures for every turn the routed lane, the persona that
answered, the reply text, the products it cited, and two automatic behavioral flags:

  * gender_leak  -- a cited product whose catalog gender is the opposite of the recipient the
                    scenario is shopping for (the "asked about father, got women's clothes" bug).
  * repeat_ask   -- the assistant asked essentially the same clarifying question it already asked a
                    prior turn (the "keeps asking what she's into" bug).

It is used two ways: (1) by the deep-loop review (Opus/Fable read the transcripts and judge them),
and (2) by the regression tests, which assert the flags stay clean on the fixed build.

Run:  PYTHONPATH=. uv run python scripts/probe_conversation.py --scenarios path.json --out out.json
Needs: keys in .env, Qdrant up (make up), an ingest done. Brain defaults to omni (the orchestrator).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

# a stated recipient gender is a hard facet; these mirror pipeline.answer but are read from the
# catalog so the check is ground truth, not a re-implementation of the engine's own logic.
_HISTORY_WINDOW = 6  # the web client sends the last 6 non-empty turns


def _product_gender_map(domain: str) -> dict[str, str]:
    """product_id and lowercased name -> 'men'/'women', read straight from the pack's products.csv,
    so the gender check is grounded in the catalog rather than the engine's inference."""
    out: dict[str, str] = {}
    path = os.path.join("domains", domain, "seed", "structured", "products.csv")
    try:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                g = (row.get("gender") or "").strip().lower()
                if g not in ("men", "women"):
                    continue
                if row.get("product_id"):
                    out[row["product_id"].strip()] = g
                if row.get("name"):
                    out[row["name"].strip().lower()] = g
    except OSError:
        pass
    return out


def _last_question(text: str) -> str:
    """The last interrogative sentence in a reply, normalized to word tokens for overlap compare."""
    qs = re.findall(r"[^.?!]*\?", text or "")
    if not qs:
        return ""
    return " ".join(re.findall(r"[a-z]+", qs[-1].lower()))


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _build():
    """Build the production components plus a small LLM for the router tie-break."""
    from adapters.factory import make_small_llm
    from api.deps import get_components
    comp = get_components()
    return comp, make_small_llm()


def run_turn(comp, small, query, history, *, brain="omni", persona=None, auth_identity=None,
             concise=False, notes=None):
    """Run one turn, return (text, final_event, routed_lane)."""
    from pipeline.answer import stream_answer
    from rag.omni import stream_omni
    from rag.router import route
    routed = route(query, history=history, signed_in=bool(auth_identity and auth_identity[0]),
                   small_llm=small)
    deps = dict(embedder=comp["embedder"], store=comp["store"], llm=comp["llm"],
                reranker=comp["reranker"], metric_resolver=comp["metric_resolver"],
                graph_retriever=comp["graph_retriever"], lang=None, persona=persona,
                history=history, concise=concise, auth_identity=auth_identity, notes=notes)
    if brain == "omni":
        gen = stream_omni(query, small_llm=small, review_queue=comp.get("review_queue"),
                          domain=comp.get("domain"), **deps)
    else:
        gen = stream_answer(query, **deps)
    parts: list[str] = []
    final: dict = {}
    for ev in gen:
        if ev.get("type") == "token":
            parts.append(ev.get("text", ""))
        elif ev.get("type") == "final":
            final = ev
    text = "".join(parts) or final.get("answer", "")
    return text, final, routed.lane


def run_scenario(comp, small, scenario, gmap):
    """Run every turn of one scenario, threading history like the web client, and flag defects."""
    recipient = (scenario.get("recipient_gender") or "").strip().lower() or None
    brain = scenario.get("brain", "omni")
    persona = scenario.get("persona")  # "agent" simulates a session stuck with the care specialist
    auth_identity = scenario.get("auth_identity")
    history: list[dict] = []
    prior_questions: list[str] = []
    turns_out = []
    for i, user_turn in enumerate(scenario.get("turns", [])):
        try:
            text, final, lane = run_turn(comp, small, user_turn, history, brain=brain,
                                         persona=persona, auth_identity=auth_identity)
            err = None
        except Exception as exc:  # one flaky Groq/Cohere call must not sink the whole scenario
            text, final, lane, err = "", {}, "error", "{}: {}".format(type(exc).__name__, exc)

        cites = final.get("citations") or []
        cited = []
        leaks = []
        for c in cites:
            cid = str(c.get("id") or "")
            g = gmap.get(cid) or gmap.get((c.get("title") or "").strip().lower())
            cited.append({"id": cid, "doc_type": c.get("doc_type"), "gender": g})
            if recipient and g and g != recipient:
                leaks.append({"id": cid, "gender": g, "where": "citation"})
        # The real leak shows up in the PROSE (a named opposite-gender product recommended for the
        # recipient), which citation metadata often misses, so scan the answer text for any
        # opposite-gender product name from the catalog. This catches the "gift for him" mismatch.
        if recipient:
            low = text.lower()
            for name, g in gmap.items():
                if " " in name and g != recipient and re.search(
                        r"\b" + re.escape(name) + r"\b", low):
                    leaks.append({"name": name, "gender": g, "where": "prose"})

        q = _last_question(text)
        repeat = bool(q) and any(_jaccard(q, pq) >= 0.6 for pq in prior_questions)
        if q:
            prior_questions.append(q)

        turns_out.append({
            "i": i, "user": user_turn, "lane": lane, "persona": final.get("persona"),
            "tier": final.get("tier"), "answer": text,
            "cited": cited, "gender_leak": leaks, "repeat_ask": repeat, "error": err,
        })
        # thread history exactly like the client: append both sides, keep the last N non-empty turns
        history = (history + [{"role": "user", "content": user_turn},
                              {"role": "assistant", "content": text}])
        history = [t for t in history if (t.get("content") or "").strip()][-_HISTORY_WINDOW:]

    return {
        "name": scenario.get("name", "scenario"),
        "recipient_gender": recipient,
        "brain": brain,
        "turns": turns_out,
        "gender_leaks": sum(len(t["gender_leak"]) for t in turns_out),
        "repeat_asks": sum(1 for t in turns_out if t["repeat_ask"]),
        "errors": sum(1 for t in turns_out if t["error"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True, help="JSON list of scenarios")
    ap.add_argument("--out", default="", help="write full transcripts here (JSON)")
    args = ap.parse_args()

    with open(args.scenarios, encoding="utf-8") as f:
        scenarios = json.load(f)
    if isinstance(scenarios, dict):
        scenarios = scenarios.get("scenarios", [])

    from adapters.config import get_settings
    domain = get_settings().domain
    gmap = _product_gender_map(domain)
    comp, small = _build()

    results = [run_scenario(comp, small, s, gmap) for s in scenarios]

    summary = {
        "scenarios": len(results),
        "gender_leaks": sum(r["gender_leaks"] for r in results),
        "repeat_asks": sum(r["repeat_asks"] for r in results),
        "errors": sum(r["errors"] for r in results),
    }
    print(json.dumps(summary, indent=2))
    for r in results:
        print("  {:32s} leaks={} repeats={} errors={}  lanes={}".format(
            r["name"][:32], r["gender_leaks"], r["repeat_asks"], r["errors"],
            [t["lane"] for t in r["turns"]]))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2)
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
