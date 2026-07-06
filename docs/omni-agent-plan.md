I have verified every load-bearing fact against both the live Claude API reference and the actual code. Here is the build plan.

---

# Skein Lite Omni-Agent: Engineer Build Plan

Verified before writing: I read the Claude API reference (model IDs, pricing, cache minimums, thinking defaults, sampling, refusal, `role:system`, cache economics) and the actual repo (`pipeline/answer.py`, all of `rag/`, `adapters/config.py`, `adapters/factory.py`, `domains/apparel_ecommerce/domain.yaml`, `api/app.py`, `web/app/ChatWidget.tsx`, `web/app/Avatar.tsx`). All line references below are confirmed against the current tree on branch `feat/omni-agent`.

## Frontier-fact verification result (the contested points, now settled)

- Model IDs correct: `claude-sonnet-5`, `claude-opus-4-8`, `claude-haiku-4-5`.
- Pricing (steady): Sonnet 5 `3.00/15.00`, Opus 4.8 `5.00/25.00`, Haiku 4.5 `1.00/5.00`. Sonnet 5 is in its intro window (`2.00/10.00`) through 2026-08-31, so today's live Sonnet-5 cost is the intro rate. Price `_PRICES` at steady for a conservative gate; optionally add an intro-rate note in the trace.
- Min cacheable prefix: Opus 4.8 = 4096, Haiku 4.5 = 4096, Sonnet 4.6/Fable 5 = 2048. Sonnet 5 is NOT published. Padding Tiffany's stable prefix to >= 4096 is therefore correct: it caches on Opus 4.8 and Haiku 4.5 with certainty and is at or above every published minimum.
- Thinking defaults: on Sonnet 5 omitting `thinking` runs adaptive (so set `thinking:{type:"disabled"}` explicitly on default Tiffany); on Opus 4.8 omitting runs without thinking (so set `thinking:{type:"adaptive"}` explicitly on the HARD path). Confirmed.
- Sampling: `temperature`/`top_p`/`top_k` return 400 on Sonnet 5 and Opus 4.8. `adapters/groq.py` sends `temperature: 0`, so `make_frontier_llm` must not be copied from it.
- Refusal: HTTP 200 with `stop_reason == "refusal"`; `stop_details` can be null; check `stop_reason` before indexing `content[0]`. Pre-output refusal has empty `content` and is unbilled; mid-stream bills the partial.
- `role:"system"` mid-conversation is Opus 4.8 only; on Sonnet 5 it 400s. Put volatile CaseFile facts in a trailing user-turn `<system-reminder>` on the Sonnet 5 path; reserve `role:system` for the Opus 4.8 HARD path.
- Cache economics: reads ~0.1x, writes 1.25x at 5-minute TTL and 2x at 1-hour. A 3-5 turn escalation breaks even at 2 requests on 5-minute TTL, so default 5-minute TTL.

Net: the synthesis's frontier decisions are all correct as written. No changes needed to the model-tiering plan.

---

## 1. Architecture overview

One master orchestrator sits in front of the existing linear streamer and a new heavy LangGraph subgraph. The orchestrator is a cheap deterministic `route()` plus a compiled graph, not a class hierarchy.

```
every turn ->  input_guard (deterministic, $0)
            ->  route()  (Layer 0 $0 -> Layer 1 $0 -> Layer 2 8B tie-break)
            ->  FAST lane  (single, certain)  ->  stream_answer(role_fragment=...)   ~90%
            ->  HEAVY subgraph (multi/ambiguous/complaint/escalate):
                   plan -> Send(clause*) -> lane node* -> stitch -> output_guard
                        -> {clarify | escalate(Tiffany) | emit}
```

Roster. Five service lanes are data rows `(retrieval_bias, role_fragment, required_artifact)` in `rag/roles.py`, never classes. Two guardrail nodes are deterministic and un-promptable. One escalation persona uses a frontier model.

| Agent | Job | Skills (the 2x) | Model tier | Owns (tools/data) |
|---|---|---|---|---|
| Input guard | Refuse/deflect before any model runs | harm decline, injection/exfiltration refusal, customer-enumeration refusal, order-PII gate | none, `$0` | the `_smalltalk` intercepts, `_problem/_account/_shopping/heuristic_route` classifiers, `retrieve()`'s `_order_access_ok` |
| Stylist (Sara) | gift matching, outfit pairing | sums stated budgets vs bullet prices; states fit truth (return_rate_by_size, critical reviews P012/P014/P021) before praise; closes gifts with CK03 exchange guarantee; never re-asks a filled slot | 70B `llama-3.3-70b-versatile` | `graph_facts`, `get_governed_metric`, retriever contexts |
| Care (Sara) | profile, order status, delivered-or-not, self-serve deadlines | three-part verdict (status + date math + next step); dates from `helpers/dates.py`, never model arithmetic; zero re-verify when signed in | 8B simple / 70B complex | `get_profile`, `lookup_order_status` (through the gate), `get_policy` |
| Complaint (Sara) | empathy-first resolution | first sentence names item+failure; settles CK04/CK45/CK02/CK03/CK13/CK52 remedies in chat with amount + computed date + citation; zero banned phrases; escalates only discretionary money | 70B | `lookup_order_status`, `get_policy`, `helpers/dates.py` |
| Answers (Sara) | product facts + policy/knowledge + fallback | one lane carrying three prompt fragments; split only if a golden slice shows dilution | 8B `llama-3.1-8b-instant` | retriever, `get_policy`, `graph_facts` |
| Output guard | tripwire on stitched/re-routed answers | on an unauthorized order-PII token, replace with the gate refusal and log `security.leak_blocked` | none, `$0` | `_ORDER_TERM` (answer.py:687), `auth_text` |
| Tiffany | AI care coordinator in front of the human review queue | gathers, confirms a numbered list, files a CaseFile, states email follow-up, asks "any other concerns", flips back to Sara | frontier `claude-sonnet-5` default; `claude-opus-4-8` HARD_RESOLUTION; `claude-haiku-4-5` confirm-back floor | `ReviewQueue.enqueue`/`.get` (rag/hitl.py), `lookup_order_status` (gate-obeying) |

Safety is a guardrail node, not a promptable agent. This is fixed, not open.

## 2. Routing, multi-task planning, clarify-not-hallucinate contract

Routing is `route(state) -> RouteDecision` in a new `rag/router.py`, unit-testable with no graph. Three cheap-first layers:

- Layer 0 (`$0`): the `_smalltalk` intercepts (answer.py:225) exactly as `stream_answer` runs them; short-circuit to END.
- Layer 1 (`$0`, ~70%): compose the existing regexes. `_problem_intent` (answer.py:512) -> Complaint; `_account_intent` (answer.py:710) -> Care; `_shopping_intent` (answer.py:496)/gift -> Stylist; `heuristic_route` (understand.py:43) factual -> Answers; the server-side escalation phrase (moved out of `ChatWidget.tsx` ~638-676) -> Tiffany.
- Layer 2 (8B tie-break, ~15-20%): one JSON call to `make_small_llm()` (factory.py:36), `max_tokens ~16`, fired only on ambiguity or an `and|also|then|plus` conjunction. Any failure falls back to the Layer-1 best guess. Routing never blocks a turn.

Fast/slow boundary is deterministic-certain streams, model-arbitrated buffers. A turn resolved by Layer 0 or a single clean Layer-1 regime streams. Anything that needed the Layer-2 8B tie-break buffers, because a misrouted streamed answer is unrecoverable while a misrouted buffered answer costs one re-dispatch. This caps the only unrecoverable error class at `$0` and makes false-fast-path rate measurable on the labeled fixture. `FAST_FLOOR` is the confidence/regime gate; the labeled ambiguous/complaint fixture must hit its Layer-1 precision target before the fast path streams by default.

Multi-task planning uses LangGraph `Send` map-reduce inside the heavy subgraph. `plan` splits deterministically on `and|also|then|plus` (cap 3) and fans out one `Send` per clause to its lane node; `stitch` reduces. Ordering felt as copy: hurt before help (any complaint clause leads), and the clause needing the shopper's next input goes last. The heavy path emits an instant deterministic ack token ("On it for both of those:") as the start of one progressive bubble, so it never feels slower. `stitch` is a deterministic two-slot template first; it spends one 8B call only when clauses share a referent or contradict.

Re-route is a tier bump before a lane switch, `reroute_budget = 1`. Reuse `decide_tier` (agent.py:40), `_MAX_STEPS=2` (agent.py:32), `_MIN_AGENT_CONFIDENCE=0.05` (agent.py:31) verbatim. If a lane misses its required artifact but the router still says the lane is right, re-dispatch the same lane one tier up (8B -> 70B). Switch to the second-choice lane only when confidence says the lane itself was wrong. The shopper sees only the corrected answer.

Clarify is a three-way deterministic decision rendered from a data structure, never an LLM paragraph:

- PROCEED when one lane is live at high confidence or every clause routes unambiguously.
- GUESS-AND-LABEL when two readings share tone and data class and a miss costs one sentence: act on the likelier, name the assumption with an easy out.
- CLARIFY (a two-option fork plus a `Both` chip) only on one of three tests: F1 tone fork (warm-sell vs empathy vs precision differ; Stylist-vs-Complaint is canonical), F2 data fork (a referent resolves to two data classes, catalog vs order), F3 consequence fork (the next action is PII-adjacent or irreversible and the target is ambiguous).
- `Clarify = {axis, a:(lane,label), b:(lane,label)}` on the blackboard, rendered with a fixed template so the fork cannot hallucinate a third option. `last_clarify_axis` prevents asking the same axis twice; an ignored fork dies silently. Never clarify tone, never emit an open "can you rephrase." Promotion gate: fork precision >= 80%, fork rate <= 5% on the unambiguous golden set.

## 3. Inter-agent communication and shared state

Brokered blackboard, three mechanisms, zero free-form agent chat.

Blackboard is `ChatState` in `rag/state.py` (already `TypedDict, total=False`, so additions are non-breaking). Add, with a one-writer-per-key ownership table: `role`, `plan`, `step_idx`, `agent_outputs`, `clarify`, `escalate`, `escalation_id`, `persona ("sara"|"tiffany")`, `reroute_budget`, `last_clarify_axis`, `auth_identity`, `auth_text`, `handoff (CaseFile)`, `tier`, `cost_by_tier`. Router writes `role`/`plan`; each lane writes only its slot in `agent_outputs`; `stitch` reads all and writes `answer`. Do not add `WearerProfile`/`threads` on day 1; profile already flows through `_personal_profile_note` (answer.py:859).

Request tools live in a new `rag/tools.py` named registry that rehouses today's `retriever_finding`/`metrics_finding`/`graph_finding` (rag/specialists.py) as: `get_profile`, `lookup_order_status`, `get_governed_metric`, `get_policy`, `graph_facts`. "Sub-agent A asks sub-agent B" is A calling B's request tool (Complaint calling `lookup_order_status` through the gate before any remedy sentence), never a message to a B agent. Retrieve once per turn within a lane.

Sanity check is `detect_conflict` (supervisor.py:105) plus `reconcile` (supervisor.py:129): governed metric and graph findings keep `authoritative=True` and outrank text at merge, so a chatty Stylist "$40" against the governed price is corrected before stitch at zero model cost. Accepted v1 limit: numeric-only. A semantic 8B judge is scoped later, never a 70B judge.

Inside the heavy graph, control transfer is `Command(goto=<lane>, update={...})`, co-locating destination and blackboard write. The pre-graph `route()` stays a pure `RouteDecision` so it is testable without a graph.

## 4. Persona plan: Aria -> Sara everywhere, plus Tiffany

The rename is engine-free plus client copy. Current state confirmed: `domains/apparel_ecommerce/domain.yaml` line 12 `assistant: Aria`, line 13 `specialist: Sara`. `_persona()` (answer.py:39) reads these and every prompt is templated, so the engine needs no change.

Manifest (`domains/apparel_ecommerce/domain.yaml`):
- `assistant: Aria` -> `assistant: Sara`
- `specialist: Sara` -> `specialist: Tiffany`
- Add an optional `cast:` block (read only by client and router): each persona's `avatar_shape` (Sara round/warm, Tiffany squircle/cool), `model_tier`, `voice_id`. `_persona()` ignores unknown keys, so this is additive.

Prompts (`pipeline/answer.py`): no code change needed for names (`_system` answer.py:214, `_agent_system` answer.py:220, `_smalltalk` answer.py:225 all pull from `_persona`). One copy change: `_AGENT_SYSTEM_TMPL` (answer.py:159) shifts from a general "take ownership, resolve the handoff" voice to Tiffany's file-confirm-close-handback contract, including the honesty sentence one: "I'm Tiffany, the AI coordinator for Aster's care team, and a real person reviews everything I file."

Voice: already distinct in config (`adapters/config.py`): `elevenlabs_voice_id` (Sarah, warm young woman) for the assistant, `elevenlabs_agent_voice_id` (Jessica) for the specialist. Keep both; the rename is copy only. `api/app.py:554` already selects the agent voice when `persona == "agent"`.

Frontend (`web/app/ChatWidget.tsx`, 1500 lines) copy that must move:
- `AGENT_INTRO` (lines 39-40): "Hi, I'm Sara, Aster's AI care specialist..." -> Tiffany's honesty-frame intro.
- `endAgent` string (line 855): "You're back with Aria..." -> "back with Sara".
- greeting (lines 1192-1194): "Sara here" -> "Tiffany here"; "I'm Aria" -> "I'm Sara".
- header (line 1340) "Sara . Aster care specialist" -> "Tiffany . Aster care specialist"; "Back to Aria" (line 1342) -> "Back to Sara".
- intro line (line 1364) "I'm Aria" -> "I'm Sara".
- agent-tag (line 1394) "Sara . care specialist" -> "Tiffany . care specialist".
- escalate-btn (lines 1431-1432) label stays "Talk to a human agent".
- The client-side escalation trigger regex (lines ~638-676) moves server-side into `route()` so it is testable and not client-bypassable; the widget then just posts the turn and honors a `persona: "tiffany"` flag returned by the server.

Avatar (`web/app/Avatar.tsx`): today one inline-SVG avatar with a green gradient. Add a `persona` prop that switches Tiffany to a squircle clip-path and a cool palette (keep Sara round/warm). This is hand-coded SVG and needs no Figma. See the connector caveat at the end for the polished-asset path.

Hand-back close: Tiffany files via `ReviewQueue.enqueue` (rag/hitl.py:51), displays the case id as `"AST-" + item_id[:6].upper()`, commits to email follow-up, asks "any other concerns", thanks, and the server flips `persona` back to `sara`. The flip is a real server-side state transition mirrored to the client so a refresh mid-handoff stays consistent (the widget already has a comment at lines 527-528 about restoring agent mode on refresh; wire it to server `persona`). Sara later answers "any update on my case?" from `ReviewQueue.get` under the same identity gate as orders.

## 5. Keeping the deterministic guards, metrics, graph, and fast path authoritative

- The PII gate does not move. Enforcement lives inside `retrieve()` (answer.py:1121, the `auth`/`_order_access_ok` block around 1152 and `_order_access_ok` at 764): it requires BOTH the account email and the full independent name phrase in the shopper's own words and drops the order doc before it reaches the prompt; `_THIRD_PARTY`/`_THIRD_PARTY_NAME` (answer.py:696, 705) strip third-party lookups. Every lane reaches evidence through the request tools, which reach it through `retrieve()`, so the gate is shared with zero code movement. Only the classifiers move: extract `_smalltalk` intercepts, `_problem_intent`, `_account_intent`, `_shopping_intent`, `heuristic_route` into a new `rag/guards.py`, imported by both `stream_answer` and `route()`. Safety/PII tests must be green before and after in a no-logic-change commit.
- PII parity is free on the fast path (inherited from `stream_answer`'s `auth_text`, answer.py:1351) and must be threaded on the heavy path via `state["auth_text"]` into every lane's `retrieve()` call, closing the documented gap where `answer_with_agent`/`retriever_finding` call `retrieve()` with no `auth_text`. Gate the whole omni promotion on a red-team fixture (name-then-email unlock, signed-in auto-unlock, third-party name probe, third-party email probe, anonymous probe) that stays zero-leak across linear and omni identically and never confirms an order exists pre-verification.
- The new leak surface is stitch. The linear path never composed multiple drafts, so `output_guard` runs on every stitched or re-routed answer: on an unauthorized order-doc PII token (reuse `_ORDER_TERM`, answer.py:687) it replaces the answer with the gate refusal and logs the block, on top of `retrieve()` dropping unauthorized docs pre-prompt.
- Governed metrics and graph stay authoritative through `reconcile` merge ordering (supervisor.py:81-102) and `detect_conflict` (supervisor.py:105): `authoritative=True` findings outrank text; the numeric-conflict fallback ships the governed value verbatim (supervisor.py:149-164).
- The linear fast path is not a separate path. It stays the standalone `CHAT_BRAIN=linear` entry (config.py:109) for the storefront until omni is promoted, and it is the body of the fast-path serve step. Add exactly one optional param to `stream_answer` (answer.py:1311): `role_fragment: str | None = None`, appended after the `_VOICE_BREVITY` block (answer.py:1332-1333); default `None` reproduces today byte-for-byte.

## 6. Phased implementation (make check green each step; promote only on the eval gate)

Each phase is one shippable `M.dot` step under the `ship` skill (branch, build, make check, independent review, chunked commits, no-ff merge).

- omni.0 (cost observability, no flag). Thread `make_small_llm()` (factory.py:36) into `route_metric` (retrieval/metric_router) and `rewrite_followup` (understand.py:67) so classification/rewrite stop paying 70B. Extend `_PRICES` (answer.py:469) with `claude-sonnet-5:(3.00,15.00)`, `claude-opus-4-8:(5.00,25.00)`, `claude-haiku-4-5:(1.00,5.00)`. Teach `_estimate_cost` (answer.py:663) to price `cache_read_input_tokens` at 0.1x and `cache_creation_input_tokens` at 1.25x (today it returns `None` for every Anthropic model, so Tiffany cost silently vanishes). Add `tier` and `cost_by_tier` to the trace dict. Files: `adapters/factory.py`, `retrieval/metric_router.py`, `rag/understand.py`, `pipeline/answer.py`, `rag/supervisor.py`, `rag/agent.py`, tests.
- omni.1 (pure copy). Persona rename in `domains/apparel_ecommerce/domain.yaml`; add `cast:` block; the `ChatWidget.tsx` string moves listed in section 4; the `_AGENT_SYSTEM_TMPL` copy change. Persona/safety/personalization tests. Files: `domain.yaml`, `web/app/ChatWidget.tsx`, `pipeline/answer.py`.
- omni.2 (extract guards, no logic change). New `rag/guards.py` holding the `_smalltalk` intercepts plus the four classifiers; import from both brains. PII enforcement stays in `retrieve()`. Safety/PII tests green before and after; run the red-team fixture on the extracted module in isolation. Files: `rag/guards.py`, `pipeline/answer.py`, `rag/understand.py`, tests.
- omni.3 (router + tools, not serving yet). New `rag/router.py` (`route() -> RouteDecision`, three-layer cascade, server-side escalation regex) and `rag/tools.py` (named registry rehousing the specialist findings). Unit-test the routing table on a labeled fixture (`domains/apparel_ecommerce/eval/routing.jsonl`). Files: `rag/router.py`, `rag/tools.py`, `rag/specialists.py`, tests.
- omni.4 (fast path + single-lane specialization + PII parity). Add `role_fragment` to `stream_answer` (default identical). New `rag/roles.py` (the five lane rows). New `helpers/dates.py` (deterministic date math note, load-bearing for Care and Complaint rung-2). Lean 8B lane prompts for Answers/simple Care, gated on the per-lane scorecard beating the 70B golden baseline. Add `CHAT_BRAIN=omni` single-task-only. Optional omni.4b: `_prepare_answer`/`_stream_from` split for pre-stream re-route. Files: `pipeline/answer.py`, `rag/roles.py`, `helpers/dates.py`, `api/app.py`, `adapters/config.py`, tests.
- omni.5 (heavy path). `input_guard`/`plan` with `Send` fan-out (cap 3); tier-bump-then-lane-switch re-route (`reroute_budget=1`); deterministic-template `stitch` with the ack token; `output_guard` tripwire; the blackboard ownership table; the heavy-path `auth_text` parity fix with its red-team fixture. Files: new `rag/omni_graph.py` (or extend `rag/graph.py`), `rag/state.py`, `rag/tools.py`, tests.
- omni.6 (Tiffany). New `adapters/anthropic.py` `make_frontier_llm()` behind the existing `LLMClient` protocol (`adapters/base.py`; must implement `.generate(prompt, system=, max_tokens=)` returning `.text/.model/.prompt_tokens/.completion_tokens` and `.stream(prompt, system=)`), reading `FRONTIER_MODEL`, defaulting to 70B when unconfigured so a keyless offline checkout runs make check green. Wire: Sonnet 5 default with `thinking:{type:"disabled"}`, no sampling params, a >=4096-token padded cache prefix on `system`, volatile CaseFile facts in a trailing user-turn `<system-reminder>`, refusal check (`stop_reason` before `content[0]`), degrade-to-70B on refusal/429/5xx with a deterministic close; Opus 4.8 HARD_RESOLUTION path with `thinking:{type:"adaptive"}`, `output_config.effort:"high"`, and `role:system` for the operator channel; Haiku 4.5 confirm-back floor with thinking off (effort unsupported). CaseFile -> `ReviewQueue.enqueue`; server-side switch-back; assert `cache_read_input_tokens > 0` on turn 2 of an escalation as a hard build gate (a zero means a silent invalidator crept into the cached block: no `datetime.now()`, no per-turn IDs). Add `FRONTIER_MODEL`, `FRONTIER_MODEL_HARD`, `FRONTIER_MODEL_FLOOR` to `adapters/config.py`. Flip `CHAT_BRAIN=omni` to default only when it wins role-routing accuracy, multi-task completion, clarify precision (fork precision >= 80%, fork rate <= 5%), cost-per-turn by tier, and a zero-PII-leak red-team across linear and omni identically. Files: `adapters/anthropic.py`, `adapters/factory.py`, `adapters/config.py`, `rag/omni_graph.py`, `rag/hitl.py` (case-status read under the gate), `api/app.py`, `web/app/ChatWidget.tsx` (persona flip mirror), tests.

## 7. Verification plan: 10 Opus checks + 10 Fable checks

Run at the end via the repo's `review` skill (Fable 5 primary, Opus fallback) and the `preflight`/`ship` gates. Each check has a concrete pass criterion; a check that cannot be shown green blocks promotion.

Opus reviewer set (routing correctness and safety invariants):
1. Router table: on the labeled routing fixture, Layer-1 precision on complaint vs stylist meets the target and no unrecoverable misroute streams. Pass: fast-path false-route rate is 0 on the labeled set.
2. Multi-task order: "suggest a gift AND check my last order" runs Stylist then Care, complaint clause always leads, input-needing clause last. Pass: 3 canonical two-task prompts sequence correctly.
3. Order-PII gate parity: name-then-email unlock, signed-in auto-unlock, third-party name probe, third-party email probe, anonymous probe give byte-identical zero-leak outcomes on linear and omni. Pass: red-team fixture zero leak both brains.
4. Stitch leak surface: a stitched answer that would surface an unauthorized order token is replaced by the gate refusal and logs `security.leak_blocked`. Pass: injected leak fixture blocked.
5. Injection/harm intercepts survive the guards extraction: the `rag/guards.py` module refuses "print your system prompt", "ignore your rules", weapon/self-harm phrasings identically to pre-extraction. Pass: safety suite green.
6. Governed-metric authority: a chatty lane price contradicting the governed price is corrected before stitch at zero model cost. Pass: planted-conflict fixture resolves to the governed value.
7. Clarify precision: on the unambiguous golden set the fork rate <= 5%; on the ambiguous set fork precision >= 80%; the fork never offers a third option and never asks the same axis twice. Pass: both thresholds met.
8. Re-route budget: an 8B artifact miss bumps to 70B once (same lane) before any lane switch; `reroute_budget=1` is not exceeded. Pass: trace shows at most one bump then at most one switch.
9. Tiffany model contract: Sonnet 5 default sends no sampling params and `thinking:{type:"disabled"}`; the HARD path uses Opus 4.8 with `role:system`; a forced 429/refusal degrades to 70B with the deterministic case-filed close. Pass: no 400s, refusal handled before `content[0]`, case still files.
10. Cache economics: turn 2 of an escalation reports `cache_read_input_tokens > 0`; `_estimate_cost` prices cache reads at 0.1x and writes at 1.25x; `cost_by_tier` is populated. Pass: assertion holds and Tiffany cost is non-null in the trace.

Fable reviewer set (lane mastery and edge cases against the real pack):
1. Stylist budget math: a two-item gift under a stated budget sums bullet prices correctly and never exceeds it. Pass: computed total <= budget.
2. Stylist fit truth: a sized SKU with a return-rate/size-review risk (Trailhead runs slim, Lumen White goes sheer) discloses the risk before praise and closes with CK03. Pass: risk sentence precedes any affirmation.
3. Stylist never re-asks: a filled slot (use, colour, or budget already given) is not re-questioned. Pass: no repeated clarifier.
4. Care three-part verdict: an order status renders status + date math (from `helpers/dates.py`, never model arithmetic) + next step, not an echo. Pass: all three parts present, dates match the deterministic note.
5. Care self-serve deadline: "can I return this" converts to the self-serve path plus its CK deadline. Pass: deadline is the policy value, not invented.
6. Complaint first sentence: names the specific item and failure; no product pitch in the thread; zero banned phrases ("I understand your frustration", "unfortunately", "as per our policy"); zero emoji. Pass: deterministic post-check clean.
7. Complaint rung-2: a policy-guaranteed remedy states amount + computed date + citation (CK04/CK45/CK02/CK03/CK13/CK52). Pass: all three fields present and cited.
8. Answers lane de-dilution: the lean ~700-token 8B prompt clears the Answers/simple-Care scorecard against the 70B golden baseline; if it fails, the lane falls back to 70B and the cost claim is withdrawn. Pass: per-lane scorecard >= 70B baseline or documented fallback.
9. Tiffany hand-back: gathers, confirms a numbered list of only stated facts, files the CaseFile (`AST-XXXXXX`), states email follow-up, asks "any other concerns", thanks, and the server flips `persona` back to Sara; a refresh mid-handoff stays consistent. Pass: full sequence and consistent rehydration.
10. Tiffany gate obedience and case status: Tiffany confirms stated facts but never reads order records aloud unless name+email match or signed in; Sara later answers "any update on my case?" from `ReviewQueue.get` under the same gate. Pass: zero unverified record disclosure.

---

Caveat (session limitation, per the environment): the claude.ai Figma connector is unauthorized in this session, so any Figma-backed avatar asset work (round-vs-squircle shapes, warm-vs-cool palettes for Sara and Tiffany) cannot be produced here. It needs authorizing via your claude.ai connector settings before any Figma-backed design step. The plan does not depend on it: `web/app/Avatar.tsx` is hand-coded inline SVG, so the Tiffany squircle/cool variant can be built directly in code (omni.1/omni.6) without Figma; the connector is only needed if you want the polished asset pipeline.

Key files to build from (absolute): `/Users/esteki/Desktop/MDS/Projects/agentic-rag-knowledge-ai-platform/pipeline/answer.py`, `.../rag/router.py` (new), `.../rag/tools.py` (new), `.../rag/roles.py` (new), `.../rag/guards.py` (new), `.../rag/omni_graph.py` (new), `.../helpers/dates.py` (new), `.../adapters/anthropic.py` (new), `.../rag/state.py`, `.../rag/supervisor.py`, `.../rag/agent.py`, `.../rag/hitl.py`, `.../adapters/factory.py`, `.../adapters/config.py`, `.../domains/apparel_ecommerce/domain.yaml`, `.../web/app/ChatWidget.tsx`, `.../web/app/Avatar.tsx`, `.../api/app.py`.