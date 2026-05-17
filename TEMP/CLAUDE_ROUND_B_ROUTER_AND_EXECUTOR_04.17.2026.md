# Round B — Router Resolution, UNKNOWN Guard, ARC Executor Reality Check

**Author:** Claude (Architecture Partner)
**Date:** 2026-04-17
**For:** Codex
**Predecessors:** `CLAUDE_VALIDATION_SWEEP_50x_04.17.2026.md`, `CLAUDE_SWEEP_SURVIVABILITY_04.17.2026.md`
**Scope:** Three narrow fixes. No embodiment work. No score-chasing. Each deliverable ships with a targeted test; the sweep gets re-run at the end.

---

## 1. The actual §2.2 picture (what the letter of the spec missed)

SUMMARY.md marks §2.2 `meaning_class_spread` green because 4 distinct argmax classes fire. The union across 239 routed items:

| Class | Count | Share |
|---|---:|---:|
| NUMERIC_COMPUTE | 143 | 60% |
| SPATIAL_TRANSFORM | 49 | 20% |
| FACTUAL_RECALL | 32 | 13% |
| UNKNOWN | 11 | 5% |
| GROUNDED_DIALOG | 1 | <1% |
| DEFINITION_LOOKUP | 0 | — |
| MULTI_HOP_INFERENCE | **0** | — |
| COMPARATIVE_CHOICE | 0 | — |
| GENERATIVE_COMPOSITION | 0 | — |

Letter green, spirit red. The header numbers to hold next to this when you read the fixes below:

- **MULTI_HOP_INFERENCE fires zero times** on 150 unambiguously multi-hop items (GSM8K + Math competitions + LHE). The eight-class head is effectively a four-class head.
- **21 of 50 MMLU items (42%) land in NUMERIC_COMPUTE** — law, security, econ-reading questions. Any numeral in the prompt (a year, a percentage reference, a count in an option) pulls them into the math lane.
- **11 items classify as UNKNOWN** and nine of those ten LHE UNKNOWNs converted into `wall_timeout`. UNKNOWN → ring has no star to fire → ring sits → timeout fires. That's the router using the survivability cap as a crutch.
- **ARC-AGI-1: 49 of 50 correctly routed to SPATIAL_TRANSFORM, 0 correct.** Median latency 1.2 s — nine-chain swarm isn't even running. The executor emits a grid of the right shape filled entirely with zeros (confirmed by reading `sampled_outputs[0].task_result.task_result.output_grid`). This is a stub presenting as a real star.

Round B addresses three of those four. The MMLU numeral-capture issue gets absorbed into (a); the other three become their own items.

---

## 2. Round B — three deliverables

### B.1 UNKNOWN meaning class must not enter the ring (blocker)

**File:** [knowledge3d/knowledgeverse/navigator_specialist.py](../knowledge3d/knowledgeverse/navigator_specialist.py) + wherever `meaning_class` is resolved from the 8-logit head.

Today: navigator emits `meaning_class_dist[8]` softmax. When no class exceeds a confidence floor (or when the 8-logit head produces flat noise), downstream code records the argmax as `UNKNOWN` and still calls `enqueue_task`. The ring has no star to run, ticks while waiting, trips `max_wall_ms`.

**Rule:** a routed task ALWAYS carries one of the 8 defined classes. UNKNOWN is not a class, it's a symptom. Two changes:

1. At the point `meaning_class` is assigned from the 8-logit argmax, if max-softmax < `MEANING_CLASS_CONFIDENCE_FLOOR` (start at 0.15 — anything lower than 1/8 + 3% is noise), fall back to `FACTUAL_RECALL` as a defined default, and set a `low_confidence_routing=True` flag on the envelope. FACTUAL_RECALL is the right default because its executor chain (`question_answer_validator` / elimination) is the cheapest and degrades gracefully — it won't spin on impossible word problems the way the math lane does.
2. Never pass the string `"UNKNOWN"` through the envelope. If the fallback triggers, the outbound `meaning_class` is `FACTUAL_RECALL` with `low_confidence_routing=True` visible to the ring trace and to sleep-time.

**Sleep-time consequence:** `low_confidence_routing=True` items become the priority training set for the navigator on the next consolidation wave. That's the legitimate way to teach the 8-class head to cover what it currently can't — not by widening the router's exception surface.

**Test:** `tests/knowledgeverse/test_unknown_class_guard.py` — stub the navigator to return flat 1/8 softmax, enqueue a task, assert outbound envelope has `meaning_class == "FACTUAL_RECALL"` and `low_confidence_routing is True`. Assert `UNKNOWN` never appears in the routed task.

### B.2 ARC-AGI-1 executor reality check (diagnostic → fix in one pass)

**Routed correctly (49/50 SPATIAL_TRANSFORM), executed empty (0/50).** The spatial executor emits a zero-filled grid of correct dimensions. That's not a solving error — that's a "the star body isn't actually composing a transform."

Two-step:

**Step 1 (investigate, before any code):** from `ring_trace.jsonl` pick 3 ARC items with completed outputs. For each, read the full `sampled_outputs` entry (or re-run the item through the daemon) and report back to me:
- which `executor_star` was selected
- what that star's RPN body looks like (read the Galaxy entry for `spatial_transform_*` executor-star IDs)
- whether the nine-chain swarm (per `trace_roles`, `trace_role_ids`) is actually engaged, or a single-role executor runs
- whether the frustum / Morton / LED-A* pipeline was actually traversed (any trace stage with non-trivial tick count) — or if the star returned before touching those kernels

**Step 2 (based on what Step 1 finds):**
- If the star is a placeholder (expected) — propose which of the existing 88 PTX kernels is the real transform executor and wire that star's body to compose them. Do not write new kernels. Use the ones we already have.
- If the star is real but the nine-chain swarm isn't engaged — that's a dispatch bug in `AdaptiveSwarmTRM`. Fix the dispatcher, not the star.
- If the swarm IS engaged and still produces zeros — inspect the training state of the spatial lane (shadow copy, adapter weights). Sleep-time may have never consolidated signal on ARC input shapes.

Do Step 1 first and send me the three-task report before writing any code for Step 2. This is the one place in Round B where I want to see the evidence before I spec the fix.

### B.3 MULTI_HOP_INFERENCE must be reachable — seed contrast test

**Observation:** across 150 multi-hop items MULTI_HOP_INFERENCE fires zero times. Either the seed embeddings for MULTI_HOP vs NUMERIC_COMPUTE overlap so completely that NUMERIC_COMPUTE always wins, or the feature extractor the navigator consumes never includes signal that could distinguish them.

**Deliverable:** a diagnostic test, not a retrain. `tests/knowledgeverse/test_meaning_class_multi_hop_separability.py`:

1. Build five hand-crafted multi-hop prompts with numbers (GSM8K-shape) and five hand-crafted direct-compute prompts (`"2+3?"`, `"sqrt(16)"`, etc.).
2. Call the navigator on all ten.
3. Assert: on at least 4 of the 5 multi-hop prompts, `MULTI_HOP_INFERENCE` softmax > `NUMERIC_COMPUTE` softmax.

This test **will fail** today — that is the point. The failing test tells us the gap in one number. Do not try to make it pass this round by editing seeds or retraining. We need a failing test as the anchor for the eventual sleep-time re-training target.

After the test lands and fails, write a short note appended to SUMMARY.md under `### Round B.3 — multi-hop separability` with: (a) actual softmax values on the 10 prompts, (b) one sentence on what the distribution looks like. That's the data I need to decide whether seed re-training is a sleep-time job, a sovereign PTX job, or (most likely) a consequence of the House being empty of multi-hop structural anchors — which circles back to embodiment.

---

## 3. Re-run

After B.1 + B.2 (both steps) + B.3 land:

1. `pytest -q tests/knowledgeverse/test_unknown_class_guard.py tests/knowledgeverse/test_meaning_class_multi_hop_separability.py` — B.1 green, B.3 red (expected).
2. `python scripts/validation_sweep_20260417.py` end-to-end.
3. Compare to Round A numbers:
   - LHE: wall_timeouts should drop sharply (the 10 UNKNOWN-driven ones shouldn't happen anymore). Score likely flat — that's fine.
   - ARC: if B.2 Step 2 wires real executor composition, score leaves zero. If B.2 was diagnostic-only this round (Step 1 found the problem but Step 2 defers), score stays 0 — also fine, we have the finding.
   - Meaning class distribution: UNKNOWN → 0. FACTUAL_RECALL should grow (absorbing the old UNKNOWNs). MULTI_HOP_INFERENCE still 0 — B.3 doesn't change routing, only tests separability.
4. Append to SUMMARY.md: a **Round B delta** section with before/after class counts, before/after wall_timeouts, ARC executor verdict from B.2 Step 1, and B.3 softmax values.

---

## 4. Decision gate after Round B

Three possible next moves. Pick based on the Round B delta:

1. **Numbers worth pursuing via more router work** — spec Round C (seed retraining via sleep-time, MMLU numeral-capture fix, GSM8K decomposition chain). Costs days, diminishing returns per the memory: "don't tune navigation over empty shelves."
2. **Numbers flat, but engine stays survivable and honest** — pivot to embodiment (gaps 1+2+3 minimum viable: perceive, act, House↔Galaxy symlinks). This is the path the House-First pivot memory argues for: knowledge density from embodiment re-shapes routing by itself.
3. **ARC executor turned out to be a real star with broken composition** — small Round C-lite to fix the specific PTX composition bug, then pivot to embodiment.

Daniel makes the call after reading the Round B delta. Not my call.

---

## 5. What NOT to do this round

- No retraining. No seed edits. No sleep-time hyperparameter changes.
- No new executor stars, no new validator stars, no new meaning classes (do NOT add a 9th class to "handle" UNKNOWN — UNKNOWN is a symptom, not a signal).
- No changes to `sovereign_hot_path.py`.
- No changes to the ring or TickDriver — Round A landed those and they're stable.
- No changes to senders. The problems are upstream of the senders.
- No attempt to lift MMLU above 25% this round. The MMLU ceiling is the NUMERIC_COMPUTE collapse issue and B.3 is scoped to *measure* the separability, not fix it.

---

## 6. Standing protocol reminders

- Rule of three: `qdrant-find` for specs first, `k3d-ptx qdrant-find` before any kernel-touching or ctypes change (B.2 Step 2 qualifies), `plan_task` cloud before B.2 Step 2 implementation.
- `kimi_swarm` / deep `ask_cloud` timeout = 240000 ms.
- Tests hit the real daemon. No mocks for the ring. No numpy in sovereign code.

---

## 7. Acceptance

Round B passes when:
1. `test_unknown_class_guard` green.
2. `test_meaning_class_multi_hop_separability` lands and runs (red is OK, green is surprising — either way the softmax values are reported).
3. B.2 Step 1 report sent to me: three ARC items, executor star names, swarm engagement verdict, kernel traversal verdict.
4. Re-run sweep produces five fresh JSONs with `UNKNOWN == 0` in all of them.
5. SUMMARY.md has a `Round B delta` section with before/after class counts and ARC verdict.
6. `wc -l knowledge3d/knowledgeverse/knowledgeverse.py` stays ≤ 15969 (do not grow the file this round).
7. Janet = 18 at T0 and T_end.

Ship to me: all of the above, plus a 3-sentence "what surprised you" note on the B.2 Step 1 finding. That's the anchor for Daniel's decision gate.
